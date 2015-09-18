from ..core import Solver
from pycllp.lp import StandardLP
from pycllp.solvers import solver_registry
from pywr.core import BaseInput, BaseOutput, BaseLink
from pywr._core import *
import numpy as np

inf = np.inf
DBL_MAX = np.finfo(np.float32).max
DBL_MAX = np.inf

def dbl_max_to_inf(a):
    if a == DBL_MAX:
        return inf
    elif a == -DBL_MAX:
        return -inf
    return a

def inf_to_dbl_max(a):
    if a == inf:
        return DBL_MAX
    elif a == -inf:
        return -DBL_MAX
    return a

class PyCLLPSolver(Solver):
    """Python wrapper the interior point solvers in pycllp

    """
    name = 'pycllp'
    def __init__(self, *args, **kwargs):
        super(PyCLLPSolver, self).__init__(*args, **kwargs)
        # TODO add pycllp specific settings

    def setup(self, model):
        # This is the interior point method to use as provided by pycllp
        # TODO make this user configurable
        self._solver = solver_registry['dense_path_following']()


        routes = model.find_all_routes(BaseInput, BaseOutput, valid=(BaseLink, BaseInput, BaseOutput))
        # Find cross-domain routes
        cross_domain_routes = model.find_all_routes(BaseOutput, BaseInput, max_length=2, domain_match='different')

        supplys = []
        demands = []
        storages = []
        for some_node in model.nodes():
            if isinstance(some_node, (BaseInput, BaseLink)):
                supplys.append(some_node)
            if isinstance(some_node, BaseOutput):
                demands.append(some_node)
            if isinstance(some_node, Storage):
                storages.append(some_node)

        assert(routes)
        assert(supplys)
        assert(demands)

        # Create new blank problem
        self.lp = lp = StandardLP()
        # first column id for routes
        self.idx_col_routes = 0
        # first column id for demands
        self.idx_col_demands = len(routes)

        # create a lookup for the cross-domain routes.
        cross_domain_cols = {}
        for cross_domain_route in cross_domain_routes:
            # These routes are only 2 nodes. From demand to supply
            demand, supply = cross_domain_route
            # TODO make this time varying.
            conv_factor = supply.get_conversion_factor()
            supply_cols = [(n, conv_factor) for n, route in enumerate(routes) if route[0] is supply]
            # create easy lookup for the route columns this demand might
            # provide cross-domain connection to
            if demand in cross_domain_cols:
                cross_domain_cols[demand].extend(supply_cols)
            else:
                cross_domain_cols[demand] = supply_cols

        # constrain supply minimum and maximum flow
        self.idx_row_supplys = lp.nrows
        for col, supply in enumerate(supplys):
            # TODO is this a bit hackish??
            if isinstance(supply, BaseInput):
                cols = [n for n, route in enumerate(routes) if route[0] is supply]
            else:
                cols = [n for n, route in enumerate(routes) if supply in route]
            ind = np.zeros(len(cols), dtype=np.int)
            val = np.zeros(len(cols), dtype=np.float64)
            for n, c in enumerate(cols):
                ind[n] = c
                val[n] = 1
            lp.add_row(ind, val, np.inf)
            lp.add_row(ind, -val, 0.0)

        # link supply and demand variables
        self.idx_row_demands = lp.nrows
        for col, demand in enumerate(demands):
            cols = [n for n, route in enumerate(routes) if route[-1] is demand]
            ind = np.zeros(len(cols)+1, dtype=np.int)
            val = np.zeros(len(cols)+1, dtype=np.float64)
            for n, c in enumerate(cols):
                ind[n] = c
                val[n] = 1
            ind[len(cols)] = self.idx_col_demands+col
            val[len(cols)] = -1
            # This constraint is added twice to enforce it as an equality
            lp.add_row(ind, val, 0.0)
            lp.add_row(ind, -val, 0.0)
            # Now add additional constraints for demand bounds
            # The standard form of an LP enforces x >= 0, but this
            # potentially needs constraining
            lp.add_row([self.idx_col_demands+col], [1.0], np.inf)
            lp.add_row([self.idx_col_demands+col], [-1.0], 0.0)


        if len(cross_domain_cols) > 0:
            self.idx_row_cross_domain = lp.nrows
        cross_domain_col = 0
        for col, demand in enumerate(demands):
            # Add constraint for cross-domain routes
            # i.e. those from a demand to a supply
            if demand in cross_domain_cols:
                col_vals = cross_domain_cols[demand]
                ind = np.zeros(len(col_vals)+1, dtype=np.int)
                val = np.zeros(len(col_vals)+1, dtype=np.float64)
                for n, (c, v) in enumerate(col_vals):
                    ind[n] = c
                    val[n] = 1./v
                ind[len(col_vals)] = self.idx_col_demands+col
                val[len(col_vals)] = -1
                lp.add_row(ind, val, 0.0)
                lp.add_row(ind, -val, 0.0)
                cross_domain_col += 1

        # storage
        if len(storages):
            self.idx_row_storages = lp.nrows
        for col, storage in enumerate(storages):
            cols_output = [n for n, demand in enumerate(demands) if demand in storage.outputs]
            cols_input = [n for n, route in enumerate(routes) if route[0] in storage.inputs]
            ind = np.zeros(len(cols_output)+len(cols_input), dtype=np.int)
            val = np.zeros(len(cols_output)+len(cols_input), dtype=np.float64)
            for n, c in enumerate(cols_output):
                ind[n] = self.idx_col_demands+c
                val[n] = 1
            for n, c in enumerate(cols_input):
                ind[len(cols_output)+n] = self.idx_col_routes+c
                val[len(cols_output)+n] = -1
            # Two rows needed again for the range constraint on change in storage volume
            lp.add_row(ind, val, 0.0)
            lp.add_row(ind, -val, 0.0)

        self.routes = routes
        self.supplys = supplys
        self.demands = demands
        self.storages = storages

        # Now expand the constraints and objective function to multiple
        lp.set_num_problems(len(model.scenarios.combinations))

    def solve(self, model):
        lp = self.lp
        timestep = model.timestep
        routes = self.routes
        supplys = self.supplys
        demands = self.demands
        storages = self.storages

        for scenario_id, scenario_indices in enumerate(model.scenarios.combinations):
            # update route properties
            for col, route in enumerate(routes):
                cost = 0.0
                for node in route:
                    cost += node.get_cost(timestep, scenario_indices)
                lp.set_objective(self.idx_col_routes+col, -cost)

            # update supply properties
            for col, supply in enumerate(supplys):
                min_flow = inf_to_dbl_max(supply.get_min_flow(timestep, scenario_indices))
                max_flow = inf_to_dbl_max(supply.get_max_flow(timestep, scenario_indices))
                lp.set_bound(self.idx_row_supplys+col*2, max_flow)
                lp.set_bound(self.idx_row_supplys+col*2+1, -min_flow)

            # update demand properties
            for col, demand in enumerate(demands):
                min_flow = inf_to_dbl_max(demand.get_min_flow(timestep, scenario_indices))
                max_flow = inf_to_dbl_max(demand.get_max_flow(timestep, scenario_indices))
                if min_flow < 0.0:
                    raise ValueError("Minimum flow less than zero not supported by this solver.")
                cost = demand.get_cost(timestep, scenario_indices)
                lp.set_bound(self.idx_row_demands+col*4+2, max_flow)
                lp.set_bound(self.idx_row_demands+col*4+3, -min_flow)
                lp.set_objective(self.idx_col_demands+col, -cost)

            # update storage node constraint
            for col, storage in enumerate(storages):
                max_volume = storage.get_max_volume(timestep, scenario_indices)
                avail_volume = max(storage._volume[scenario_id] - storage.get_min_volume(timestep, scenario_indices), 0.0)
                # change in storage cannot be more than the current volume or
                # result in maximum volume being exceeded
                lb = -avail_volume/timestep.days
                ub = (max_volume-storage._volume[scenario_id])/timestep.days
                lp.set_bound(self.idx_row_storages+col*2, ub)
                lp.set_bound(self.idx_row_storages+col*2+1, -lb)


        # Initialise the interior point solver.
        # remove constraints with infinite bounds
        # this is to improve numerical stability, but is potentially
        # a huge bottleneck.
        ublp = lp.remove_unbounded()
        ublp.init(self._solver)
        ublp.solve(self._solver, verbose=0)
        x = self._solver.x
        status = self._solver.status

        if status != 0:
            raise RuntimeError("Solver did find an optimal solution for at least one problems.")

        route_flow = np.round(x[0, self.idx_col_routes:self.idx_col_routes+len(routes)], 5)
        change_in_storage = []

        result = {}

        for route, flow in zip(routes, route_flow):
            for node in route:
                node.commit(scenario_id, flow)
