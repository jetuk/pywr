from . import Solver
from pycllp.lp import EqualityLP
from pycllp.solvers import solver_registry
from pywr.core import BaseInput, BaseOutput, BaseLink
from pywr._core import *
import numpy as np

inf = np.inf
DBL_MAX = 99999.0

def dbl_max_to_inf(a):
    a[np.where(a==DBL_MAX)] = np.inf
    a[np.where(a==-DBL_MAX)] = -np.inf
    return a

def inf_to_dbl_max(a):
    a[np.where(a>DBL_MAX)] = DBL_MAX
    a[np.where(a<-DBL_MAX)] = -DBL_MAX
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
        self._solver = solver_registry['dense_primal_normal']()

        routes = model.find_all_routes(BaseInput, BaseOutput, valid=(BaseLink, BaseInput, BaseOutput))
        # Find cross-domain routes
        cross_domain_routes = model.find_all_routes(BaseOutput, BaseInput, max_length=2, domain_match='different')

        non_storages = []
        storages = []
        for some_node in sorted(model.nodes(), key=lambda n: n.name):
            if isinstance(some_node, (BaseInput, BaseLink, BaseOutput)):
                non_storages.append(some_node)
            elif isinstance(some_node, Storage):
                storages.append(some_node)

        assert(routes)
        assert(non_storages)

        # Create new blank problem
        self.lp = lp = EqualityLP()
        # first column id for routes
        self.idx_col_routes = 0
        self.idx_col_slacks = self.idx_col_routes + len(routes)

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
        self.idx_row_non_storages_upper = np.empty(len(non_storages), dtype=np.int)
        self.idx_row_non_storages_lower = np.empty(len(non_storages), dtype=np.int)
        cross_domain_col = 0
        slack_col = 0
        for row, some_node in enumerate(non_storages):
            # Differentiate betwen the node type.
            # Input & Output only apply their flow constraints when they
            # are the first and last node on the route respectively.
            if isinstance(some_node, BaseInput):
                cols = [n for n, route in enumerate(routes) if route[0] is some_node]
            elif isinstance(some_node, BaseOutput):
                cols = [n for n, route in enumerate(routes) if route[-1] is some_node]
            else:
                # Other nodes apply their flow constraints to all routes passing through them
                cols = [n for n, route in enumerate(routes) if some_node in route]

            if len(cols) == 0:
                # Mark this node as having no row entries
                self.idx_row_non_storages_lower[row] = -1
                self.idx_row_non_storages_upper[row] = -1
                continue

            ind = np.zeros(len(cols)+1, dtype=np.int)
            val = np.zeros(len(cols)+1, dtype=np.float64)
            for n, c in enumerate(cols):
                ind[n] = c
                val[n] = 1

            if not some_node.is_max_flow_unbounded:
                # Add slack variable
                ind[-1] = self.idx_col_slacks + slack_col
                val[-1] = 1.0
                slack_col += 1
                # Only add upper bound if max_flow is finite
                lp.add_row(ind, val, DBL_MAX)
                self.idx_row_non_storages_upper[row] = lp.nrows - 1
            else:
                self.idx_row_non_storages_upper[row] = -1

            # Add slack variable
            ind[-1] = self.idx_col_slacks + slack_col
            val[-1] = -1.0
            slack_col += 1
            lp.add_row(ind, -val, 0.0)
            self.idx_row_non_storages_lower[row] = lp.nrows - 1
            # Add constraint for cross-domain routes
            # i.e. those from a demand to a supply
            if some_node in cross_domain_cols:
                col_vals = cross_domain_cols[some_node]
                ind = np.zeros(len(col_vals)+len(cols), dtype=np.int)
                val = np.zeros(len(col_vals)+len(cols), dtype=np.float64)
                for n, c in enumerate(cols):
                    ind[n] = c
                    val[n] = -1
                for n, (c, v) in enumerate(col_vals):
                    ind[n+len(cols)] = c
                    val[n+len(cols)] = 1./v
                # Equality LP only needs one row for this constraint
                lp.add_row(ind, val, 0.0)
                cross_domain_col += 1

        # storage
        if len(storages):
            self.idx_row_storages = lp.nrows
        for col, storage in enumerate(storages):
            cols_output = [n for n, route in enumerate(routes) if route[-1] in storage.outputs]
            cols_input = [n for n, route in enumerate(routes) if route[0] in storage.inputs]
            ind = np.zeros(len(cols_output)+len(cols_input)+1, dtype=np.int)
            val = np.zeros(len(cols_output)+len(cols_input)+1, dtype=np.float64)
            for n, c in enumerate(cols_output):
                ind[n] = self.idx_col_routes+c
                val[n] = 1
            for n, c in enumerate(cols_input):
                ind[len(cols_output)+n] = self.idx_col_routes+c
                val[len(cols_output)+n] = -1
            # Two rows needed again for the range constraint on change in storage volume
            # Add slack variable
            ind[-1] = self.idx_col_slacks + slack_col
            val[-1] = 1.0
            slack_col += 1
            lp.add_row(ind, val, 0.0)
            # Add slack variable
            ind[-1] = self.idx_col_slacks + slack_col
            val[-1] = -1.0
            slack_col += 1
            lp.add_row(ind, -val, 0.0)

        self.routes = routes
        self.non_storages = non_storages
        self.storages = storages

        # Now expand the constraints and objective function to multiple
        lp.set_num_problems(len(model.scenarios.combinations))
        self.combinations = np.array([s for s in model.scenarios.combinations])

        # Initialise the interior point solver.
        lp.init(self._solver, verbose=2)

    def solve(self, model):
        lp = self.lp
        timestep = model.timestep
        routes = self.routes
        non_storages = self.non_storages
        storages = self.storages
        combinations = self.combinations

        # update route properties
        for col, route in enumerate(routes):
            cost = np.array(route[0].get_all_cost(timestep, combinations))
            for node in route[1:-1]:
                if isinstance(node, BaseLink):
                    cost += np.array(node.get_all_cost(timestep, combinations))
            cost += np.array(route[-1].get_all_cost(timestep, combinations))
            lp.set_objective(self.idx_col_routes+col, -cost)

        # update non-storage properties
        for i, node in enumerate(non_storages):
            row = self.idx_row_non_storages_upper[i]
            # Only update upper bound if the node has a valid row (i.e. is not unbounded).
            if row >= 0:
                max_flow = np.array(node.get_all_max_flow(timestep, combinations))
                lp.set_bound(row, max_flow)

            # Now update lower bounds
            row = self.idx_row_non_storages_lower[i]
            if row >= 0:
                min_flow = np.array(node.get_all_min_flow(timestep, combinations))
                lp.set_bound(row, -min_flow)

        # update storage node constraint
        for col, storage in enumerate(storages):
            volume = np.array(storage._volume)
            max_volume = np.array(storage.get_all_max_volume(timestep, combinations))
            avail_volume = volume - np.array(storage.get_all_min_volume(timestep, combinations))
            avail_volume[avail_volume < 0.0] = 0.0
            # change in storage cannot be more than the current volume or
            # result in maximum volume being exceeded
            lb = -avail_volume/timestep.days
            ub = (max_volume-volume)/timestep.days
            lp.set_bound(self.idx_row_storages+col*2, ub)
            lp.set_bound(self.idx_row_storages+col*2+1, -lb)

        # Solve the problem
        lp.solve(self._solver, verbose=2)
        route_flow = self._solver.x
        status = self._solver.status

        if np.any(status != 0):
            raise RuntimeError("Solver did find an optimal solution for at least one problems.")

        for i, route in enumerate(routes):
            flow = route_flow[:, i]
            # TODO make this cleaner.
            route[0].commit_all(flow)
            route[-1].commit_all(flow)
            for node in route[1:-1]:
                if isinstance(node, BaseLink):
                    node.commit_all(flow)
