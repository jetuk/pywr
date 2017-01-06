from . import Solver
from pycllp.lp import EqualityLP
from pycllp.solvers import solver_registry
from pywr._core import BaseInput, BaseOutput, BaseLink
from pywr._core cimport *
import numpy as np
cimport numpy as np

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

cdef class PyCLLPSolver:
    """Python wrapper the interior point solvers in pycllp

    """
    cdef int idx_col_routes
    cdef int idx_col_slacks
    cdef int[:] idx_row_non_storages_upper
    cdef int[:] idx_row_non_storages_lower
    cdef int[:, :] combinations
    cdef int idx_row_storages
    cdef object _solver_name
    cdef object _solver
    cdef object lp
    cdef object routes
    cdef object non_storages
    cdef object storages


    def __init__(self, *args, **kwargs):
        self._solver_name = kwargs.pop('solver', 'dense_primal_normal')
        super(PyCLLPSolver, self).__init__(*args, **kwargs)

    def setup(self, model):
        cdef Node supply
        cdef Node demand
        cdef Node node
        cdef double min_flow
        cdef double max_flow
        cdef double cost
        cdef double avail_volume
        cdef int n, col, row
        cdef int[:] ind
        cdef double[:] val
        cdef double lb
        cdef double ub
        cdef Timestep timestep
        cdef int status
        cdef cross_domain_row
        cdef ScenarioIndex s

        # This is the interior point method to use as provided by pycllp
        self._solver = solver_registry[self._solver_name]()

        routes = model.find_all_routes(BaseInput, BaseOutput, valid=(BaseLink, BaseInput, BaseOutput))
        # Find cross-domain routes
        cross_domain_routes = model.find_all_routes(BaseOutput, BaseInput, max_length=2, domain_match='different')

        non_storages = []
        storages = []
        for some_node in sorted(model.graph.nodes(), key=lambda x: x.name):
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
        self.idx_row_non_storages_upper = np.empty(len(non_storages), dtype=np.int32)
        self.idx_row_non_storages_lower = np.empty(len(non_storages), dtype=np.int32)
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

            ind = np.zeros(len(cols)+1, dtype=np.int32)
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
            lp.add_row(ind, -np.array(val), 0.0)
            self.idx_row_non_storages_lower[row] = lp.nrows - 1

        # storage
        if len(storages):
            self.idx_row_storages = lp.nrows
        for col, storage in enumerate(storages):
            cols_output = [n for n, route in enumerate(routes) if route[-1] in storage.outputs]
            cols_input = [n for n, route in enumerate(routes) if route[0] in storage.inputs]
            ind = np.zeros(len(cols_output)+len(cols_input)+1, dtype=np.int32)
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
            lp.add_row(ind, -np.array(val), 0.0)


        # This work could be moved in to the loop above (as in the other solvers), however by
        # adding these mass balance constraints last the system matrix contains all the equality straights first.
        # This gives a nice structure, when including the slack variables, to the matrix. It looks like,
        #    [[A1 I],
        #     [A2 0]]
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
                continue
            # Add constraint for cross-domain routes
            # i.e. those from a demand to a supply
            if some_node in cross_domain_cols:
                col_vals = cross_domain_cols[some_node]
                ind = np.zeros(len(col_vals)+len(cols), dtype=np.int32)
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

        self.routes = routes
        self.non_storages = non_storages
        self.storages = storages

        # Now expand the constraints and objective function to multiple
        lp.set_num_problems(len(model.scenarios.combinations))
        self.combinations = np.array([s.indices for s in model.scenarios.combinations])

        # Initialise the interior point solver.
        lp.init(self._solver, verbose=0)

    def solve(self, model):
        cdef Node node
        cdef double[:] min_flow
        cdef double[:] max_flow
        cdef double[:] cost
        cdef double[:] avail_volume
        cdef double[:] max_volume
        cdef double[:] volume
        cdef double[:] vals
        cdef int i, col
        cdef double[:] lb
        cdef double[:] ub
        cdef double[:, :] route_flow
        cdef Timestep timestep
        cdef int[:] status
        cdef int[:, :] combinations
        cdef cross_domain_col

        lp = self.lp
        timestep = model.timestep
        routes = self.routes
        non_storages = self.non_storages
        storages = self.storages
        combinations = self.combinations

        # Preallocate all working arrays
        # TODO move this to setup()?
        vals = np.empty(combinations.shape[0])
        cost = np.empty(combinations.shape[0])
        max_volume = np.empty(combinations.shape[0])
        avail_volume = np.empty(combinations.shape[0])
        # update route properties
        for col, route in enumerate(routes):
            route[0].get_all_cost(timestep, combinations, cost)
            for node in route[1:-1]:
                if isinstance(node, BaseLink):
                    node.get_all_cost(timestep, combinations, vals)
                    for i in range(combinations.shape[0]):
                        cost[i] += vals[i]
            route[-1].get_all_cost(timestep, combinations, vals)
            for i in range(combinations.shape[0]):
                cost[i] += vals[i]
                cost[i] *= -1.0
            lp.set_objective(self.idx_col_routes+col, cost)

        # update non-storage properties
        for i, node in enumerate(non_storages):
            row = self.idx_row_non_storages_upper[i]
            # Only update upper bound if the node has a valid row (i.e. is not unbounded).
            if row >= 0:
                node.get_all_max_flow(timestep, combinations, vals)
                lp.set_bound(row, vals)

            # Now update lower bounds
            row = self.idx_row_non_storages_lower[i]
            if row >= 0:
                node.get_all_min_flow(timestep, combinations, vals)
                for i in range(combinations.shape[0]):
                    vals[i] *= -1.0
                lp.set_bound(row, vals)

        # update storage node constraint
        for col, storage in enumerate(storages):
            volume = storage._volume
            storage.get_all_max_volume(timestep, combinations, max_volume)
            storage.get_all_min_volume(timestep, combinations, avail_volume)

            # change in storage cannot be more than the current volume or
            # result in maximum volume being exceeded
            for i in range(combinations.shape[0]):
                avail_volume[i] = max(volume[i] - avail_volume[i], 0.0)/timestep.days
                max_volume[i] = (max_volume[i] - volume[i])/timestep.days

            lp.set_bound(self.idx_row_storages+col*2, max_volume)
            lp.set_bound(self.idx_row_storages+col*2+1, avail_volume)

        # Solve the problem
        lp.solve(self._solver, verbose=1)
        route_flow = self._solver.x
        status = self._solver.status

        for i in range(combinations.shape[0]):
            if status[i] != 0:
                raise RuntimeError("Solver did find an optimal solution for at least one problems.")

        for i, route in enumerate(routes):
            vals = route_flow[:, i]
            # TODO make this cleaner.
            route[0].commit_all(vals)
            route[-1].commit_all(vals)
            for node in route[1:-1]:
                if isinstance(node, BaseLink):
                    node.commit_all(vals)
