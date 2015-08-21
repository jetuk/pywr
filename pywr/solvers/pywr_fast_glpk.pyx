# cython: profile=False
from cpython cimport array
from cpython.ref cimport PyObject
from libc.stdlib cimport malloc, free
import array

from ..core import Input, Link, Output, Storage, Solver, PiecewiseLink, InputFromOtherDomain
from collections import defaultdict
import glpk
import datetime

inf = float('inf')

cdef int NODE_TYPE_LINK = 0
cdef int NODE_TYPE_INPUT = 1
cdef int NODE_TYPE_OUTPUT = 2


cdef struct CyNode:
    # A very simple representation of a node in the model.

    # The purpose of this class is to shadow the core.Node instances in the model
    # while solving. The general principle is that at the beginning of the timestep
    # the attributes of this node will be updated with those from the parent (if
    # required) once. All looping through nodes for updating the LP is then
    # undetaken using this class which can be more aggressively optimised.
    PyObject* parent
    int node_type
    int index

    float min_flow
    float max_flow
    int has_max_flow
    float cost
    float benefit
    float opt_flow

cdef init_cynode(CyNode *self, object parent, int index):
    """
    Init function for CyNode
    """
    self.parent = <PyObject*>parent
    self.node_type = 0
    if isinstance(parent, Link):
        self.node_type = NODE_TYPE_LINK
    elif isinstance(parent, Input):
        self.node_type = NODE_TYPE_INPUT
    elif isinstance(parent, Output):
        self.node_type = NODE_TYPE_OUTPUT
    else:
        raise ValueError("Parent node type ({}) not understood.".format(parent.__class__))
    self.max_flow = 0.0
    self.has_max_flow = 0
    self.min_flow = 0.0
    self.cost = 0.0
    self.benefit = 0.0
    # For the result of the LP solve
    self.opt_flow = 0.0
    self.index = index


cdef struct CyStorage:
    PyObject* parent
    float min_volume
    float current_volume
    float max_volume

cdef init_cystorage(CyStorage *self, object parent):
        if not isinstance(parent, Storage):
            raise ValueError("Parent node type ({}) not understood.".format(parent.__class__))
        self.parent = <PyObject*>parent

        self.min_volume = 0.0
        self.current_volume = 0.0
        self.max_volume = 0.0


cdef class CyRoute:
    cdef int[:] node_indices
    cdef int size
    # glpk Column for this route
    cdef object col

    def __init__(self, indices):
        self.node_indices = array.array('i', indices)


cdef class CySolverFastGLPK:
    cdef int nnodes, nstorage_nodes
    cdef CyNode* nodes
    cdef CyStorage* storage_nodes
    cdef object lp, output_cross_domain_nodes
    cdef dict input_nodes, output_nodes, intermediate_max_flow_constraints, storage_rows
    cdef list routes, cross_domain_routes
    cdef float[:] result

    def solve(self, model):
        cdef CyNode* nodes
        cdef CyNode nd, ind, ond
        cdef CyStorage* storage_nodes
        cdef CyStorage snd
        cdef CyRoute route, route2
        cdef int inode, isnode, iroute, i
        cdef int *nnodes, *nstorage_nodes

        nnodes = &self.nnodes
        nstorage_nodes = &self.nstorage_nodes

        timestep = model.parameters['timestep']
        if isinstance(timestep, datetime.timedelta):
            timestep = timestep.days

        if model.dirty:
            '''
            This section should only need to be run when the model has changed
            structure (e.g. a new connection has been made).
            '''

            lp = self.lp = glpk.LPX()
            lp.obj.maximize = True
            # Make a local Node or Storage instances for each node in the
            # model.

            # Free any existing node arrays
            if nodes:
                free(nodes)
            nnodes[0] = 0

            if storage_nodes:
                free(storage_nodes)
            nstorage_nodes[0] = 0

            input_nodes = self.input_nodes = {}
            output_nodes = self.output_nodes = {}

            py_nodes = model.nodes()

            # Count number nodes required
            for py_node in py_nodes:
                if isinstance(py_node, Storage):
                    nstorage_nodes[0] += 1
                elif isinstance(py_node, PiecewiseLink):
                    continue
                else:
                    nnodes[0] += 1

            nodes = <CyNode*>malloc(nnodes[0]*sizeof(CyNode))
            self.nodes = nodes
            storage_nodes = <CyStorage*>malloc(nstorage_nodes[0]*sizeof(CyStorage))
            self.storage_nodes = storage_nodes

            inode = 0
            isnode = 0
            for py_node in py_nodes:
                if isinstance(py_node, Storage):
                    # Storage requires a special case
                    init_cystorage(&storage_nodes[isnode], py_node)
                    py_node._cy_node_index = isnode
                    isnode += 1
                elif isinstance(py_node, PiecewiseLink):
                    # PiecewiseLink is a dummy and shouldn't be involved in
                    # any routes, so it can be ignored.
                    continue
                else:
                    # All other nodes are of type Node
                    init_cynode(&nodes[inode], py_node, inode)
                    nd = nodes[inode]
                    # Make a reference to the 'solver' clone on the parent.
                    py_node._cy_node_index = inode
                    if nd.node_type == NODE_TYPE_INPUT:
                        input_nodes[inode] = {'cols': [], 'col_idxs': []}
                    if nd.node_type == NODE_TYPE_OUTPUT:
                        output_nodes[inode] = {'cols': [], 'col_idxs': []}

                    inode += 1


            py_routes = model.find_all_routes(Input, Output, valid=(Link, Input, Output))
            # Swap core.Node for CyNode
            routes = self.routes = []
            # Allocate space for results
            self.result = array.array('f', [0.0 for i in range(len(py_routes))])

            for py_route in py_routes:
                route = CyRoute([node._cy_node_index for node in py_route])
                col_idx = lp.cols.add(1)
                route.col = lp.cols[col_idx]
                routes.append(route)

            #routes = [[node._cy_node for node in route] for route in routes]

            #first_index = lp.cols.add(len(routes))
            #routes = self.routes = list(zip([lp.cols[index] for index in range(first_index, first_index+len(routes))], routes))

            intermediate_max_flow_constraints = self.intermediate_max_flow_constraints = {}

            for route in routes:
                col = route.col
                col.bounds = 0, None  # input must be >= 0
                ind = nodes[route.node_indices[0]]
                ond = nodes[route.node_indices[-1]]

                input_nodes[ind.index]['cols'].append(col)
                input_nodes[ind.index]['col_idxs'].append(col.index)

                output_nodes[ond.index]['cols'].append(col)
                output_nodes[ond.index]['col_idxs'].append(col.index)

                # find constraints on intermediate nodes
                intermediate_nodes = [inode for inode in route.node_indices[1:-1]]
                for inode in intermediate_nodes:
                    if inode not in intermediate_max_flow_constraints:
                        nd = nodes[inode]
                        row_idx = lp.rows.add(1)
                        row = lp.rows[row_idx]
                        intermediate_max_flow_constraints[nd.index] = row
                        col_idxs = []
                        for route2 in routes:
                            if nd.index in route2.node_indices:
                                col_idxs.append(route2.col.index)
                        row.matrix = [(idx, 1.0) for idx in col_idxs]

            # initialise the structure (only) for the input constraint
            for inode, info in input_nodes.items():
                if len(info['col_idxs']) > 0:
                    row_idx = lp.rows.add(1)
                    row = lp.rows[row_idx]
                    info['input_constraint'] = row
                    info['matrix'] = [(idx, 1.0) for idx in info['col_idxs']]

            # Find cross-domain routes
            cross_domain_routes = self.cross_domain_routes = model.find_all_routes(Output, InputFromOtherDomain,
                                                                                   max_length=2,
                                                                                   domain_match='different')
            # translate to a dictionary with the Ouput node as a key
            output_cross_domain_nodes = self.output_cross_domain_nodes = defaultdict(list)
            for output_node, input_node in cross_domain_routes:
                ond = nodes[output_node._cy_node_index]
                ind = nodes[input_node._cy_node_index]
                output_cross_domain_nodes[ond.index].append(ind.index)

            for inode, info in output_nodes.items():
                # add a column for each output
                ond = nodes[inode]
                col_idx = lp.cols.add(1)
                col = lp.cols[col_idx]
                info['output_col'] = col
                if len(info['col_idxs']) > 0:
                    # mass balance between input and output
                    row_idx = lp.rows.add(1)
                    row = lp.rows[row_idx]
                    row.bounds = 0, 0
                    input_matrix = [(idx, 1.0) for idx in info['col_idxs']]
                    output_matrix = [(col_idx, -1.0)]
                    row.matrix = input_matrix + output_matrix
                    info['output_row'] = row

                # Deal with exports from this output node to other input nodes
                info['cross_domain_row'] = None
                cross_domain_nodes = output_cross_domain_nodes[ond.index]
                if len(cross_domain_nodes) > 0:
                    row_idx = lp.rows.add(1)
                    row = lp.rows[row_idx]
                    row.bounds = 0, 0
                    input_matrix = []
                    for inode in cross_domain_nodes:
                        input_info = input_nodes[inode]
                        # TODO Make this vary with timestep
                        coef = (<object>ind.parent).properties['conversion_factor'].value()
                        input_matrix.extend([(idx, 1/coef) for idx in input_info['col_idxs']])
                    output_matrix = [(col_idx, -1.0)]
                    row.matrix = input_matrix + output_matrix
                    info['cross_domain_row'] = row

            storage_rows = self.storage_rows = {}
            # Setup Storage node constraints
            for inode in range(nstorage_nodes[0]):
                snd = storage_nodes[inode]

                input_info = input_nodes[(<object>snd.parent).input._cy_node_index]
                output_info = output_nodes[(<object>snd.parent).output._cy_node_index]
                # mass balance between input and output
                row_idx = lp.rows.add(1)
                row = lp.rows[row_idx]
                row.bounds = 0, 0
                input_matrix = [(idx, -1.0) for idx in input_info['col_idxs']]
                output_matrix = [(output_info['output_col'].index, 1.0)]
                row.matrix = input_matrix + output_matrix
                storage_rows[inode] = row

            # TODO add min flow requirement
            """
            # add mrf constraint rows
            for river_gauge_node, info in river_gauge_nodes.items():
                row_idx = lp.rows.add(1)
                row = lp.rows[row_idx]
                info['mrf_constraint'] = row
            """
            model.dirty = False
        else:
            lp = self.lp
            nodes = self.nodes
            storage_nodes = self.storage_nodes
            input_nodes = self.input_nodes
            output_nodes = self.output_nodes
            routes = self.routes
            intermediate_max_flow_constraints = self.intermediate_max_flow_constraints
            cross_domain_routes = self.cross_domain_routes
            storage_rows = self.storage_rows
            #blenders = self.blenders
            #groups = self.groups

        timestamp = model.timestamp

        for inode in range(nnodes[0]):
            parent = <object>nodes[inode].parent

            try:
                nodes[inode].max_flow = parent.properties['max_flow'].value(timestamp)
                nodes[inode].has_max_flow = 1
            except TypeError:
                # Float not found. Assume max_flow is unbounded
                nodes[inode].has_max_flow = 0
                nodes[inode].max_flow = 0.0

            try:
                nodes[inode].min_flow = parent.properties['min_flow'].value(timestamp)
            except KeyError:
                nodes[inode].min_flow = 0.0

            nodes[inode].cost = parent.properties['cost'].value(timestamp)
            try:
                nodes[inode].benefit = parent.properties['benefit'].value(timestamp)
            except KeyError:
                nodes[inode].benefit = 0.0
            nodes[inode].opt_flow = 0.0

        for inode in range(nstorage_nodes[0]):
            parent = <object>storage_nodes[inode].parent

            storage_nodes[inode].current_volume = parent.properties['current_volume'].value(timestamp)
            storage_nodes[inode].max_volume = parent.properties['max_volume'].value(timestamp)

        # the cost of a route is equal to the sum of the route's node's costs
        costs = []
        cdef float cost
        for route in routes:
            cost = 0.0
            for i in range(route.node_indices.shape[0]):
                cost += nodes[route.node_indices[i]].cost
            lp.obj[route.col.index] = -cost

        # there is a benefit for inputting water to outputs
        for inode, info in output_nodes.items():
            col = info['output_col']
            cost = nodes[inode].benefit
            lp.obj[col.index] = cost

        # input is limited by a minimum and maximum flow, and any licenses
        for inode, info in input_nodes.items():
            if len(info['col_idxs']) > 0:
                ind = nodes[inode]
                row = info['input_constraint']
                # This is an inefficiency with requiring a Python None for
                # unbounded in the glpk interface
                if ind.has_max_flow == 1:
                    max_flow = ind.max_flow
                else:
                    max_flow = None
                """
                max_flow_license = inf
                if input_node.licenses is not None:
                    max_flow_license = input_node.licenses.available(timestamp)
                if max_flow_parameter is not None:
                    max_flow = min(max_flow_parameter, max_flow_license)
                else:
                    max_flow = max_flow_license
                """
                min_flow = ind.min_flow
                row.matrix = info['matrix']
                row.bounds = min_flow, max_flow

        # outputs require a water between a min and maximium flow
        total_water_outputed = defaultdict(lambda: 0.0)
        for inode, info in output_nodes.items():
            ond = nodes[inode]
            # update output for the current timestep
            col = info['output_col']
            if ond.has_max_flow == 1:
                max_flow = ond.max_flow
            else:
                max_flow = None
            min_flow = ond.min_flow
            col.bounds = min_flow, max_flow
            total_water_outputed[(<object>ond.parent).domain] += <object>min_flow

        # intermediate node max flow constraints
        for inode, row in intermediate_max_flow_constraints.items():
            nd = nodes[inode]
            if nd.has_max_flow == 1:
                max_flow = nd.max_flow
            else:
                max_flow = None
            row.bounds = 0, max_flow

        # storage limits
        for inode, row in storage_rows.items():
            snd = storage_nodes[inode]
            current_volume = snd.current_volume
            max_volume = snd.max_volume
            # Change in storage limits
            #   lower bound ensures a net loss is not more than current volume
            #   upper bound ensures a net gain is not more than capacity
            row.bounds = -current_volume/timestep, (max_volume-current_volume)/timestep

        # TODO add min flow requirement
        """
        # mrf constraints
        for river_gauge_node, info in river_gauge_nodes.items():
            mrf_value = river_gauge_node.properties['mrf'].value(timestamp)
            row = info['mrf_constraint']
            if mrf_value is None:
                row.bounds = None, None
                continue
            river_routes = info['river_routes']
            flow_constraint, abstraction_idxs, abstraction_coefficients = self.upstream_constraint(river_routes)
            flow_constraint = max(0, flow_constraint - mrf_value)
            row.matrix = [(abstraction_idxs[n], abstraction_coefficients[n]) for n in range(0, len(abstraction_idxs))]
            row.bounds = 0, flow_constraint


        # blender constraints
        for blender, info in blenders.items():
            matrix = []
            row = info['blender_constraint']
            ratio = blender.properties['ratio'].value(self.timestamp)
            for col_idx, sign in info['routes']:
                if sign == 1:
                    matrix.append((col_idx, sign*(1-ratio)))
                else:
                    matrix.append((col_idx, sign*ratio))
            row.matrix = matrix

        # groups
        for group, info in groups.items():
            if group.licenses is None:
                continue
            row = info['group_constraint']
            if group.licenses is None:
                row.bounds = None, None
            else:
                row.bounds = 0, group.licenses.available(timestamp)
        """
        #print_matrix(lp)
        # solve the linear programme
        lp.simplex()
        assert(lp.status == 'opt')
        status = 'optimal'

        # retrieve the results
        #result = [round(route.col.primal, 3) for route in routes]
        #result = array.array('f', indices)
        total_water_supplied = defaultdict(lambda: 0.0)
        for iroute, route in enumerate(routes):
            self.result[iroute] = round(route.col.primal, 3)
            nd = nodes[route.node_indices[-1]]
            total_water_supplied[(<object>nd.parent).domain] += round(route.col.primal, 3)

        # commit the volume of water actually supplied
        for iroute, route in enumerate(routes):
            for i in range(route.node_indices.shape[0]):
                inode = route.node_indices[i]
                nodes[inode].opt_flow += self.result[iroute]

        for inode in range(nnodes[0]):
            nd = nodes[inode]
            #print (<object>nd.parent).name, nd.opt_flow
            (<object>nd.parent).commit(nd.opt_flow)

        # calculate the total amount of water transferred via each node/link
        volumes_links = {}
        volumes_nodes = {}
        """
        for n, route in enumerate(routes):
            if result[n] > 0:
                for m in range(0, len(route)):
                    volumes_nodes.setdefault(route[m], 0.0)
                    volumes_nodes[route[m]] += result[n]

                    if m+1 == len(route):
                        break

                    pair = (route[m], route[m+1])
                    volumes_links.setdefault(pair, 0.0)
                    volumes_links[pair] += result[n]
        for k, v in volumes_links.items():
            v = round(v, 3)
            if v:
                volumes_links[k] = v
            else:
                del(volumes_links[k])
        for k, v in volumes_nodes.items():
            v = round(v, 3)
            if v:
                volumes_nodes[k] = v
            else:
                del(volumes_nodes[k])
        """
        return status, total_water_outputed, total_water_supplied, volumes_links, volumes_nodes

class SolverFastGLPK(Solver):
    name = 'FastGLPK'

    def __init__(self, ):
        self.cy_solver = CySolverFastGLPK()

    def solve(self, model):
        return self.cy_solver.solve(model)
