

from ..core import Input, Link, Output, Storage, Solver, PiecewiseLink, InputFromOtherDomain
from collections import defaultdict
import glpk
import datetime

inf = float('inf')

cdef int NODE_TYPE_LINK = 0
cdef int NODE_TYPE_INPUT = 1
cdef int NODE_TYPE_OUTPUT = 2


cdef class CyNode:
    cdef object parent
    cdef int node_type
    # TODO This is not optimal because of glpk interfacing requiring
    # a Python object so it can cope with None for unbounded.
    cdef object min_flow
    cdef object max_flow
    cdef float cost
    cdef float benefit
    cdef float opt_flow
    """
    A very simple representation of a node in the model.

    The purpose of this class is to shadow the core.Node instances in the model
    while solving. The general principle is that at the beginning of the timestep
    the attributes of this node will be updated with those from the parent (if
    required) once. All looping through nodes for updating the LP is then
    undetaken using this class which can be more aggressively optimised.
    """
    def __init__(self, parent):
        self.parent = parent
        if isinstance(parent, Link):
            self.node_type = NODE_TYPE_LINK
        elif isinstance(parent, Input):
            self.node_type = NODE_TYPE_INPUT
        elif isinstance(parent, Output):
            self.node_type = NODE_TYPE_OUTPUT
        else:
            raise ValueError("Parent node type ({}) not understood.".format(parent.__class__))
        self.max_flow = 0.0
        self.min_flow = 0.0
        self.cost = 0.0
        self.benefit = 0.0
        # For the result of the LP solve
        self.opt_flow = 0.0


cdef class CyStorage:
    cdef object parent
    cdef float min_volume
    cdef float current_volume
    cdef float max_volume
    """
    A very simple representation of a Storage node in the model
    """
    def __init__(self, parent):
        self.parent = parent
        if not isinstance(parent, Storage):
            raise ValueError("Parent node type ({}) not understood.".format(parent.__class__))
        self.min_volume = 0.0
        self.current_volume = 0.0
        self.max_volume = 0.0


class SolverFastGLPK(Solver):
    name = 'FastGLPK'

    def solve(self, model):
        cdef CyNode nd, ind, ond
        cdef CyStorage snd
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
            nodes = self.nodes = []
            storage_nodes = self.storage_nodes = []
            input_nodes = self.input_nodes = {}
            output_nodes = self.output_nodes = {}

            for py_node in model.nodes():
                if isinstance(py_node, Storage):
                    # Storage requires a special case
                    snd = CyStorage(py_node)
                    storage_nodes.append(snd)
                    py_node._cy_node = snd
                elif isinstance(py_node, PiecewiseLink):
                    # PiecewiseLink is a dummy and shouldn't be involved in
                    # any routes, so it can be ignored.
                    continue
                else:
                    # All other nodes are of type Node
                    nd = CyNode(py_node)
                    nodes.append(nd)
                    # Make a reference to the 'solver' clone on the parent.
                    py_node._cy_node = nd

                    if nd.node_type == NODE_TYPE_INPUT:
                        input_nodes[nd] = {'cols': [], 'col_idxs': []}
                    if nd.node_type == NODE_TYPE_OUTPUT:
                        output_nodes[nd] = {'cols': [], 'col_idxs': []}


            routes = model.find_all_routes(Input, Output, valid=(Link, Input, Output))
            # Swap core.Node for CyNode
            routes = [[node._cy_node for node in route] for route in routes]

            first_index = lp.cols.add(len(routes))
            routes = self.routes = list(zip([lp.cols[index] for index in range(first_index, first_index+len(routes))], routes))

            intermediate_max_flow_constraints = self.intermediate_max_flow_constraints = {}

            for col, route in routes:
                col.bounds = 0, None  # input must be >= 0
                ind = route[0]
                ond = route[-1]

                input_nodes[ind]['cols'].append(col)
                input_nodes[ind]['col_idxs'].append(col.index)

                output_nodes[ond]['cols'].append(col)
                output_nodes[ond]['col_idxs'].append(col.index)

                # find constraints on intermediate nodes
                intermediate_nodes = route[1:-1]
                for nd in intermediate_nodes:
                    if nd not in intermediate_max_flow_constraints:
                        row_idx = lp.rows.add(1)
                        row = lp.rows[row_idx]
                        intermediate_max_flow_constraints[nd] = row
                        col_idxs = []
                        for col, route in routes:
                            if nd in route:
                                col_idxs.append(col.index)
                        row.matrix = [(idx, 1.0) for idx in col_idxs]

            # initialise the structure (only) for the input constraint
            for ind, info in input_nodes.items():
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
                output_cross_domain_nodes[output_node._cy_node].append(input_node._cy_node)

            for ond, info in output_nodes.items():
                # add a column for each output

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
                cross_domain_nodes = output_cross_domain_nodes[ond]
                if len(cross_domain_nodes) > 0:
                    row_idx = lp.rows.add(1)
                    row = lp.rows[row_idx]
                    row.bounds = 0, 0
                    input_matrix = []
                    for ind in cross_domain_nodes:
                        input_info = input_nodes[ind]
                        # TODO Make this vary with timestep
                        coef = ind.parent.properties['conversion_factor'].value()
                        input_matrix.extend([(idx, 1/coef) for idx in input_info['col_idxs']])
                    output_matrix = [(col_idx, -1.0)]
                    row.matrix = input_matrix + output_matrix
                    info['cross_domain_row'] = row

            storage_rows = self.storage_rows = {}
            # Setup Storage node constraints
            for snd in storage_nodes:

                input_info = input_nodes[snd.parent.input._cy_node]
                output_info = output_nodes[snd.parent.output._cy_node]
                # mass balance between input and output
                row_idx = lp.rows.add(1)
                row = lp.rows[row_idx]
                row.bounds = 0, 0
                input_matrix = [(idx, -1.0) for idx in input_info['col_idxs']]
                output_matrix = [(output_info['output_col'].index, 1.0)]
                row.matrix = input_matrix + output_matrix
                storage_rows[snd] = row

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

        timestamp = self.timestamp = model.timestamp

        # TODO: move outside the solver.
        for node in model.nodes():
            node.before()

        for nd in nodes:
            nd.max_flow = nd.parent.properties['max_flow'].value(timestamp)
            try:
                nd.min_flow = nd.parent.properties['min_flow'].value(timestamp)
            except KeyError:
                nd.min_flow = 0.0
            nd.cost = nd.parent.properties['cost'].value(timestamp)
            try:
                nd.benefit = nd.parent.properties['benefit'].value(timestamp)
            except KeyError:
                nd.benefit = 0.0
            nd.opt_flow = 0.0
            #print nd.parent.name, nd.min_flow, nd.max_flow

        for snd in storage_nodes:
            snd.current_volume = snd.parent.properties['current_volume'].value(timestamp)
            snd.max_volume = snd.parent.properties['max_volume'].value(timestamp)

        # the cost of a route is equal to the sum of the route's node's costs
        costs = []
        cdef float cost
        for col, route in routes:
            cost = 0.0
            for nd in route[0:-1]:
                cost += nd.cost
            lp.obj[col.index] = -cost

        # there is a benefit for inputting water to outputs
        for ond, info in output_nodes.items():
            col = info['output_col']
            cost = ond.benefit
            lp.obj[col.index] = cost

        # input is limited by a minimum and maximum flow, and any licenses
        for ind, info in input_nodes.items():
            if len(info['col_idxs']) > 0:
                row = info['input_constraint']

                max_flow = ind.max_flow
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
        for ond, info in output_nodes.items():
            # update output for the current timestep
            col = info['output_col']
            max_flow = ond.max_flow
            min_flow = ond.min_flow
            col.bounds = min_flow, max_flow
            total_water_outputed[ond.parent.domain] += min_flow

        # intermediate node max flow constraints
        for nd, row in intermediate_max_flow_constraints.items():
            row.bounds = 0, nd.max_flow

        # storage limits
        for snd, row in storage_rows.items():
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
        result = [round(col.primal, 3) for col, route in routes]
        total_water_supplied = defaultdict(lambda: 0.0)
        for col, route in routes:
            nd = route[-1]
            total_water_supplied[nd.parent.domain] += round(col.primal, 3)

        # commit the volume of water actually supplied
        for n, (col, route) in enumerate(routes):
            for nd in route:
                nd.opt_flow += result[n]

        for nd in nodes:
            #print nd.parent.name, nd.opt_flow
            nd.parent.commit(nd.opt_flow)

        # calculate the total amount of water transferred via each node/link
        volumes_links = {}
        volumes_nodes = {}
        for n, (col, route) in enumerate(routes):
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

        return status, total_water_outputed, total_water_supplied, volumes_links, volumes_nodes
