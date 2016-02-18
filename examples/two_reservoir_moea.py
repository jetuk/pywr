"""
This example shows the trade-off of deficit against cost by altering a reservoir control curve.
"""

import os
import pandas
import numpy as np
import inspyred
from pywr.core import Model, Input, Output, Link, Storage
from pywr.parameters import Parameter, ParameterArrayIndexed, ParameterMonthlyProfile, ParameterAnnualHarmonicSeries
from pywr.recorders import NodeRecorder, Recorder


class BaseConstantNodeRecorder(NodeRecorder):
    """
    Base class for NodeRecorder classes with a single value for each scenario combination
    """
    def __init__(self, *args, **kwargs):
        super(BaseConstantNodeRecorder, self).__init__(*args, **kwargs)
        self._values = None

    def setup(self):
        self._values = np.zeros(len(self.model.scenarios.combinations))

    def reset(self):
        self._values[...] = 0.0

    def save(self):
        raise NotImplementedError()

    def value(self, agg_func=np.mean):
        return agg_func(self._values)


class TotalDeficitNodeRecorder(BaseConstantNodeRecorder):
    """
    Recorder to total the difference between modelled flow and max_flow for a Node
    """
    def save(self):
        ts = self.model.timestepper.current
        node = self.node
        for i, scenario_combination in enumerate(self.model.scenarios.combinations):
            max_flow = node.get_max_flow(ts, scenario_combination)
            self._values[i] += max_flow - node.flow[i]


class TotalFlowRecorder(BaseConstantNodeRecorder):
    """
    Recorder to total the flow for a Node.

    A factor can be provided to scale the total flow (e.g. for calculating operational costs).
    """
    def __init__(self, *args, factor=1.0, **kwargs):
        super(TotalFlowRecorder, self).__init__(*args, **kwargs)
        self.factor = factor

    def save(self):
        self._values += self.node.flow*self.factor


class AggregatedRecorder(Recorder):
    def __init__(self, model, recorders):
        super(AggregatedRecorder, self).__init__(model)
        self.recorders = recorders

    def value(self, agg_func=np.max):
        return agg_func([r.value() for r in self.recorders])


class TransferController(Parameter):
    """ Control the flow based on a control_curve
    """
    def __init__(self, reservoir1, control_curve):
        super(TransferController, self).__init__()
        self.reservoir1 = reservoir1
        self.control_curve = control_curve
        control_curve.parent = self

    def value(self, timestep, *args, **kwargs):
        percent_full = self.reservoir1._volume[0] / 20000.0  # TODO: this doesn't work with scenarios

        control_curve = self.control_curve.value(timestep, *args)

        if percent_full < control_curve:
            max_flow = 10.0
        else:
            max_flow = 0.0
        return max_flow

    @property
    def is_variable(self):
        return self.control_curve.is_variable

    @property
    def size(self):
        return self.control_curve.size

    def update(self, values):
        self.control_curve.update(values)

    def lower_bounds(self):
        return self.control_curve.lower_bounds()

    def upper_bounds(self):
        return self.control_curve.upper_bounds()


class OptimisationModel(Model):

    def _cache_variable_parameters(self):
        variables = []
        variable_map = [0, ]
        for node in self.nodes:
            for var in node.variables:
                variable_map.append(variable_map[-1]+var.size)
                variables.append(var)

        self._variables = variables
        self._variable_map = variable_map

    def _cache_objectives(self):
        objectives = []
        for r in self.recorders:
            if r.is_objective:
                objectives.append(r)

        self._objectives = objectives

    def setup(self):
        super(OptimisationModel, self).setup()
        self._cache_variable_parameters()
        self._cache_objectives()

    def generator(self, random, args):
        size = self._variable_map[-1]

        return [random.uniform(0.0, 1.0) for i in range(size)]

    def evaluator(self, candidates, args):

        fitness = []
        for i, candidate in enumerate(candidates):
            print('Running candidiate: {:03d}'.format(i))
            for ivar, var in enumerate(self._variables):
                j = slice(self._variable_map[ivar], self._variable_map[ivar+1])
                var.update(candidate[j])

            self.reset()
            self.run()

            fitness.append(inspyred.ec.emo.Pareto([r.value() for r in self._objectives]))
        return fitness

    def bounder(self, candidate, args):
        for ivar, var in enumerate(self._variables):
            lower = var.lower_bounds()
            upper = var.upper_bounds()
            j = slice(self._variable_map[ivar], self._variable_map[ivar+1])
            candidate[j] = np.minimum(upper, np.maximum(lower, candidate[j]))
        return candidate


def create_model(harmonic=True):
    # import flow timeseries for catchments
    flow = pandas.read_csv(os.path.join('data', 'Not a real flow series.csv'))

    flow['Date'] = flow['Date'].apply(pandas.to_datetime)
    flow.set_index('Date', inplace=True)
    flow_parameter = ParameterArrayIndexed(flow['Flow'].values)

    model = OptimisationModel(
        solver='glpk',
        parameters={
            'timestamp_start': flow.index[0],
            'timestamp_finish': flow.index[-1],
        }
    )

    catchment1 = Input(model, 'catchment1', min_flow=flow_parameter, max_flow=flow_parameter)
    catchment2 = Input(model, 'catchment2', min_flow=flow_parameter, max_flow=flow_parameter)

    reservoir1 = Storage(model, 'reservoir1', min_volume=3000, max_volume=20000, volume=16000)
    reservoir2 = Storage(model, 'reservoir2', min_volume=3000, max_volume=20000, volume=16000)

    if harmonic:
        control_curve = ParameterAnnualHarmonicSeries(0.5, [0.5], [0.0], mean_upper_bounds=1.0, amplitude_upper_bounds=1.0)
    else:
        control_curve = ParameterMonthlyProfile(np.array([0.0]*12, np.float32), lower_bounds=0.0, upper_bounds=1.0)

    control_curve.is_variable = True
    controller = TransferController(reservoir1, control_curve)
    transfer = Link(model, 'transfer', max_flow=controller, cost=-500)

    demand1 = Output(model, 'demand1', max_flow=100.0, cost=-101)
    demand2 = Output(model, 'demand2', max_flow=81.0, cost=-100)

    river1 = Link(model, 'river1')
    river2 = Link(model, 'river2')

    # compensation flows from reservoirs
    compensation1 = Link(model, 'compensation1', max_flow=5.0, cost=-9999)
    compensation2 = Link(model, 'compensation2', max_flow=5.0, cost=-9998)

    terminator = Output(model, 'terminator', cost=1.0)

    catchment1.connect(reservoir1)
    catchment2.connect(reservoir2)
    reservoir1.connect(demand1)
    reservoir2.connect(demand2)
    reservoir2.connect(transfer)
    transfer.connect(reservoir1)
    reservoir1.connect(river1)
    reservoir2.connect(river2)
    river1.connect(terminator)
    river2.connect(terminator)
    compensation1.connect(terminator)
    compensation2.connect(terminator)

    r1 = TotalDeficitNodeRecorder(model, demand1)
    r2 = TotalDeficitNodeRecorder(model, demand2)
    r3 = AggregatedRecorder(model, [r1, r2])
    r3.is_objective = True
    r4 = TotalFlowRecorder(model, transfer)
    r4.is_objective = True

    return model



def main(prng=None, display=False, harmonic=False):
    from random import Random
    from time import time

    if prng is None:
        prng = Random()
        prng.seed(time())

    problem = create_model(harmonic=harmonic)
    problem.setup()
    ea = inspyred.ec.emo.NSGA2(prng)
    ea.variator = [inspyred.ec.variators.blend_crossover,
                   inspyred.ec.variators.gaussian_mutation]
    ea.terminator = inspyred.ec.terminators.generation_termination
    ea.observer = [
        inspyred.ec.observers.file_observer,
    ]
    final_pop = ea.evolve(generator=problem.generator,
                          evaluator=problem.evaluator,
                          pop_size=25,
                          bounder=problem.bounder,
                          maximize=False,
                          max_generations=20)

    if display:
        from matplotlib import pylab
        final_arc = ea.archive
        print('Best Solutions: \n')
        for f in final_arc:
            print(f)
        import pylab
        x = []
        y = []

        for f in final_arc:
            x.append(f.fitness[0])
            y.append(f.fitness[1])

        pylab.scatter(x, y, c='b')
        pylab.xlabel('Maximum demand deficit [Ml/d]')
        pylab.ylabel('Transferred volume [Ml/d]')
        title = 'Harmonic Control Curve' if harmonic else 'Monthly Control Curve'
        pylab.savefig('{0} Example ({1}).pdf'.format(ea.__class__.__name__, title), format='pdf')
        pylab.show()
    return ea

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--harmonic', action='store_true', help='Use an harmonic control curve.')
    args = parser.parse_args()

    main(display=True, harmonic=args.harmonic)