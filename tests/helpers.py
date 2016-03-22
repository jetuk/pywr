import os
from numpy.testing import assert_allclose
import pywr.core
from pywr.domains import river
from pywr.core import Model


def load_model(filename=None, data=None, solver=None):
    '''Load a test model and check it'''
    if data is None:
        path = os.path.join(os.path.dirname(__file__), 'models', filename)
        with open(path, 'r') as f:
            data = f.read()
    else:
        path = None
    #xml = ET.fromstring(data)
    #model = pywr.core.Model.from_xml(xml, path=path, solver=solver)
    model = Model.loads(data)
    model.check()
    return model


def assert_model(model, expected_node_results):
    __tracebackhide__ = True
    model.step()

    for node in model.nodes():
        if node.name in expected_node_results:
            if isinstance(node, pywr.core.BaseNode):
                assert_allclose(expected_node_results[node.name], node.flow, atol=1e-7)
            elif isinstance(node, pywr.core.Storage):
                assert_allclose(expected_node_results[node.name], node.volume, atol=1e-7)
