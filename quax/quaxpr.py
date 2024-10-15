from jax import core
from enum import Enum
from jax.interpreters import ad


class Operation(Enum):
    UNKNOWN = 0 
    FC = 1
    QUANTIZE = 2
    CONV = 3 
    TANH = 4
    ACTIVATION = 5
    RESHAPE = 6

class AppendedActivation(Enum):
    RELU = 0
    RELU6 = 1

QUAXPR_NAME = "quaxpr"
quaxpr_p = core.Primitive(QUAXPR_NAME)

def quaxpr_prim(x, quax_pytree):
    return quaxpr_p.bind(x.x, quax_pytree = quax_pytree)

def quaxpr_unquant_prim(x, quax_pytree):
    return quaxpr_p.bind(x, quax_pytree = quax_pytree)

def quaxpr_functional(x, op):
    quaxpr_pytree = {}
    quaxpr_pytree['op'] = op 
    return quaxpr_prim(x, quaxpr_pytree)

def quaxpr_default(x, op, mdl, **kwargs):
    pytree_branch = ('quax', ) + mdl.scope.path
    quaxpr_pytree = {}
    quaxpr_pytree['op'] = op 
    quaxpr_pytree['branch'] = pytree_branch
    for key,value in kwargs.items():
        quaxpr_pytree[key] = value 
    return quaxpr_prim(x, quaxpr_pytree)

def quaxpr_impl(x, quax_pytree):
    return x

def quaxpr_abstract_eval(xs, quax_pytree):
    return core.ShapedArray(xs.shape, xs.dtype)

quaxpr_p.def_impl(quaxpr_impl)
quaxpr_p.def_abstract_eval(quaxpr_abstract_eval)
# Define the JVP rule for differentiation
def quaxpr_jvp(primals, tangents, **params):
    x = primals
    t_x = tangents
    y = quaxpr_p.bind(x, **params)
    return y, t_x

ad.defjvp(quaxpr_p, quaxpr_jvp)


def pytree_fc(weight_pytree_branch):
    fc_quaxpr = {}
    fc_quaxpr['op'] = Operation.FC
    fc_quaxpr['branch'] = weight_pytree_branch

    return fc_quaxpr





