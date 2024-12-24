from jax import core
from enum import Enum
from jax.interpreters import ad
import jax.numpy as jnp 


class Operation(Enum):
    UNKNOWN = 0 
    FC = 1
    QUANTIZE = 2
    CONV = 3 
    TANH = 4
    ACTIVATION = 5
    RESHAPE = 6
    ADD = 7
    MUL = 8
    CONCATENATE = 9
    SLICE = 10

class AppendedActivation(Enum):
    RELU = 0
    RELU6 = 1

QUAXPR_NAME = "quaxpr"
quaxpr_p = core.Primitive(QUAXPR_NAME)
def unused(x):
    #return jnp.zeros(x.shape)
    return x

def quaxpr_prim(*args, quax_pytree):
    return quaxpr_p.bind(*args,quax_pytree = quax_pytree)


def quaxpr_unquant_prim(x, quax_pytree):
    return quaxpr_p.bind(x,unused(x), quax_pytree = quax_pytree)

def quaxpr_functional(x, op):
    quaxpr_pytree = {}
    quaxpr_pytree['op'] = op 
    return quaxpr_prim(x.x, unused(x.x), quax_pytree=quaxpr_pytree)

def quaxpr_default(x, op, mdl, t2 = None, **kwargs):
    pytree_branch = ('quax', ) + mdl.scope.path
    quaxpr_pytree = {}
    quaxpr_pytree['op'] = op 
    quaxpr_pytree['branch'] = pytree_branch
    for key,value in kwargs.items():
        quaxpr_pytree[key] = value 
    if t2 != None:
        #other = unused(t2.x)
        other = t2.x
    else:
        other = unused(x.x)
    return quaxpr_prim(x.x,other, quax_pytree=quaxpr_pytree)

def quaxpr_multiarg(*args, op, mdl, **kwargs):
    pytree_branch = ('quax', ) + mdl.scope.path
    quaxpr_pytree = {}
    quaxpr_pytree['op'] = op 
    quaxpr_pytree['branch'] = pytree_branch
    for key,value in kwargs.items():
        quaxpr_pytree[key] = value 
    args = [x.x for x in args]
    return quaxpr_prim(*args, quax_pytree = quaxpr_pytree)


def quaxpr_impl(*args, quax_pytree):
    return args[0] 

def quaxpr_abstract_eval(*avals, quax_pytree):
    xs = avals[0]
    return core.ShapedArray(xs.shape, xs.dtype)

#def quaxpr_impl(x,y, quax_pytree):
#    return x
#
#def quaxpr_abstract_eval(xs,ys, quax_pytree):
#    assert xs.shape == ys.shape
#    return core.ShapedArray(xs.shape, xs.dtype)

quaxpr_p.def_impl(quaxpr_impl)
quaxpr_p.def_abstract_eval(quaxpr_abstract_eval)
# Define the JVP rule for differentiation
def quaxpr_jvp(primals, tangents, **params):
    #xp, yp = primals
    t_x= tangents
    y = quaxpr_p.bind(*primals, **params)
    return y, t_x

#i guess need jvp rules for every value

#what the hell is this doing....it messes things up badly 
#ad.defjvp(quaxpr_p, quaxpr_jvp, quaxpr_jvp)
ad.primitive_jvps[quaxpr_p] = quaxpr_jvp


def pytree_fc(weight_pytree_branch):
    fc_quaxpr = {}
    fc_quaxpr['op'] = Operation.FC
    fc_quaxpr['branch'] = weight_pytree_branch

    return fc_quaxpr





