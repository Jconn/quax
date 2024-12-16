import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten
from tflite_schema_py_generated import (Model, SubGraph, Tensor, OperatorCode,
                                        Buffer, Operator, BuiltinOperator, 
                                        BuiltinOptions, FullyConnectedOptions,
                                        ActivationFunctionType, ActivationFunctionType)

from quax.tflite_utils import (get_empty_buffer, add_buffer, add_tensor, 
                            add_empty_tensor, add_tensor_with_buffer, add_fc_layer, add_conv_layer,
                                add_add_layer,add_mul_layer, create_subgraph, create_model,create_signature_def,
                               export_tflite, create_runtime_metadata,create_conversion_metadata, add_reshape_layer, add_relu_layer,add_activation_layer,
                               add_quantization_params, add_quant_layer)
import quax.tflite_utils as tflite_utils
import quax.tflite_utils as tflu 
import flatbuffers
from dataclasses import dataclass
import logging
from quax.quax import Operation, AppendedActivation 
from quax.quax_utils import bits_to_type 
import numpy as np
from quax.quaxpr import QUAXPR_NAME

def parse_quaxprs(model_jaxpr):
    quaxprs = []
    for eqn in model_jaxpr.eqns:
        if eqn.primitive.name == QUAXPR_NAME:
            quaxprs.append(eqn)
    return quaxprs


def make_quant_params(builder, qxt):
    aqt_weights = qxt.qx
    weight = aqt_weights.qvalue
    weight_scale = aqt_weights.scale[0]
    #TODO - why are we indexing into weight scale
    dequantized_weights = weight * weight_scale[0]
    target_shape = weight_scale.shape
    reduce_axes = tuple(i for i, (d1, d2) in enumerate(zip(dequantized_weights.shape, target_shape)) if d1 != d2)
    weight_mins = jnp.min(dequantized_weights, axis=reduce_axes, keepdims=True)
    weight_maxs = jnp.max(dequantized_weights, axis=reduce_axes, keepdims=True)

    #need to find the algorithm that matches the mins to the scales
    #TODO - aqt has no zero point quant support
    weight_zero_point = jnp.zeros(weight_scale.shape, dtype=np.int32)
    weight_qparams = add_quantization_params(builder, weight_mins, weight_maxs, weight_scale, weight_zero_point, quantized_dim = 0)
    return weight_qparams
        

def find_path(tree, graph_id):
    flat_tree = jax.tree_util.tree_flatten_with_path(tree)[0]
    root_path = None
    for route in flat_tree:
        path = route[0]
        path_id = route[1]
        if graph_id == path_id:
            root_path = path[:-2]
            break
            #we found our section, now we extract the path 
            #shave off the last two pieces of the graph
    #we found the root path
    return root_path

#converts a jax model to a flatbuffer 
class FBB:
    def __init__(self):
        self.quant_map = {}
        self.buffers = []
        self.tensors = []
        self.ops = []
        self.opcodes = []
        self.builder = flatbuffers.Builder(1024)
        self.tensor_act_map = {}
        self.tensor_weight_map = {}
        self.handlers = {}
        self.weight_buffer_map = {}
        self.handlers[Operation.FC] = self.fc_handler
        self.handlers[Operation.QUANTIZE] = self.quant_handler
        self.handlers[Operation.CONV] = self.conv_handler
        self.handlers[Operation.ACTIVATION] = self.activation_handler
        self.handlers[Operation.RESHAPE] = self.reshape_handler
        self.handlers[Operation.ADD] = self.add_handler
        self.handlers[Operation.MUL] = self.mul_handler
        self.handlers[Operation.CONCATENATE] = self.concat_handler





    def process_quax_op(self, op, quaxbegin, quaxend):
        self.handlers[op](quaxbegin, quaxend)

    def convert(self, model, params, inputs):
        #model_jaxpr = jax.make_jaxpr(model.apply)(params, inputs)
        x = model.apply(params, inputs,rngs={'params': jax.random.key(0)}, mutable=False )
        #mutable=False fails here, something about the transform doesn't like it
        #mutable=True also fails, it doesn't like that "Invalid Filter" but why
        model_jaxpr = jax.make_jaxpr(model.apply)(params, inputs,rngs={'params': jax.random.key(0)})
        invars = model_jaxpr.jaxpr.invars
        flat_params, _ = tree_flatten(params)
        param_map = {str(var): value for var, value in zip(invars[:len(flat_params)], flat_params)}
        self.model_params = params
        self.buffers.append(get_empty_buffer(self.builder))
        #for k,v in param_map.items():
        #    self.weight_buffer_map[k] = add_buffer(self.builder, self.buffers, data = v)

        #for eqn in model_jaxpr.eqns:
        #    self.eqn_handler(eqn)
        quaxpr_eqns =  parse_quaxprs(model_jaxpr)

        for quaxbegin, quaxend in zip(quaxpr_eqns[::2], quaxpr_eqns[1::2]):
            op = quaxbegin.params['quax_pytree']['op']
            self.process_quax_op(op, quaxbegin, quaxend)
        
        sg_in, sg_out = self.find_model_io(model_jaxpr)
        subgraphs = []
        subgraphs.append(create_subgraph(self.builder, self.tensors, sg_in, sg_out, self.ops, subgraph_name="main"))
        signature_defs = []
        #TODO - assuming a single subgraph index
        sig_def = create_signature_def(self.builder, sg_in, sg_out, self.tensors, 0)
        signature_defs.append(sig_def)
        metadata_list = []
        metadata = create_runtime_metadata(self.builder, self.buffers)
        metadata_list.append(metadata)
        metadata_list.append(create_conversion_metadata(self.builder, self.buffers))

        model = create_model(self.builder, subgraphs, self.opcodes, self.buffers, signature_defs, metadata_list)
        tflite = export_tflite(self.builder, model)
        return tflite


    def parse_composite_eqns(self, eqns):
        composite_eqns = []
        composite_eqn = None
        composite_marker = 'marker'

        for eqn in eqns:
            #check for if the eqn is a composite eqn
            if eqn.primitive.name == composite_marker:
                if composite_eqn is None:
                    composite_eqn = []
                    composite_eqn.append(eqn)
                else:
                    composite_eqn.append(eqn)
                    composite_eqns.append(composite_eqn)
                    composite_eqn = None
            elif composite_eqn is not None:
                #we are in a composite eqn
                composite_eqn.append(eqn)
            else:
                #we are not in a composite eqn - so the composite is just one 
                composite_eqns.append([eqn])
        return composite_eqns

    def composite_eqn_handler(self, eqn):
        #check if our first is a marker
        if eqn[0].primitive.name == 'marker':
            self.handlers[eqn[0].params['op_id']](eqn)
        else:
            self.eqn_handler(eqn[0])

    
    def parse_details(self, eqns):
        graph_id = eqns[0].params['graph_id']
        root_path = find_path(self.model_params['quax'], graph_id)
        #traverse the tree
        weight_path = self.model_params['aqt']
        quax_path = self.model_params['quax']
        for segment in root_path:
            weight_path = weight_path[segment.key]
            quax_path = quax_path[segment.key]
        return weight_path, quax_path

    def make_empty_tensor(self, qxt, shape = None):
        scale = qxt.qx.scale[0] 
        zero_point = self.parse_zp(qxt)
        dtype = bits_to_type(qxt.bits) 
        if shape is None:
            shape = qxt.shape
        out_qparams = add_quantization_params(self.builder, None, None, scale, zero_point, quantized_dim = 0)
        tensor = add_empty_tensor(self.builder, "activation", shape, self.buffers, quantization_params = out_qparams,dtype=dtype)

        return tensor

    def add_handler(self, quaxbegin, quaxend):
        #add has two input vars, one output var

        out_qxt = self.get_quaxtensor(quaxbegin, quaxbegin.params['quax_pytree']['op_name'])
        out_tensor = self.make_empty_tensor(out_qxt) 
        in_tensors = [self.tensor_act_map[str(x)] for x in quaxbegin.invars]
        out_var = quaxend.invars[0]

        self.record_activation(out_var, out_tensor)

        op = add_add_layer(self.builder,in_tensors[0], in_tensors[1],out_tensor, all_tensors=self.tensors, all_opcodes=self.opcodes) 
        self.record_op(op)

    def concat_handler(self, quaxbegin, quaxend):
        out_var = quaxend.invars[0]
        out_qxt = self.get_quaxtensor(quaxbegin, quaxbegin.params['quax_pytree']['op_name'])
        out_tensor = self.make_empty_tensor(out_qxt, shape = out_var.aval.shape) 
        in_tensors = [self.tensor_act_map[str(x)] for x in quaxbegin.invars]
        axis = quaxbegin.params['quax_pytree']['axis']

        self.record_activation(out_var, out_tensor)
        op = tflu.add_concat_layer(self.builder,in_tensors,out_tensor, axis,all_tensors=self.tensors, all_opcodes=self.opcodes) 
        self.record_op(op)

    def mul_handler(self, quaxbegin, quaxend):
        out_qxt = self.get_quaxtensor(quaxbegin, quaxbegin.params['quax_pytree']['op_name'])
        out_tensor = self.make_empty_tensor(out_qxt) 
        in_tensors = [self.tensor_act_map[str(x)] for x in quaxbegin.invars]
        out_var = quaxend.invars[0]

        self.record_activation(out_var, out_tensor)

        op = add_mul_layer(self.builder,in_tensors[0], in_tensors[1],out_tensor, all_tensors=self.tensors, all_opcodes=self.opcodes) 
        self.record_op(op)
        

    def reshape_handler(self, quaxbegin, quaxend):
        #TODO - need to grab the quantization details from the input tensor
        invar = quaxbegin.invars[0]
        outvar = quaxend.invars[0]
        in_tensor = self.get_activation_tensor(invar)

        out_tensors = []
        input_qparams, input_dtype = self.quant_map[str(invar)]
        out_tensor = add_empty_tensor(self.builder, "activation", outvar.aval.shape, self.buffers, quantization_params = input_qparams, dtype = input_dtype)
        self.record_activation(outvar, out_tensor)
        out_tensors.append(out_tensor)

        op = add_reshape_layer(self.builder, in_tensor, out_tensor, outvar.aval.shape, self.tensors, all_opcodes = self.opcodes) 
        self.record_op(op)

    def conv_handler(self, quaxbegin, quaxend):

        out_qxt = self.get_quaxtensor(quaxend, 'output')
        weight_qxt = self.get_quaxtensor(quaxend, 'kernel')
        has_bias = True

        bias_qxt = self.get_quaxtensor(quaxend, 'bias')
        if bias_qxt == None:
            has_bias = False
            bias_tensor = None

        in_var = quaxbegin.invars[0]
        out_var = quaxend.invars[0]
    
        out_dtype = bits_to_type(out_qxt.bits) 
        weight_dtype = bits_to_type(weight_qxt.bits) 
        bias_dtype = bits_to_type(weight_qxt.bits + out_qxt.bits + 16)

        #TODO - aqt has no zero point quant support
        weight_qparams = make_quant_params(self.builder, weight_qxt)
        #TODO why does weight have to be transposed
        weight = weight_qxt.quantized_tensor() 
        weight = weight.astype(weight_dtype)
        strides = quaxend.params['quax_pytree']['window_strides']
        lhs_dilation = quaxend.params['quax_pytree']['lhs_dilation']
        rhs_dilation = quaxend.params['quax_pytree']['rhs_dilation']
        padding = quaxend.params['quax_pytree']

        #set weights to be in shape (OC, KH, KW, IC)
        weight = jnp.transpose(weight, [3,0,1,2])
        weight_tensor = add_tensor(self.builder, "weight", weight, self.buffers, quantization_params = weight_qparams, dtype = weight_dtype)
        self.record_weight(weight_tensor)

        act_key = str(in_var)
        if act_key in self.tensor_act_map.keys():
            activation_tensor = self.tensor_act_map[act_key]
        else:
            raise Exception(f"couldn't find activation tensor with key {act_key}")
        if has_bias:
            #now we gotta be weird
            bias_weight = bias_qxt.quantized_tensor()
            bias_weight = np.array(bias_weight, dtype=bias_dtype)
            
            bias_qparams = make_quant_params(self.builder, bias_qxt)
            bias_tensor = add_tensor(self.builder, "bias", bias_weight, self.buffers, quantization_params = bias_qparams, dtype = bias_dtype)
            self.record_weight(bias_tensor)
        
        out_scale = out_qxt.qx.scale[0] 
        zero_point = self.parse_zp(out_qxt)
        out_qparams = add_quantization_params(self.builder, None, None, out_scale, zero_point, quantized_dim = 0)
        out_tensor = add_empty_tensor(self.builder, "activation", out_var.aval.shape, self.buffers, quantization_params = out_qparams,dtype=out_dtype)

        self.record_activation(in_var, activation_tensor)
        #TODO - this is a mess..
        self.record_activation(out_var, out_tensor)
        self.record_quantization_details(out_var, (out_qparams, out_dtype) )
        #self.record_weight(weight_var, weight_tensor)
        activation_op = tflite_utils.map_appended_activation(quaxend.params['quax_pytree']['act_fn'])
        #TODO - how to record weight now
        #self.record_weight(bias_var, bias_tensor)
        #self.record_weight(weight_var, weight_tensor)
        op = add_conv_layer(self.builder,input_tensor=activation_tensor, weight_tensor=weight_tensor,bias_tensor=bias_tensor,output_tensor=out_tensor, bias_dtype = bias_dtype, all_tensors=self.tensors, all_opcodes=self.opcodes,activation_op=activation_op, quax_params=quaxend.params['quax_pytree']) 
        self.record_op(op)

    
    def fc_handler(self, quaxbegin, quaxend):
        '''
        structure of fc ops should be
        marker
        dg
        optional bias reshape
        bias add
        optional activation
        marker end
        
        still need to associate the weights and the activations with eachother
        '''
        #weight_path, quax_path = self.parse_details(eqns)

        in_var = quaxbegin.invars[0]
        out_var = quaxend.invars[0]
    

        has_bias = True
        out_qxt = self.get_quaxtensor(quaxend, 'output')
        weight_qxt = self.get_quaxtensor(quaxend, 'kernel')
        bias_qxt = self.get_quaxtensor(quaxend, 'bias')
        if bias_qxt is None:
            has_bias = False
            bias_tensor = None

        out_dtype = bits_to_type(out_qxt.bits) 
        weight_dtype = bits_to_type(weight_qxt.bits) 
        bias_dtype = bits_to_type(weight_qxt.bits + out_qxt.bits + 16)

        weight = weight_qxt.quantized_tensor() 
        weight = weight.astype(weight_dtype)
        #TODO - why are we indexing into weight scale

        #TODO - aqt has no zero point quant support
        weight_qparams = make_quant_params(self.builder, weight_qxt)

        act_var = in_var 
        act_key =  str(act_var)
        if act_key in self.tensor_act_map.keys():
            activation_tensor = self.tensor_act_map[act_key]
        else:
            raise Exception(f"couldn't find activation tensor with key {act_key}")
            
        #TODO why does weight have to be transposed
        weight_tensor = add_tensor(self.builder, "weight", jnp.transpose(weight), self.buffers, quantization_params = weight_qparams, dtype = weight_dtype)
        self.record_weight(weight_tensor)
        
        if has_bias:
            #now we gotta be weird
            bias_weight = bias_qxt.quantized_tensor()
            bias_weight = np.array(bias_weight, dtype=bias_dtype)
            
            bias_qparams = make_quant_params(self.builder, bias_qxt)
            bias_tensor = add_tensor(self.builder, "bias", bias_weight, self.buffers, quantization_params = bias_qparams, dtype = bias_dtype)
            self.record_weight(bias_tensor)
            #TODO -recording weight?
            #self.record_weight(bias_var, bias_tensor)
        
        #record the output tensor

        #TODO - this is incorrect I thin
        out_scale = out_qxt.qx.scale[0] 
        zero_point = self.parse_zp(out_qxt)
        out_qparams = add_quantization_params(self.builder, None, None, out_scale, zero_point, quantized_dim = 0)
        out_tensor = add_empty_tensor(self.builder, "activation", out_var.aval.shape, self.buffers, quantization_params = out_qparams,dtype=out_dtype)

        self.record_activation(in_var, activation_tensor)
        #TODO - this is a mess..
        self.record_activation(out_var, out_tensor)
        self.record_quantization_details(out_var, (out_qparams, out_dtype) )

        #self.record_weight(weight_var, weight_tensor)
        activation_op = tflite_utils.map_appended_activation(quaxend.params['quax_pytree']['act_fn'])
        op = add_fc_layer(self.builder,input_tensor=activation_tensor, weight_tensor=weight_tensor,bias_tensor=bias_tensor,output_tensor=out_tensor, bias_dtype = bias_dtype, activation_op = activation_op ,all_tensors=self.tensors, all_opcodes=self.opcodes) 
        self.record_op(op)

    def get_quaxtensor(self, quaxop, var_name):
        weight_path =  quaxop.params['quax_pytree']['branch'] 
        #now walk weight path 
        param = self.model_params
        for link in weight_path:
            param = param[link]
        if var_name not in param.keys():
            return None
        return param[var_name]
       

    def parse_zp(self, quaxtensor):
        #soon zeropoint support..
        out_scale = quaxtensor.qx.scale[0]
        out_zp = jnp.zeros(out_scale.shape, dtype=np.int32)
        return out_zp

    def quant_handler(self, quaxbegin, quaxend):
        if 'op_name' in quaxend.params['quax_pytree'].keys():
            op_name = quaxend.params['quax_pytree']['op_name']
        else:
            op_name = 'output'

        quaxtensor = self.get_quaxtensor(quaxend, op_name)
        
        out_dtype = bits_to_type(quaxtensor.bits) 
        out_scale = quaxtensor.qx.scale[0]
        out_zp = self.parse_zp(quaxtensor) 
        in_var = quaxbegin.invars[0]
        quant_var = quaxend.invars[0]

        out_qparams = add_quantization_params(self.builder, None, None, out_scale, out_zp, quantized_dim = 0)
        #TODO - correct datatype for input tensor

        if str(in_var) in self.tensor_act_map.keys():
            in_tensor = self.tensor_act_map[str(in_var)]
        else:
            in_tensor = add_empty_tensor(self.builder, "activation", quant_var.aval.shape, self.buffers, quantization_params = None, dtype = np.float32)
        self.record_activation(in_var, in_tensor)
        out_tensor = add_empty_tensor(self.builder, "activation", quant_var.aval.shape, self.buffers, quantization_params = out_qparams, dtype = out_dtype)

        self.record_quantization_details(quant_var, (out_qparams, out_dtype) )
        self.record_activation(quant_var, out_tensor)
        #also map the invar to this tensor, because quant is special
        #self.record_activation(quaxbegin.invars[0], out_tensor)
        #TODO - add quant op
        op = add_quant_layer(self.builder,input_tensor=in_tensor,output_tensor=out_tensor, all_tensors=self.tensors, all_opcodes=self.opcodes) 
        self.record_op(op)

    def find_model_io(self, model_jaxpr):
        activation_inputs = []
        activation_outputs = []
        for invar in model_jaxpr.jaxpr.invars:
            if self.is_recorded_activation(invar):
                activation_inputs.append(self.get_activation_tensor(invar))
        for outvar in model_jaxpr.jaxpr.outvars:
            if self.is_recorded_activation(outvar):
                activation_outputs.append(self.get_activation_tensor(outvar))
        return activation_inputs, activation_outputs
        

    def is_recorded_activation(self, var):
        return str(var) in self.tensor_act_map.keys()

    def get_activation_tensor(self, var):
        return self.tensor_act_map[str(var)]
    
    def record_quantization_details(self, var, quant_details):
        self.quant_map[str(var)] = quant_details 

    def record_activation(self,var, tensor):
        self.tensors.append(tensor)
        self.tensor_act_map[str(var)] = tensor

    def record_weight(self, tensor):
        self.tensors.append(tensor)

    def eqn_handler(self, eqn):
        print(f"eqn {eqn}")
        self.handlers[str(eqn.primitive.name)](eqn)

    def record_op(self, op):
        self.ops.append(op)

    def is_weight(self, invar):
        return str(invar) in self.weight_buffer_map.keys()
    def get_weight_buffer(self, invar):
        return self.weight_buffer_map[str(invar)]
    def assign_weight_buffer(self, invar, outvar):
     self.weight_buffer_map[str(outvar)] = self.weight_buffer_map[str(invar)]

    def process_invars(self, eqn):
        in_tensors = []
        for invar in eqn.invars:
            if self.is_weight(invar):
                logging.debug(f"recording weight for {eqn}")
                weight_buffer = self.get_weight_buffer(invar)
                weight_shape = invar.aval.shape
                in_tensor = add_tensor_with_buffer(self.builder, "weight", weight_shape, weight_buffer, self.buffers)
                self.record_weight(invar, in_tensor)
            elif not self.is_recorded_activation(invar):
                in_tensor = add_empty_tensor(self.builder, "add_invar", invar.aval.shape, self.buffers)
                self.record_activation(invar, in_tensor)
            else:
                in_tensor = self.get_activation_tensor(invar)
                self.record_activation(invar, in_tensor)
            in_tensors.append(in_tensor)
        return in_tensors

    def process_outvars(self, eqn):
        out_tensors = []
        for outvar in eqn.outvars:
            out_tensor = add_empty_tensor(self.builder, "add_outvar", outvar.aval.shape, self.buffers)
            self.record_activation(eqn.outvars[0], out_tensor)
            out_tensors.append(out_tensor)
        return out_tensors

    def custom_jvp_handler(self, eqn):
        #custom_op = eqn.params['call_jaxpr'].jaxpr.eqns[0].params['name']
        #activation_handler = {}
        #activation_handler['relu'] = self.relu_handler
        self.activation_handler(eqn, activation_name=eqn.params['call_jaxpr'].jaxpr.eqns[0].params['name'])

    def tanh_handler(self, eqn):
        self.activation_handler(eqn, str(eqn.primitive))

    def activation_handler(self, eqns):
        weight_path, quax_path = self.parse_details(eqns)
        act_eqn = eqns[1]
        in_tensors = self.process_invars(act_eqn)

        activation_dtype, weight_dtype, out_dtype, bias_dtype = self.construct_mathy_dtypes(quax_path)

        out_quantize = weight_path['output']
        out_bits = quax_path['bits'][0]['out']
        out_dtype = bits_to_type(out_bits)

        out_scale = out_quantize.scale[0]
        out_var = act_eqn.outvars[0]
        zero_point = jnp.zeros(out_scale.shape, dtype=np.int32)
        out_qparams = add_quantization_params(self.builder, None, None, out_scale, zero_point, quantized_dim = 0)
        out_tensor = add_empty_tensor(self.builder, "activation", out_var.aval.shape, self.buffers, quantization_params = out_qparams,dtype=out_dtype)

        self.record_activation(act_eqn.outvars[0], out_tensor)
        self.record_quantization_details(act_eqn.outvars[0], (out_qparams, out_dtype))
        def eqn_to_tflite_op(eqn):
            key = eqn.primitive.name
            eqn_map = {}
            eqn_map['tanh'] = BuiltinOperator.BuiltinOperator().TANH
            eqn_map['logistic'] = BuiltinOperator.BuiltinOperator().LOGISTIC
            return eqn_map[key]
        activation_operator = eqn_to_tflite_op(act_eqn)


        op = tflite_utils.add_activation_layer(self.builder, in_tensors[0], out_tensor, activation_operator, self.tensors, self.opcodes)
        self.record_op(op)


    def relu_handler(self, eqn):
        in_tensors = self.process_invars(eqn)
        out_tensors = self.process_outvars(eqn)
        op = add_relu_layer(self.builder, in_tensors[0], out_tensors[0], self.tensors, self.opcodes)
        self.record_op(op)

    



