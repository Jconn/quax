import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten
from tflite_schema_py_generated import (Model, SubGraph, Tensor, OperatorCode,
                                        Buffer, Operator, BuiltinOperator, 
                                        BuiltinOptions, FullyConnectedOptions,
                                        ActivationFunctionType, ActivationFunctionType)

from quax.tflite_utils import (get_empty_buffer, add_buffer, add_tensor, 
                            add_empty_tensor, add_tensor_with_buffer, add_fc_layer, add_conv_layer, map_activation_eqn,
                                add_add_layer, create_subgraph, create_model,create_signature_def,
                               export_tflite, create_runtime_metadata,create_conversion_metadata, add_reshape_layer, add_relu_layer,add_activation_layer,
                               add_quantization_params)
import quax.tflite_utils as tflite_utils
import flatbuffers
from dataclasses import dataclass
import logging
from quax.quax import Operation
from quax.quax_utils import bits_to_type 
import numpy as np
    

def make_quant_params(builder, aqt_weights):
    weight = aqt_weights.qvalue
    weight_scale = aqt_weights.scale[0]
    #TODO - why are we indexing into weight scale
    dequantized_weights = weight * weight_scale[0]
    weight_mins = jnp.min(dequantized_weights, axis=0)
    weight_maxs = jnp.max(dequantized_weights, axis=0)
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
        self.handlers['dot_general'] = self.dot_general_handler
        self.handlers['add'] = self.add_handler
        self.handlers['reshape'] = self.reshape_handler
        self.handlers['custom_jvp_call'] = self.custom_jvp_handler
        self.handlers['tanh'] = self.tanh_handler
        self.handlers[Operation.FC.value] = self.fc_handler
        self.handlers[Operation.QUANTIZE.value] = self.quant_handler
        self.handlers[Operation.CONV.value] = self.conv_handler
        self.handlers[Operation.ACTIVATION.value] = self.activation_handler





    def convert(self, model, params, inputs):
        #model_jaxpr = jax.make_jaxpr(model.apply)(params, inputs)
        x = model.apply(params, inputs,rngs={'params': jax.random.key(0)}, mutable=False )
        #mutable=False fails here, something about the transform doesn't like it
        model_jaxpr = jax.make_jaxpr(model.apply)(params, inputs,rngs={'params': jax.random.key(0)})
        import pdb; pdb.set_trace()
        invars = model_jaxpr.jaxpr.invars
        flat_params, _ = tree_flatten(params)
        param_map = {str(var): value for var, value in zip(invars[:len(flat_params)], flat_params)}
        self.model_params = params
        self.buffers.append(get_empty_buffer(self.builder))
        #for k,v in param_map.items():
        #    self.weight_buffer_map[k] = add_buffer(self.builder, self.buffers, data = v)

        #for eqn in model_jaxpr.eqns:
        #    self.eqn_handler(eqn)
        
        parsed_eqns = self.parse_composite_eqns(model_jaxpr.eqns)
        for composite_eqn in parsed_eqns:
            self.composite_eqn_handler(composite_eqn)
        
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

    def conv_handler(self, eqns):
        weight_path, quax_path = self.parse_details(eqns)
        #marker_eqn = eqns.pop(0)
        conv_eqn, bias_eqn, activation_eqn = self.parse_mathy_composite_details(eqns)
        weight_var = conv_eqn.invars[1]
        in_var = conv_eqn.invars[0]
        if bias_eqn:
            bias_var = bias_eqn.invars[1]
            out_var = bias_eqn.outvars[0]
        else:
            out_var = conv_eqn.outvars[0]
    
        activation_dtype, weight_dtype, out_dtype, bias_dtype = self.construct_mathy_dtypes(quax_path)
        #TODO - aqt has no zero point quant support
        weight_quantize = weight_path['AqtConvGeneralDilated_0']['qrhs']
        weight_qparams = make_quant_params(self.builder, weight_quantize['frozen'])
        #TODO why does weight have to be transposed
        weight = weight_quantize['frozen'].qvalue
        weight = weight.astype(weight_dtype)
        strides = conv_eqn.params['window_strides']
        lhs_dilation = conv_eqn.params['lhs_dilation']
        rhs_dilation = conv_eqn.params['rhs_dilation']

        #set weights to be in shape (OC, KW, KH, IC)
        weight = jnp.transpose(weight, [3,0,1,2])
        weight_tensor = add_tensor(self.builder, "weight", weight, self.buffers, quantization_params = weight_qparams, dtype = weight_dtype)

        act_key = str(in_var)
        if act_key in self.tensor_act_map.keys():
            activation_tensor = self.tensor_act_map[act_key]
        else:
            raise Exception(f"couldn't find activation tensor with key {act_key}")
        if bias_eqn:
            #now we gotta be weird
            bias = weight_path['bias']
            bias_weight = bias.qvalue
            bias_weight = np.array(bias_weight, dtype=bias_dtype)
            
            bias_qparams = make_quant_params(self.builder, bias)
            bias_tensor = add_tensor(self.builder, "bias", bias_weight, self.buffers, quantization_params = bias_qparams, dtype = bias_dtype)
        

        act_quantize = weight_path['AqtConvGeneralDilated_0']['qlhs']
        out_scale = act_quantize['frozen'].scale[0]
        zero_point = jnp.zeros(out_scale.shape, dtype=np.int32)
        out_qparams = add_quantization_params(self.builder, None, None, out_scale, zero_point, quantized_dim = 0)
        out_tensor = add_empty_tensor(self.builder, "activation", out_var.aval.shape, self.buffers, quantization_params = out_qparams,dtype=out_dtype)

        self.record_activation(in_var, activation_tensor)
        #TODO - this is a mess..
        self.record_activation(out_var, out_tensor)
        self.record_quantization_details(out_var, (out_qparams, out_dtype) )
        self.record_weight(bias_var, bias_tensor)
        self.record_weight(weight_var, weight_tensor)
        op = add_conv_layer(self.builder,input_tensor=activation_tensor, weight_tensor=weight_tensor,bias_tensor=bias_tensor,output_tensor=out_tensor, bias_dtype = bias_dtype, all_tensors=self.tensors, all_opcodes=self.opcodes) 
        self.record_op(op)

    def parse_mathy_composite_details(self, eqns):
        math_eqn = eqns[1]
        if eqns[2].primitive.name == 'reshape':
            assert eqns[3].primitive.name == 'add'
            bias_eqn = eqns[3]
        else:
            bias_eqn = None

        #now possibly have the activation eqn
        if len(eqns) > 5:
            activation_eqn = eqns[4]
        else:
            activation_eqn = None

        return math_eqn, bias_eqn, activation_eqn

    def construct_mathy_dtypes(self, quax_path):
        activation_bits = quax_path['bits'][0]['lhs']
        weight_bits = quax_path['bits'][0]['rhs']
        out_bits = quax_path['bits'][0]['out']

        activation_dtype = bits_to_type(activation_bits) 
        weight_dtype = bits_to_type(weight_bits) 
        out_dtype = bits_to_type(out_bits) 
        bias_dtype = bits_to_type(weight_bits + activation_bits + 16)
        return activation_dtype, weight_dtype, out_dtype, bias_dtype

    def fc_handler(self, eqns):
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
        weight_path, quax_path = self.parse_details(eqns)
        dg_eqn, bias_eqn, activation_eqn = self.parse_mathy_composite_details(eqns)
        weight_var = dg_eqn.invars[1]
        in_var = dg_eqn.invars[0]

        if activation_eqn:
            out_var = activation_eqn.outvars[0]
        elif bias_eqn:
            out_var = bias_eqn.outvars[0]
        else:
            out_var = dg_eqn.outvars[0]


        activation_dtype, weight_dtype, out_dtype, bias_dtype = self.construct_mathy_dtypes(quax_path)

        #now we have our tensors 
        in_quantize = weight_path['AqtDotGeneral_0']['qlhs'] #this should be unused
        out_quantize = weight_path['output']
        weight_quantize = weight_path['AqtDotGeneral_0']['qrhs']
        weight = weight_quantize['frozen'].qvalue
        #TODO - why are we indexing into weight scale

        #TODO - aqt has no zero point quant support
        weight_qparams = make_quant_params(self.builder, weight_quantize['frozen'])

        act_var = dg_eqn.invars[0]
        act_key =  str(act_var)
        if act_key in self.tensor_act_map.keys():
            activation_tensor = self.tensor_act_map[act_key]
        else:
            raise Exception(f"couldn't find activation tensor with key {act_key}")
            
        #TODO why does weight have to be transposed
        weight_tensor = add_tensor(self.builder, "weight", jnp.transpose(weight), self.buffers, quantization_params = weight_qparams, dtype = weight_dtype)
        
        if bias_eqn:
            #now we gotta be weird
            bias_var = bias_eqn.invars[1]
            bias = weight_path['bias']
            bias_weight = bias.qvalue
            bias_weight = np.array(bias_weight, dtype=bias_dtype)
            
            bias_qparams = make_quant_params(self.builder, bias)
            bias_tensor = add_tensor(self.builder, "bias", bias_weight, self.buffers, quantization_params = bias_qparams, dtype = bias_dtype)
            self.record_weight(bias_var, bias_tensor)
        
        #record the output tensor

        #TODO - this is incorrect I thin
        out_scale = out_quantize.scale[0]
        zero_point = jnp.zeros(out_scale.shape, dtype=np.int32)
        out_qparams = add_quantization_params(self.builder, None, None, out_scale, zero_point, quantized_dim = 0)
        out_tensor = add_empty_tensor(self.builder, "activation", out_var.aval.shape, self.buffers, quantization_params = out_qparams,dtype=out_dtype)

        self.record_activation(in_var, activation_tensor)
        #TODO - this is a mess..
        self.record_activation(out_var, out_tensor)
        self.record_quantization_details(out_var, (out_qparams, out_dtype) )

        self.record_weight(weight_var, weight_tensor)
        activation_op = map_activation_eqn(activation_eqn)
        op = add_fc_layer(self.builder,input_tensor=activation_tensor, weight_tensor=weight_tensor,bias_tensor=bias_tensor,output_tensor=out_tensor, bias_dtype = bias_dtype, activation_op = activation_op ,all_tensors=self.tensors, all_opcodes=self.opcodes) 
        self.record_op(op)


    def quant_handler(self, eqns):
        weight_path, quax_path = self.parse_details(eqns)

        marker_eqn = eqns.pop(0)
        quant_var = marker_eqn.invars[0]
        out_bits = quax_path['bits'][0]['out']
        out_dtype = bits_to_type(out_bits) 
        out_quantize = weight_path['output']  
        out_scale = out_quantize.scale[0]
        out_zp = jnp.zeros(out_scale.shape, dtype=np.int32)
        out_qparams = add_quantization_params(self.builder, None, None, out_scale, out_zp, quantized_dim = 0)
        out_tensor = add_empty_tensor(self.builder, "activation", quant_var.aval.shape, self.buffers, quantization_params = out_qparams, dtype = out_dtype)

        self.record_quantization_details(quant_var, (out_qparams, out_dtype) )
        self.record_activation(quant_var, out_tensor)
        #TODO - add quant op

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

    def record_weight(self, var, tensor):
        self.tensors.append(tensor)
        self.tensor_weight_map[str(var)] = tensor

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

    def reshape_handler(self, eqn):
        if self.is_weight(eqn.invars[0]):
            logging.warning("skipping weight reshape, not a tflite op")
            #but we can't skip, because we need to assign the output var to also be this weight
            self.assign_weight_buffer(eqn.invars[0], eqn.outvars[0])
            return
        in_tensors = self.process_invars(eqn)
        #TODO - need to grab the quantization details from the input tensor
        out_tensors = []
        input_qparams, input_dtype = self.quant_map[str(eqn.invars[0])]
        for outvar in eqn.outvars:
            out_tensor = add_empty_tensor(self.builder, "activation", outvar.aval.shape, self.buffers, quantization_params = input_qparams, dtype = input_dtype)
            self.record_activation(eqn.outvars[0], out_tensor)
            out_tensors.append(out_tensor)

        op = add_reshape_layer(self.builder, in_tensors[0], out_tensors[0], eqn.outvars[0].aval.shape, self.tensors, all_opcodes = self.opcodes) 
        self.record_op(op)
    
    def add_handler(self, eqn):
        in_tensors = self.process_invars(eqn)
        out_tensors = self.process_outvars(eqn)
        op = add_add_layer(self.builder,in_tensors[0], in_tensors[1],out_tensors[0], all_tensors=self.tensors, all_opcodes=self.opcodes) 
        self.record_op(op)

    def dot_general_handler(self, eqn):
        weight_buffer = self.weight_buffer_map[str(eqn.invars[1])]
        #TODO - why does weight shape need to be reversed
        weight_shape = [x for x in reversed(eqn.invars[1].aval.shape)]
        weight_tensor = add_tensor_with_buffer(self.builder, "weight", weight_shape, weight_buffer, self.buffers)

        if not self.is_recorded_activation(eqn.invars[0]):
            in_tensor = add_empty_tensor(self.builder, "in_tensor", eqn.invars[0].aval.shape, self.buffers)
        else:
            in_tensor = self.get_activation_tensor(eqn.invars[0])
        out_tensor = add_empty_tensor(self.builder, "out_tensor", eqn.outvars[0].aval.shape, self.buffers)

        self.record_activation(eqn.invars[0],in_tensor )
        self.record_weight(eqn.invars[1], weight_tensor)

        self.record_activation(eqn.outvars[0],out_tensor)

        op = add_fc_layer(self.builder,input_tensor=in_tensor, weight_tensor=weight_tensor,bias_tensor=None,output_tensor=out_tensor, all_tensors=self.tensors, all_opcodes=self.opcodes) 
        self.record_op(op)

        #TODO - need to map the from the model invars and outvars to the subgraph input and output
        #we can do this by mapping the created tensors to invars and outvars
        #but the invars are also weights, so we need to be smart about the mapping there
        #(e.g), anything that is not a weight is an invar 
        #need an activation map for tensors
    



