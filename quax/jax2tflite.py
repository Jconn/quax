import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten
from tflite_schema_py_generated import (Model, SubGraph, Tensor, OperatorCode,
                                        Buffer, Operator, BuiltinOperator, 
                                        BuiltinOptions, FullyConnectedOptions,
                                        ActivationFunctionType)

from quax.tflite_utils import (get_empty_buffer, add_buffer, add_tensor, 
                            add_empty_tensor, add_tensor_with_buffer, add_fc_layer,
                                add_add_layer, create_subgraph, create_model,create_signature_def,
                               export_tflite, create_runtime_metadata,create_conversion_metadata, add_reshape_layer, add_relu_layer,add_activation_layer,
                               add_quantization_params)
import flatbuffers
from dataclasses import dataclass
import logging
from quax.quax import Operation
import numpy as np

#converts a jax model to a flatbuffer 
class FBB:
    def __init__(self):
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





    def convert(self, model, params, inputs):
        #model_jaxpr = jax.make_jaxpr(model.apply)(params, inputs)
        x = model.apply(params, inputs,rngs={'params': jax.random.key(0)}, mutable=False )
        #mutable=False fails here, something about the transform doesn't like it
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
            self.eqn_handler(eqn[0].primitive.name)

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
        graph_id = eqns[0].params['graph_id']
        eqns.pop(0)
        dg_eqn = eqns.pop(0)
        weight_var = dg_eqn.invars[1]
        in_var = dg_eqn.invars[0]
        if eqns[0].primitive.name == 'reshape':
            assert eqns[1].primitive.name == 'add'
            bias_eqn = eqns.pop(1)
            bias_var = bias_eqn.invars[1]
            eqns.pop(0)
            out_var = bias_eqn.outvars[0]
        else:
            bias_eqn = None
            bias_var = None
            out_var = dg_eqn.outvars[0]

        #now possibly have the activation eqn
        if len(eqns) > 1:
            activation_eqn = eqns.pop(0)
        else:
            activation_eqn = None
        assert len(eqns) == 1 

        flat_tree = jax.tree_util.tree_flatten_with_path(self.model_params['quax'])[0]
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
        assert root_path, "didn't find path for this now what"
        weight_path = self.model_params['aqt']
        quax_path = self.model_params['quax']
        #traverse the tree
        def bits_to_type(bits):
            if bits <= 8:
                dtype = np.int8
            elif bits <= 16:
                dtype = np.int16
            elif bits <= 32:
                dtype = np.int32
            elif bits <= 64:
                dtype = np.int64
            return dtype
        for segment in root_path:
            weight_path = weight_path[segment.key]
            quax_path = quax_path[segment.key]

        activation_bits = quax_path['bits'][0]['lhs']
        weight_bits = quax_path['bits'][0]['rhs']
        out_bits = quax_path['bits'][0]['out']

        activation_dtype = bits_to_type(activation_bits) 
        weight_dtype = bits_to_type(weight_bits) 
        out_dtype = bits_to_type(out_bits) 

        bias_dtype = bits_to_type(weight_bits + activation_bits + 16)

        #now we have our tensors 
        out_quantize = weight_path['output']  
        in_quantize = weight_path['AqtDotGeneral_0']['qlhs'] #this should be unused
        weight_quantize = weight_path['AqtDotGeneral_0']['qrhs']
        weight = weight_quantize['frozen'].qvalue
        weight_scale = weight_quantize['frozen'].scale[0]
        #TODO - why are we indexing into weight scale
        dequantized_weights = weight * weight_scale[0]
        weight_mins = jnp.min(dequantized_weights, axis=0)
        weight_maxs = jnp.max(dequantized_weights, axis=0)
        #TODO - aqt has no zero point quant support
        weight_zero_point = jnp.zeros(weight_scale.shape, dtype=np.int32)


        act_var = dg_eqn.invars[0]
        act_key =  str(act_var)
        if act_key in self.tensor_act_map.keys():
            activation_tensor = self.tensor_act_map[act_key]
        else:
            #if we can't find the activation tensor, we just make it up on the spot
            act_scale = in_quantize['frozen'].scale[0]
            zero_point = jnp.zeros(act_scale.shape, dtype=np.int32)
            act_qparams = add_quantization_params(self.builder, None, None, act_scale, zero_point, quantized_dim = 0)
            activation_tensor = add_empty_tensor(self.builder, "activation", act_var.aval.shape, self.buffers, quantization_params = act_qparams, dtype = activation_dtype)
            logging.warning("making up the activation quant params")


        weight_qparams = add_quantization_params(self.builder, weight_mins, weight_maxs, weight_scale, weight_zero_point, quantized_dim = 0)
            
        #TODO why does weight have to be transposed
        weight_tensor = add_tensor(self.builder, "weight", jnp.transpose(weight), self.buffers, quantization_params = weight_qparams, dtype = weight_dtype)
        
        if bias_eqn:
            #now we gotta be weird
            bias = weight_path['bias']
            bias_scale =  bias.scale[0]
            bias_weight = bias.qvalue
            zero_point = jnp.zeros(bias_scale.shape, dtype=np.int32)
            #TODO - convert bias dtype to correct value - is a fn of 8x8 or 8x16 or 16x16
            bias_weight = jnp.array(bias_weight, dtype=bias_dtype)
            
            bias_qparams = add_quantization_params(self.builder, None, None, bias_scale, zero_point, quantized_dim = 0)
            bias_tensor = add_tensor(self.builder, "bias", bias_weight, self.buffers, quantization_params = bias_qparams, dtype = bias_dtype)
        
        #record the output tensor

        #TODO deal with possible fused activation
        import pdb; pdb.set_trace()
        out_scale = in_quantize['frozen'].scale[0]
        zero_point = jnp.zeros(act_scale.shape, dtype=np.int32)
        out_qparams = add_quantization_params(self.builder, None, None, out_scale, zero_point, quantized_dim = 0)
        out_tensor = add_empty_tensor(self.builder, "activation", out_var.aval.shape, self.buffers, quantization_params = out_qparams,dtype=out_dtype)

        self.record_activation(in_var, activation_tensor)
        self.record_activation(out_var, out_tensor)
        self.record_weight(bias_var, bias_tensor)
        self.record_weight(weight_var, weight_tensor)
        op = add_fc_layer(self.builder,input_tensor=activation_tensor, weight_tensor=weight_tensor,bias_tensor=bias_tensor,output_tensor=out_tensor, bias_dtype = bias_dtype, all_tensors=self.tensors, all_opcodes=self.opcodes) 
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

    def record_activation(self,var, tensor):
        self.tensors.append(tensor)
        self.tensor_act_map[str(var)] = tensor

    def record_weight(self, var, tensor):
        self.tensors.append(tensor)
        self.tensor_weight_map[str(var)] = tensor

    def eqn_handler(self, eqn):
        self.handlers[str(eqn.primitive)](eqn)

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

    def activation_handler(self, eqn, activation_name):
        in_tensors = self.process_invars(eqn)
        out_tensors = self.process_outvars(eqn)
        op = add_activation_layer(self.builder, activation_name, in_tensors[0], out_tensors[0], self.tensors, self.opcodes)
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
        out_tensors = self.process_outvars(eqn)
        op = add_reshape_layer(self.builder, in_tensors[0], out_tensors[0], eqn.outvars[0].aval.shape, self.tensors) 
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
    



