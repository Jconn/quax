import jax
from jax.tree_util import tree_flatten
from tflite_schema_py_generated import (Model, SubGraph, Tensor, OperatorCode,
                                        Buffer, Operator, BuiltinOperator, 
                                        BuiltinOptions, FullyConnectedOptions,
                                        ActivationFunctionType)

from quax.tflite_utils import (get_empty_buffer, add_buffer, add_tensor, 
                            add_empty_tensor, add_tensor_with_buffer, add_fc_layer,
                                add_add_layer, create_subgraph, create_model,create_signature_def,
                               export_tflite, create_runtime_metadata,create_conversion_metadata )
import flatbuffers
from dataclasses import dataclass
import logging

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

    def convert(self, model, params, inputs):
        model_jaxpr = jax.make_jaxpr(model.apply)(params, inputs)
        invars = model_jaxpr.jaxpr.invars
        flat_params, _ = tree_flatten(params)
        param_map = {str(var): value for var, value in zip(invars[:len(flat_params)], flat_params)}
        self.buffers.append(get_empty_buffer(self.builder))
        for k,v in param_map.items():
            self.weight_buffer_map[k] = add_buffer(self.builder, self.buffers, data = v)
        for eqn in model_jaxpr.eqns:
            logging.debug(f"trying eqn {eqn}")
            self.eqn_handler(eqn)
        
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

    def record_op(self, op, opcode):
        self.ops.append(op)
        self.opcodes.append(opcode)
    def is_weight(self, invar):
        return str(invar) in self.weight_buffer_map.keys()
    def get_weight_buffer(self, invar):
        return self.weight_buffer_map[str(invar)]

    def add_handler(self, eqn):
        in_tensors = []
        for invar in eqn.invars:
            #TODO weight refactor
            if self.is_weight(invar):
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

        out_tensor = add_empty_tensor(self.builder, "add_outvar", eqn.outvars[0].aval.shape, self.buffers)
        self.record_activation(eqn.outvars[0], out_tensor)

        op, opcode = add_add_layer(self.builder,in_tensors[0], in_tensors[1],out_tensor, all_tensors=self.tensors) 
        self.record_op(op, opcode)

    def dot_general_handler(self, eqn):
        #the weight buffers have already been created
        weight_buffer = self.weight_buffer_map[str(eqn.invars[1])]
        weight_shape = eqn.invars[1].aval.shape
        weight_tensor = add_tensor_with_buffer(self.builder, "weight", weight_shape, weight_buffer, self.buffers)

        if not self.is_recorded_activation(eqn.invars[0]):
            in_tensor = add_empty_tensor(self.builder, "in_tensor", eqn.invars[0].aval.shape, self.buffers)
        else:
            in_tensor = self.get_activation_tensor(eqn.invars[0])
        out_tensor = add_empty_tensor(self.builder, "out_tensor", eqn.outvars[0].aval.shape, self.buffers)

        self.record_activation(eqn.invars[0],in_tensor )
        self.record_weight(eqn.invars[1], weight_tensor)

        self.record_activation(eqn.outvars[0],out_tensor)

        op, opcode = add_fc_layer(self.builder,input_tensor=in_tensor, weight_tensor=weight_tensor,bias_tensor=None,output_tensor=out_tensor, all_tensors=self.tensors) 
        self.record_op(op, opcode)

        #TODO - need to map the from the model invars and outvars to the subgraph input and output
        #we can do this by mapping the created tensors to invars and outvars
        #but the invars are also weights, so we need to be smart about the mapping there
        #(e.g), anything that is not a weight is an invar 
        #need an activation map for tensors
    



