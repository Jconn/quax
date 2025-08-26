import flatbuffers
import numpy as np
from tensorflow.lite.tools import flatbuffer_utils
# Assuming the flatbuffer schema files have been generated and are available in your path
from quax.schema_py_generated import (Model, ModelT, SubGraph, SubGraphT, Tensor, TensorT, OperatorCode, OperatorCodeT,
                                        Buffer, BufferT, Operator, OperatorT, BuiltinOperator, 
                                        BuiltinOptions, FullyConnectedOptions, FullyConnectedOptionsT,
                                        ActivationFunctionType)
import argparse

#model = flatbuffer_utils.read_model("/home/johnconn/fc_in_128_128_relu.tflite")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('model', type=str, default='checkpoints', help='checkpoint directory')
    args = parser.parse_args()
    model = flatbuffer_utils.read_model(args.model)
    #model = flatbuffer_utils.read_model("/home/johnconn/fc_32_64_f32.tflite")
    #model = flatbuffer_utils.read_model("/home/johnconn/f32_no_quant.tflite")
    print(f"version: {model.version}")
    print(f"opcodes: {len(model.operatorCodes)}")
    print(f"subgraphs: {len(model.subgraphs)}")
    print(f"description: {model.description}")
    print(f"buffers: {len(model.buffers)}")
    print("metadata:")
    if model.metadata:
        for md in model.metadata:
            print(f"{md.name}: buffer {md.buffer}")
    print("signature defs:")
    if model.signatureDefs:
        for sd in model.signatureDefs:
            print("inputs: ")
            if sd.inputs:
                for tens_in in sd.inputs:
                    print(f"{tens_in.name}: {tens_in.tensorIndex}")
            print("outputs: ")
            if sd.outputs:
                for tens_out in sd.outputs:
                    print(f"{tens_out.name}: {tens_out.tensorIndex}")
            print("signature key:")
            print(sd.signatureKey)
            print("subgraph Idx:")
            print(sd.subgraphIndex)

    import pdb; pdb.set_trace()
