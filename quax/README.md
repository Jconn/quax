goals:

arbitrary quantized training
tflite conversion


quantized training how to do - 
need to pass tensors through the model as objects - they must maintain their quant info
need to have some way to lower while maintaining tracking of tensor states
