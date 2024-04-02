import os
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
onnx_file = 'hardnet.onnx'
trt_file = 'hardnet.trt'
batch_size = 1


"""Takes an ONNX file and creates a TensorRT engine to run inference with"""
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 设置最大工作空间大小为 256 MiB
    # builder.max_workspace_size = 1 << 30 # 256MiB
    builder.max_batch_size = batch_size
    config.flags |= 1 << int(trt.BuilderFlag.FP16)
    # Parse model file
    with open(onnx_file, 'rb') as model:
        print('Beginning ONNX file parsing')
        if not parser.parse(model.read()):
            print ('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print (parser.get_error(error))
    print('Completed parsing of ONNX file')
    engine = builder.build_engine(network, config)
    print("Completed creating Engine")
    with open(trt_file, "wb") as f:
        f.write(engine.serialize())


# 直接轉INT  ?
# with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
#     config = builder.create_builder_config()
#     config.max_workspace_size = 1 << 30  # 设置最大工作空间大小为 256 MiB
#     config.flags |= 1 << int(trt.BuilderFlag.FP16)  # 使用 FP16 模式加速（如果支持）
#     config.flags |= 1 << int(trt.BuilderFlag.INT8)  # 设置数据类型为 INT8
#     builder.max_batch_size = batch_size

#     # Parse model file
#     with open(onnx_file, 'rb') as model:
#         print('Beginning ONNX file parsing')
#         if not parser.parse(model.read()):
#             print('ERROR: Failed to parse the ONNX file.')
#             for error in range(parser.num_errors):
#                 print(parser.get_error(error))
#             exit(1)
#     print('Completed parsing of ONNX file')

#     engine = builder.build_engine(network, config)
#     print("Completed creating Engine")

#     with open(trt_file, "wb") as f:
#         f.write(engine.serialize())