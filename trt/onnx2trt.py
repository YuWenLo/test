import os
import tensorrt as trt
# from SimpleCalibrator import SimpleCalibrator # local module
# config.int8_calibrator = SimpleCalibrator()

# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
# onnx_file = 'hardnet39_dynamic.onnx'
# trt_file = 'hardnet39_dynamic.trt'
# batch_size = 12


# """Takes an ONNX file and creates a TensorRT engine to run inference with"""
# with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
#     config = builder.create_builder_config()
#     config.max_workspace_size = 1 << 30  # 设置最大工作空间大小为 256 MiB
#     # builder.max_workspace_size = 1 << 30 # 256MiB
#     builder.max_batch_size = batch_size
#     config.flags |= 1 << int(trt.BuilderFlag.FP16)
#     # Parse model file
#     with open(onnx_file, 'rb') as model:
#         print('Beginning ONNX file parsing')
#         if not parser.parse(model.read()):
#             print ('ERROR: Failed to parse the ONNX file.')
#             for error in range(parser.num_errors):
#                 print (parser.get_error(error))
#     print('Completed parsing of ONNX file')
#     engine = builder.build_engine(network, config)
#     print("Completed creating Engine")
#     with open(trt_file, "wb") as f:
#         f.write(engine.serialize())


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
onnx_file = 'hardnet39_dynamic.onnx'
trt_file = 'hardnet39_dynamic.trt'
batch_size = 12


"""Takes an ONNX file and creates a TensorRT engine to run inference with"""
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 设置最大工作空间大小为 256 MiB
    # builder.max_workspace_size = 1 << 30 # 256MiB
    builder.max_batch_size = batch_size

    # 创建优化配置文件
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 432, 768), (12, 3, 432, 768), (12, 3, 432, 768))  # 设置输入形状

    # 为引擎设置优化配置文件
    config.add_optimization_profile(profile)

    # 构建TensorRT引擎
    engine = builder.build_engine(network, config)
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

# @classmethod
# def build_engine(cls,
#                     onnx_file_path,
#                     engine_file_path,
#                     *,
#                     use_fp16=True,
#                     dynamic_shapes={},
#                     dynamic_batch_size=1):
#     """Build TensorRT Engine
#     :use_fp16: set mixed flop computation if the platform has fp16.
#     :dynamic_shapes: {binding_name: (min, opt, max)}, default {} represents not using dynamic.
#     :dynamic_batch_size: set it to 1 if use fixed batch size, else using max batch size
#     """
#     builder = trt.Builder(cls.trt_logger)
#     network = builder.create_network(
#         1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#     config = builder.create_builder_config()
#     config.set_tactic_sources(trt.TacticSource.CUBLAS_LT)

#     # Default workspace is 2G
#     config.max_workspace_size = 2 << 30

#     if builder.platform_has_fast_fp16 and use_fp16:
#         config.set_flag(trt.BuilderFlag.FP16)

#     # parse ONNX
#     parser = trt.OnnxParser(network, cls.trt_logger)
#     with open(onnx_file_path, 'rb') as model:
#         if not parser.parse(model.read()):
#             print('ERROR: Failed to parse the ONNX file.')
#             for error in range(parser.num_errors):
#                 print(parser.get_error(error))
#             return None
#     print("===> Completed parsing ONNX file")

#     # default = 1 for fixed batch size
#     builder.max_batch_size = 1

#     if len(dynamic_shapes) > 0:
#         print(f"===> using dynamic shapes: {str(dynamic_shapes)}")
#         builder.max_batch_size = dynamic_batch_size
#         profile = builder.create_optimization_profile()

#         for binding_name, dynamic_shape in dynamic_shapes.items():
#             min_shape, opt_shape, max_shape = dynamic_shape
#             profile.set_shape(
#                 binding_name, min_shape, opt_shape, max_shape)

#         config.add_optimization_profile(profile)

#     # Remove existing engine file
#     if os.path.isfile(engine_file_path):
#         try:
#             os.remove(engine_file_path)
#         except Exception:
#             print(f"Cannot remove existing file: {engine_file_path}")

#     print("===> Creating Tensorrt Engine...")
#     engine = builder.build_engine(network, config)
#     if engine:
#         with open(engine_file_path, "wb") as f:
#             f.write(engine.serialize())
#         print("===> Serialized Engine Saved at: ", engine_file_path)
#     else:
#         print("===> build engine error")
#     return engine