import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
onnx_file = 'rep_weight/rep_tiny.onnx'
trt_file = 'rep_weight/rep_tiny.trt'
batch_size = 12


"""Takes an ONNX file and creates a TensorRT engine to run inference with"""
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 设置最大工作空间大小为 256 MiB
    # builder.max_workspace_size = 1 << 30 # 256MiB
    builder.max_batch_size = batch_size

    # 创建优化配置文件
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 48, 60), (6, 3, 48, 60), (12, 3, 48, 60))  # 设置输入形状

    # 为引擎设置优化配置文件
    config.add_optimization_profile(profile)

    # 构建TensorRT引擎
    engine = builder.build_engine(network, config)
    # config.flags |= 1 << int(trt.BuilderFlag.FP16)
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

