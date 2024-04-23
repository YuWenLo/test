import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
onnx_file = 'repvgg2_weight_fixed/repvgg2.onnx'
trt_file = 'repvgg2_weight_fixed/repvgg2.trt'
batch_size = 6


"""Takes an ONNX file and creates a TensorRT engine to run inference with"""
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 设置最大工作空间大小为 1 MiB
    builder.max_batch_size = batch_size

    # 构建TensorRT引擎
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

