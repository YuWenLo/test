nvcc -V
Cuda compilation tools, release 12.3, V12.3.52

dpkg -l | grep cudnn
8.9.6.50-1+cuda12.2

安裝tensorrt
https://blog.csdn.net/qq_37541097/article/details/114847600
pip install nvidia-pyindex
pip install nvidia-tensorrt==8.2.5.1
# -------------------------------------------------------------------------------------------------
import tensorrt
print(tensorrt.__version__)

# -------------------------------------------------------------------------------------------------
import tensorrt as trt
import numpy as np

# 1. 加载已有的TensorRT模型
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_file_path = 'your_model.trt'

with open(trt_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# 2. 定义量化器配置
quantization_flags = 1 << int(trt.QuantizeFlag.INT8)

# 3. 创建量化器配置
config = trt.IInt8Calibrator()
config.set_quantize_flag(quantization_flags)

# 4. 应用量化器配置到引擎
with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = config

    # 将现有的TensorRT引擎复制到新的量化引擎中
    new_engine = builder.build_engine(network, config)
