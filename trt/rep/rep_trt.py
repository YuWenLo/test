import torch
from hardnet import HarDNet
import time
import onnxruntime
import onnx
import tensorrt as trt
# 注册自定义插件
# 定义TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 注册自定义插件
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
from rep import create_RepVGG_tiny

# model = HarDNet(False, 68, pretrained=True).cuda()

model = create_RepVGG_tiny(deploy=False).cuda()
model.eval()
torch.save(model.state_dict(),  'rep_weight/rep_tiny.pth')

# print(model(torch.randn(1, 3, 48, 60).cuda()).shape) #torch.Size([1, 2])

batch = 6
input_tensor = torch.randn(batch, 3, 48, 60).cuda()
time0 = time.time()
out = model(input_tensor)
time1 = time.time()

'''-------------------onnx----------------------------'''
# # 將模型的權重類型轉換為 float32
# model = model.float()
# input_names = ["input"]
# output_names = ["output"]
# dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

# # 將模型轉換為 ONNX 格式
# torch.onnx.export(model, input_tensor, "rep_weight/rep_tiny.onnx", input_names=input_names, output_names=output_names, opset_version=13, verbose=False, dynamic_axes=dynamic_axes)

# # # 讀取 ONNX 模型
# onnx_model = onnx.load("rep_weight/rep_tiny.onnx")

# # 使用 ONNX runtime 執行推理
# ort_session = onnxruntime.InferenceSession("rep_weight/rep_tiny.onnx")
# input_data = {"input": input_tensor.cpu().numpy()}
# time2 = time.time()
# out_onnx = ort_session.run(output_names, input_data)
# time3 = time.time()

# output_mse = torch.mean((out.cpu() - torch.from_numpy(out_onnx[0])) ** 2)
# print("out Mean Squared Error:", output_mse.item())
# print("ori time = ", time1-time0)
# print("onnx time = ", time3-time2)

'''-------------------trt----------------------------'''
with open('rep_weight/rep_tiny.trt', 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

    # 创建 TensorRT 推理上下文
    context = engine.create_execution_context()

# 分配输入和输出内存
output_shape = (batch, 2)
output_buf = torch.empty(output_shape).cuda()  # 使用 PyTorch 创建 GPU 张量
bindings = [int(input_tensor.data_ptr()), int(output_buf.data_ptr())]

# 执行推理
time4 = time.time()
# context.execute(1, bindings)
context.execute_v2(bindings)
time5 = time.time() 
out_trt = output_buf

output_mse = torch.mean((out - out_trt) ** 2)
print("out Mean Squared Error:", output_mse.item())
print("ori time = ", time1-time0)
print("trt time = ", time5-time4)