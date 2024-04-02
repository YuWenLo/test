import torch
from hardnet import HarDNet
import time
import onnxruntime
import onnx
import tensorrt as trt
from onnxruntime.quantization import quantize

model = HarDNet(False, 68, pretrained=True).cuda()
# model = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet68', pretrained=True)
model.eval()

# 打印模型的输出节点
print(model(torch.randn(1, 3, 120, 180).cuda()).shape) #torch.Size([1, 1000])

input_tensor = torch.randn(1, 3, 120, 180).cuda()
time0 = time.time()
out = model(input_tensor)
time1 = time.time()

''''''
# # ---------------------------------------------------------------------------
# 导出模型到 ONNX 格式
input_names = ["input"]
output_names = ["output"]
# torch.onnx.export(model, input_tensor, "hardnet.onnx", input_names=input_names, output_names=output_names)

# 讀取 ONNX 模型# 讀取 ONNX 模型
onnx_model = onnx.load("hardnet.onnx")
# # 量化 ONNX 模型
# quantized_model = quantize(onnx_model)
# onnx.save(quantized_model, "quantized_model.onnx")
# 使用 ONNX runtime 執行推理
ort_session = onnxruntime.InferenceSession("hardnet.onnx")
input_data = {"input": input_tensor.cpu().numpy()}
time2 = time.time()
out_trt = ort_session.run(output_names, input_data)
time3 = time.time()

# 檢查輸出是否與 PyTorch 一致
# 计算绝对值差
# print(out.size())
# print(torch.from_numpy(out_trt[0]).cuda().size())
abs_diff = torch.abs(out - torch.from_numpy(out_trt[0]).cuda())
# 计算绝对值差的和
sum_abs_diff = torch.sum(abs_diff)

print("ori跟ONNX绝对值差的和:", sum_abs_diff.item())
''''''
# ---------------------------------------------------------------------------
# 加载 TensorRT 引擎文件
with open('hardnet.trt', 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# 创建 TensorRT 推理上下文
context = engine.create_execution_context()

# 分配输入和输出内存
input_shape = (1, 3, 120, 180)  # 根据您的模型输入大小设置
output_shape = (1, 1000) 
# input_buf = torch.randn(*input_shape).cuda()  # 使用 PyTorch 创建 GPU 张量
output_buf = torch.empty(output_shape).cuda()  # 使用 PyTorch 创建 GPU 张量
bindings = [int(input_tensor.data_ptr()), int(output_buf.data_ptr())]

# 执行推理
time4 = time.time()
context.execute(1, bindings)
time5 = time.time()

# 获取推理结果
# print(output_buf.cpu().detach().numpy())  # 将 GPU 张量转换为 numpy 数组并打印
# print(out)
# print(output_buf)
abs_diff = torch.abs(out - output_buf)
# 计算绝对值差的和
sum_abs_diff = torch.sum(abs_diff)

print("ori跟TRT绝对值差的和:", sum_abs_diff.item())
print(f"原始模型執行時間 = {time1 - time0}, 轉換為 ONNX 後的模型執行時間 = {time3 - time2}, 轉換為 TRT 後的模型執行時間 = {time5 - time4}")


# ---------------------------------------------------------------------------------------------------------------------------
# ori跟ONNX绝对值差的和: 0.001177128404378891
# ori跟TRT绝对值差的和: 3.3554210662841797
# 原始模型執行時間 = 0.005881071090698242, 轉換為 ONNX 後的模型執行時間 = 0.02107834815979004, 轉換為 TRT 後的模型執行時間 = 0.0020742416381835938