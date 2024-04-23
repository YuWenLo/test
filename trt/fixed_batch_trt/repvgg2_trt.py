import torch
import time
import onnxruntime
import onnx
import tensorrt as trt
import os
# 注册自定义插件
# 定义TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# 注册自定义插件
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
# from rep import create_RepVGG_tiny
from repvgg2 import create_RepVGG_tiny

model = create_RepVGG_tiny(deploy=False).cuda()
model.eval()
# os.makedirs('repvgg2_weight_fixed', exist_ok=True)
# torch.save(model.state_dict(),  'repvgg2_weight_fixed/repvgg2.pth')
model.load_state_dict(torch.load('repvgg2_weight_fixed/repvgg2.pth'))

torch.manual_seed(1)
batch = 6
input_tensor = torch.randn(batch, 3, 48, 60).cuda()
time0 = time.time()
out = model(input_tensor)
time1 = time.time()

'''-------------------onnx----------------------------'''
if True:
# if False:
    # 將模型的權重類型轉換為 float32
    model = model.float()
    input_names = ["input"]
    output_names = ["output"]

    # 將模型轉換為 ONNX 格式
    # torch.onnx.export(model, input_tensor, "repvgg2_weight_fixed/repvgg2.onnx", input_names=input_names, output_names=output_names, opset_version=13, verbose=False)

    # # 讀取 ONNX 模型
    onnx_model = onnx.load("repvgg2_weight_fixed/repvgg2.onnx")

    # 使用 ONNX runtime 執行推理
    ort_session = onnxruntime.InferenceSession("repvgg2_weight_fixed/repvgg2.onnx")
    input_data = {"input": input_tensor.cpu().numpy()}
    time2 = time.time()
    out_onnx = ort_session.run(output_names, input_data)
    time3 = time.time()

    output_mse = torch.mean((out.cpu() - torch.from_numpy(out_onnx[0])) ** 2, dim=1)
    print("out-onnx MSE:", output_mse) #out Mean Squared Error: 2.2005544969235825e-14
    print("ori time = ", time1-time0) #ori time =  0.005249738693237305
    print("onnx time = ", time3-time2) #onnx time =  0.011258125305175781
    print('----------------------------------------------------------')
'''-------------------trt----------------------------'''
if True:
# if False:
    with open('repvgg2_weight_fixed/repvgg2.trt', 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

        # 创建 TensorRT 推理上下文
        context = engine.create_execution_context()
        # context.set_binding_shape(0, (batch, 3, 48, 60))
        # context.set_binding_shape(1, (batch, 2))


    # 分配输入和输出内存
    output_shape = (batch, 2)
    output_buf = torch.empty(output_shape).cuda()  # 使用 PyTorch 创建 GPU 张量
    bindings = [int(input_tensor.data_ptr()), int(output_buf.data_ptr())]

    # 执行推理
    time4 = time.time()
    context.execute_v2(bindings)
    time5 = time.time() 
    out_trt = output_buf

    print("out-trt MSE:", torch.mean((out - out_trt) ** 2, dim=1))
    print("ori time = ", time1-time0)
    print("trt time = ", time5-time4)