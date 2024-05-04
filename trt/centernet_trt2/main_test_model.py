from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
import numpy as np
import matplotlib.pyplot as plt

import time
import onnxruntime
import onnx
import tensorrt as trt


def derivative_mod(m, gain):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            print(key)

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  # print(opt)

  logger = Logger(opt)

  # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)

  # print(model)
  return model

if __name__ == '__main__':
    pth_flag = False
    onnx_flag = False
    trt_flag = True
    opt = opts().parse()
    model = main(opt)
    
    model = model.cuda()
    model.eval()

    if pth_flag:
      # state_dict = model.state_dict()
      # data = {'epoch': 0, 'state_dict': state_dict}
      # torch.save(data,  'centernet/hardnet48.pth')
      torch.save(model.state_dict(),  'centernet/hardnet48.pth')

    model.load_state_dict(torch.load('centernet/hardnet48.pth')) # TODO 1_54.pth
    batch = 12 # TODO 1for onnx, 1~12 for inference
    input_tensor = torch.randn(batch, 3, 432, 768).cuda()
    
    time0 = time.time()
    out = model(input_tensor)
    time1 = time.time()
    
    # ---------------------------------------------------------------------------------------
    
    # 將模型的權重類型轉換為 float32
    model = model.float()
    input_names = ["input"]
    output_names = ["hm", "wh"]
    dynamic_axes = {"input": {0: "batch"}, "hm": {0: "batch"}, "wh": {0: "batch"}}

    # 將模型轉換為 ONNX 格式
    if onnx_flag: # TODO path to save onnx 
      torch.onnx.export(model, input_tensor, "centernet/hardnet48_dynamic.onnx", export_params=True, do_constant_folding=True,\
                        input_names=input_names, output_names=output_names, opset_version=13, verbose=False, dynamic_axes=dynamic_axes)

    # 讀取 ONNX 模型
    # onnx_model = onnx.load("centernet/hardnet48_dynamic.onnx")

    # 使用 ONNX runtime 執行推理
    ort_session = onnxruntime.InferenceSession("centernet/hardnet48_dynamic.onnx") # TODO path to onnx 
    input_data = {"input": input_tensor.cpu().numpy()}
    time2 = time.time()
    out_onnx = ort_session.run(output_names, input_data)
    time3 = time.time()

    # print(out_onnx)
    print(type(out[0]['hm']), ", ", out[0]['hm'].size())
    print(torch.from_numpy(out_onnx[0]).size())

    hm_mse = torch.mean((out[0]['hm'].cpu() - torch.from_numpy(out_onnx[0])) ** 2)
    print("hm Mean Squared Error:", hm_mse.item())
    wh_mse = torch.mean((out[0]['wh'].cpu() - torch.from_numpy(out_onnx[1])) ** 2)
    print("wh Mean Squared Error:", wh_mse.item())
    print(f"原始模型執行時間 = {time1 - time0}, 轉換為 ONNX 後的模型執行時間 = {time3 - time2}")

    '''
    batch = 1
    hm Mean Squared Error: 7.94974500839398e-17
    wh Mean Squared Error: 1.18072820460979e-19
    原始模型執行時間 = 1.5978989601135254, 轉換為 ONNX 後的模型執行時間 = 0.25577259063720703

    batch = 12
    hm Mean Squared Error: 8.315250313046052e-17
    wh Mean Squared Error: 1.7047074438421247e-19
    原始模型執行時間 = 1.7412919998168945, 轉換為 ONNX 後的模型執行時間 = 3.8112165927886963
    '''
    # ------------------------------------------------------------------------------------------------------------
    if trt_flag:
      # 加载 TensorRT 引擎文件
      with open('centernet/hardnet48_dynamic_1_12_12.trt', 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:# TODO path to trt
          engine = runtime.deserialize_cuda_engine(f.read())

      # 创建 TensorRT 推理上下文
      context = engine.create_execution_context()

      # 分配输入和输出内存
      output_shape = {'hm': (batch, 1, 108, 192), 'wh': (batch, 4, 108, 192)}
      
      print(input_tensor.size())
      print(output_shape)
      output_buf = {name: torch.empty(shape).cuda() for name, shape in output_shape.items()}
      bindings = [int(input_tensor.data_ptr())] + [int(buf.data_ptr()) for buf in output_buf.values()]

      # 执行推理
      time4 = time.time()
      print('shape = ', context.get_binding_shape(0))
      context.active_optimization_profile = 0
      context.set_binding_shape(0, input_tensor.cpu().numpy().shape)
      context.execute_v2(bindings)
      time5 = time.time()

      # 获取推理结果
      output_data = {name: tensor.cpu().detach().numpy() for name, tensor in output_buf.items()}

      # 示例：打印输出
      # print(output_data)

      # 檢查輸出是否與 PyTorch 一致
      # 从字典中获取hm和wh的值
      out_hm = out[0]['hm']
      out_wh = out[0]['wh']

      output_data_hm = output_data['hm']
      output_data_wh = output_data['wh']

      # 将CUDA张量移动到CPU上并转换为NumPy数组
      out_hm = out_hm.cpu().detach().numpy()
      out_wh = out_wh.cpu().detach().numpy()

      # 计算hm和wh之间的绝对值差的和
      hm_abs_diff_sum = np.sum(np.abs(out_hm - output_data_hm))
      wh_abs_diff_sum = np.sum(np.abs(out_wh - output_data_wh))

      print("hm 绝对值差的和:", hm_abs_diff_sum)
      print("wh 绝对值差的和:", wh_abs_diff_sum)
      print(f"原始模型執行時間 = {time1 - time0}, 轉換為 trt 後的模型執行時間 = {time5 - time4}")

      '''
      hm 绝对值差的和: 1.1598647
      wh 绝对值差的和: 0.13366115
      原始模型執行時間 = 0.18133020401000977, 轉換為 trt 後的模型執行時間 = 0.0014691352844238281
      '''
