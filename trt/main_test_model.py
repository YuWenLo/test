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

# from torch2trt import torch2trt
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
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)

  # print(model)
  return model

  # print(model)
  #optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  optimizer = torch.optim.SGD(model.parameters(), opt.lr, weight_decay = 1e-4, momentum = 0.9)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10
  train_losses = []
  train_hm_losses = []
  train_wh_losses = []
  train_off_losses = []
  val_losses = []
  val_hm_losses = []
  val_wh_losses = []
  val_off_losses = []
  lr_record = []
  val_loss_tmp, val_hmloss_tmp, val_whloss_tmp, val_offloss_tmp = 10,10,10,10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    if epoch <= 2 and opt.wlr > 0:
      cur_lr = opt.lr if epoch == 2 else opt.wlr
      for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
      if k == 'loss': train_losses.append(v)
      elif k == 'hm_loss': train_hm_losses.append(v)
      elif k == 'wh_loss': train_wh_losses.append(v)
      elif k == 'off_loss': train_off_losses.append(v)
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
        if k == 'loss': val_losses.append(v)
        elif k == 'hm_loss': val_hm_losses.append(v)
        elif k == 'wh_loss': val_wh_losses.append(v)
        elif k == 'off_loss': val_off_losses.append(v)
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
      if epoch < 5:
        val_losses.append(val_loss_tmp)
        val_hm_losses.append(val_hmloss_tmp)
        val_wh_losses.append(val_whloss_tmp)
        val_off_losses.append(val_offloss_tmp)
      else:
        val_losses.append(val_losses[-1])
        val_hm_losses.append(val_hm_losses[-1])
        val_wh_losses.append(val_wh_losses[-1])
        val_off_losses.append(val_off_losses[-1])
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      lr_record.append(lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()
  
  print(lr_record)
  #plot
  fig, ax = plt.subplots()
  x = np.arange(0, opt.num_epochs, 1)
  line1, = ax.plot(x, train_losses, label='train Loss', color='r')
  line2, = ax.plot(x, train_hm_losses, label='train hm Loss', color='g')
  line3, = ax.plot(x, train_wh_losses, label='train wh Loss', color='b')
  line4, = ax.plot(x, train_off_losses, label='train off Loss', color='yellow')
  line5, = ax.plot(x, val_losses, label='val Loss', color='r', linestyle='--')
  line6, = ax.plot(x, val_hm_losses, label='val hm Loss', color='g', linestyle='--')
  line7, = ax.plot(x, val_wh_losses, label='val wh Loss', color='b', linestyle='--')
  line8, = ax.plot(x, val_off_losses, label='val off Loss', color='yellow', linestyle='--')

  ax.set_xlabel('Epoch')
  # ax.set_ylabel('Loss', color='g')
  # ax2.set_ylabel('Acc', color='b')
  ax.set_ylim([0, 10])
  ax.legend(loc='upper left')
  # ax2.legend(loc='upper right')
  pic_save_path = opt.save_dir + "/train_plot.jpg"
  plt.savefig(pic_save_path, dpi=300)

if __name__ == '__main__':
    opt = opts().parse()
    model = main(opt)
    
    model = model.cuda()
    model.eval()
    input_tensor = torch.randn(1, 3, 120, 180).cuda()
    
    time0 = time.time()
    out = model(input_tensor)
    time1 = time.time()

    # 打印模型的输出节点
    # for item in out:
    #   for key, value in item.items():
    #       print(key, value.shape)
    
    # print(model(input_tensor)[0].shape) #torch.Size([1, 1000]) 
    
    # ---------------------------------------------------------------------------------------
    
    # 將模型的權重類型轉換為 float32
    model = model.float()
    input_names = ["input"]
    output_names = ["hm", "wh"]

    # 將模型轉換為 ONNX 格式
    # torch.onnx.export(model, input_tensor, "model.onnx", input_names=input_names, output_names=output_names, opset_version=11, verbose=False)

    # 讀取 ONNX 模型
    onnx_model = onnx.load("model.onnx")

    # 使用 ONNX runtime 執行推理
    ort_session = onnxruntime.InferenceSession("model.onnx")
    input_data = {"input": input_tensor.cpu().numpy()}
    time2 = time.time()
    out_trt = ort_session.run(output_names, input_data)
    time3 = time.time()

    print(out_trt)

    # 檢查輸出是否與 PyTorch 一致
    out_cpu = [{'hm': item['hm'].cpu() for item in out}]
    out_trt_cpu = [torch.from_numpy(item[0]).cpu() for item in out_trt]
    abs_diff = [torch.abs(item1['hm'] - item2) for item1, item2 in zip(out_cpu, out_trt_cpu)]
    print(abs_diff)
    sum_abs_diff = sum(torch.sum(diff) for diff in abs_diff)
    print("hm絕對值差的和:", sum_abs_diff.item())
    print(f"原始模型執行時間 = {time1 - time0}, 轉換為 ONNX 後的模型執行時間 = {time3 - time2}")

    '''
    hm絕對值差的和: 9.5367431640625e-07
    原始模型執行時間 = 0.18599867820739746, 轉換為 ONNX 後的模型執行時間 = 0.01972651481628418
    '''

    # ------------------------------------------------------------------------------------------------------------
    # # 加载 TensorRT 引擎文件
    # with open('model.trt', 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
    #     engine = runtime.deserialize_cuda_engine(f.read())

    # # 创建 TensorRT 推理上下文
    # context = engine.create_execution_context()

    # # 分配输入和输出内存
    # input_shape = (1, 3, 120, 180)  # 根据您的模型输入大小设置
    # output_shape = {'hm': (1, 1, 30, 45), 'wh': (1, 4, 30, 45)}
    # input_buf = torch.randn(*input_shape).cuda()  # 使用 PyTorch 创建 GPU 张量
    # output_buf = {name: torch.empty(shape).cuda() for name, shape in output_shape.items()}
    # # bindings = [int(buf.data_ptr()) for buf in output_buf.values()]
    # bindings = [int(input_tensor.data_ptr())] + [int(buf.data_ptr()) for buf in output_buf.values()]

    # # 执行推理
    # time4 = time.time()
    # context.execute(1, bindings)
    # time5 = time.time()

    # # 获取推理结果
    # output_data = {name: tensor.cpu().detach().numpy() for name, tensor in output_buf.items()}

    # # 示例：打印输出
    # # print(output_data)

    # # 檢查輸出是否與 PyTorch 一致
    # # 从字典中获取hm和wh的值
    # out_hm = out[0]['hm']
    # out_wh = out[0]['wh']

    # output_data_hm = output_data['hm']
    # output_data_wh = output_data['wh']

    # # 将CUDA张量移动到CPU上并转换为NumPy数组
    # out_hm = out_hm.cpu().detach().numpy()
    # out_wh = out_wh.cpu().detach().numpy()

    # # 计算hm和wh之间的绝对值差的和
    # hm_abs_diff_sum = np.sum(np.abs(out_hm - output_data_hm))
    # wh_abs_diff_sum = np.sum(np.abs(out_wh - output_data_wh))

    # print("hm 绝对值差的和:", hm_abs_diff_sum)
    # print("wh 绝对值差的和:", wh_abs_diff_sum)
    # print(f"原始模型執行時間 = {time1 - time0}, 轉換為 trt 後的模型執行時間 = {time5 - time4}")

    '''
    hm 绝对值差的和: 1.1598647
    wh 绝对值差的和: 0.13366115
    原始模型執行時間 = 0.18133020401000977, 轉換為 trt 後的模型執行時間 = 0.0014691352844238281
    '''
