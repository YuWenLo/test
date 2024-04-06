from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
  from external.nms import soft_nms
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector
import time

class CtdetDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetDetector, self).__init__(opt)
  
  def process(self, images, return_time=False):
    print("-@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-")
    with torch.no_grad():
      images = torch.randn(12, 3, 432, 768).cuda()
      batch, ch, w, h = images.shape
      time0 = time.time()
      output = self.model(images)[-1]
      print("ori time = ", time.time()-time0)
      # print("output hm = ", output['hm'].shape, ", output wh = ", output['wh'].shape) #torch.Size([12, 1, 108, 192]) ,  torch.Size([12, 4, 108, 192])

      hm = output['hm'].sigmoid_() #<class 'torch.Tensor'>
      wh = output['wh']
      reg = wh[:,2:,:,:]
      wh  = wh[:,:2,:,:]

      ''' --------------------------------------'''
      time1 = time.time()
      self.context.active_optimization_profile = 0
      self.context.set_binding_shape(0, images.shape)
      output_shape = {'hm': (batch, 1, 108, 192), 'wh': (batch, 4, 108, 192)}
      output_buf = {name: torch.empty(shape).cuda() for name, shape in output_shape.items()}
      bindings = [int(images.data_ptr())] + [int(buf.data_ptr()) for buf in output_buf.values()]
      # self.context.execute(batch, bindings)
      self.context.execute_v2(bindings)
      print("trt time = ", time.time()-time1)

      print("output_trt hm = ", output_buf['hm'].shape, ", output_trt wh = ", output_buf['wh'].shape)
      hm_trt = output_buf['hm'].sigmoid_()
      wh_trt = output_buf['wh']
      reg_trt = wh_trt[:,2:,:,:]
      wh_trt  = wh_trt[:,:2,:,:]

      # hm = output_buf['hm'].sigmoid_()
      # wh = output_buf['wh']
      # reg = wh[:,2:,:,:]
      # wh  = wh[:,:2,:,:]

      # # 计算hm和wh之间的绝对值差的和
      for idx in range(batch):
        print(f"now {idx} round ---------------------")
        hm_mse = torch.mean((hm[idx] - hm_trt[idx]) ** 2)
        print(hm[idx])
        print(hm_trt[idx])
        print("hm Mean Squared Error:", hm_mse.item())
        wh_mse = torch.mean((wh[idx] - wh_trt[idx]) ** 2)
        print("wh Mean Squared Error:", wh_mse.item())
        reg_mse = torch.mean((reg[idx] - reg_trt[idx]) ** 2)
        print("reg Mean Squared Error:", reg_mse.item())

      # '''
      # hm Mean Squared Error: 3.9516062644295857e-10
      # wh Mean Squared Error: 3.5924216934546394e-09
      # reg Mean Squared Error: 3.117188063228582e-09
      # '''

      # ''' --------------------------------------'''
      if self.opt.flip_test:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        reg_flip = flip_tensor(reg[1:2])
        reg = reg[0:1]
        reg[:,0,:,:] = (reg[:,0,:,:] + 1- reg_flip[:,0,:,:])/2
        reg[:,1,:,:] = (reg[:,1,:,:] +    reg_flip[:,1,:,:])/2
              
      torch.cuda.synchronize()
      forward_time = time.time()
      dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
      
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))

  def show_results(self, debugger, image, results, name):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    debugger.show_all_imgs(pause=self.pause, name=name)
