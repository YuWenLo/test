import torch.nn as nn
import numpy as np
import torch
import copy
import torch.utils.checkpoint as checkpoint

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.InstanceNorm2d(num_features=out_channels, eps=1e-04))
    return result

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.InstanceNorm2d(num_features=in_channels, eps=1e-04) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.data.size(0),-1)

class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False, use_checkpoint=False):
        super(RepVGG, self).__init__()
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.outplanes = [128 * width_multiplier[1], 256 * width_multiplier[2], 512 * width_multiplier[3]]
        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=1, padding=1)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage2 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage3 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.classifier = nn.Sequential(
                            nn.AdaptiveAvgPool2d((1,1)),
                            Flatten(),
                            nn.Linear(int(512 * width_multiplier[3]), num_classes))

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)

    def forward(self, x):
        features = []
        out = self.stage0(x)
        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                out = block(out)
        out = self.classifier(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

def create_RepVGG_tiny(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[1, 2, 2, 4], num_classes=2,
                  width_multiplier=[0.5, 0.5, 0.5, 0.5])


func_dict = {
'RepVGG-tiny': create_RepVGG_tiny}
def get_RepVGG_func_by_name(name):
    return func_dict[name]

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model