# Reference
# https://github.com/fuankarion/active-speakers-context/blob/master/core/models.py
# https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/temporal_shift.py

import torch
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment, n_div=8):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class TwoStreamResNet(nn.Module):
    def __init__(self, block, layers, rgb_stack_size, num_classes=2,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super(TwoStreamResNet, self).__init__()
        self.rgb_stack_size = rgb_stack_size
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # audio stream
        self.inplanes = 64
        self.audio_conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)
        self.a_bn1 = norm_layer(self.inplanes)
        self.a_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.a_layer1 = self._make_layer(block, 64, layers[0])
        self.a_layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.a_layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.a_layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # visual stream
        self.inplanes = 64
        self.video_conv1 = nn.Conv2d(3, self.inplanes,
                                     kernel_size=7, stride=2, padding=3,
                                     bias=False)
        self.v_bn1 = norm_layer(self.inplanes)
        self.v_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.v_layer1 = self._make_layer(block, 64, layers[0])
        self.v_layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.v_layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.v_layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_128_a = nn.Linear(512 * block.expansion, 128)
        self.fc_128_v = nn.Linear(512 * block.expansion, 128)

        # prediction heads
        self.fc_final = nn.Linear(128*2, num_classes)
        self.fc_aux_a = nn.Linear(128, num_classes)
        self.fc_aux_v = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # this improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, a, v, aa=[]):
        # audio Stream
        a = self.audio_conv1(a)
        a = self.a_bn1(a)
        a = self.relu(a)
        a = self.maxpool(a)

        a = self.a_layer1(a)
        a = self.a_layer2(a)
        a = self.a_layer3(a)
        a = self.a_layer4(a)
        a = self.avgpool(a)

        # visual Stream
        if len(v.shape) == 5:
            v = v.view(-1, v.shape[2], v.shape[3], v.shape[4])
        v = self.video_conv1(v)
        v = self.v_bn1(v)
        v = self.relu(v)
        v = self.maxpool(v)

        v = self.v_layer1(v)
        v = self.v_layer2(v)
        v = self.v_layer3(v)
        v = self.v_layer4(v)
        v = self.avgpool(v)

        # concat stream feats
        a = a.reshape(a.size(0), -1)
        v = v.reshape(v.size(0), -1)
        if a.shape[0] != v.shape[0]:
            v = v.view(-1, self.rgb_stack_size, a.shape[1])
            v = v.mean(1)
        stream_feats = torch.cat((a, v), 1)

        # auxiliary supervisions
        a = self.fc_128_a(a)
        a = self.relu(a)
        v = self.fc_128_v(v)
        v = self.relu(v)

        aux_a = self.fc_aux_a(a)
        aux_v = self.fc_aux_v(v)

        # global supervision
        av = torch.cat((a, v), 1)

        x = self.fc_final(av)

        return x, aux_a, aux_v, stream_feats


def make_temporal_shift(net, n_segment, n_div=8):
    n_round = 1
    if len(list(net.v_layer3.children())) >= 23:
        n_round = 2
        print('=> Using n_round {} to insert temporal shift'.format(n_round))

    def make_block_temporal(stage, this_segment):
        blocks = list(stage.children())
        for i, b in enumerate(blocks):
            if i % n_round == 0:
                blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div)
        return nn.Sequential(*blocks)

    net.v_layer1 = make_block_temporal(net.v_layer1, n_segment)
    net.v_layer2 = make_block_temporal(net.v_layer2, n_segment)
    net.v_layer3 = make_block_temporal(net.v_layer3, n_segment)
    net.v_layer4 = make_block_temporal(net.v_layer4, n_segment)

def _load_weights_into_two_stream_resnet(model, rgb_stack_size, arch, progress):
    resnet_state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

    own_state = model.state_dict()
    for name, param in resnet_state_dict.items():
        if 'v_'+name in own_state:
            own_state['v_'+name].copy_(param)
        if 'a_'+name in own_state:
            own_state['a_'+name].copy_(param)
        if 'v_'+name not in own_state and 'a_'+name not in own_state:
            print('No assignation for ', name)

    conv1_weights = resnet_state_dict['conv1.weight']
    own_state['video_conv1.weight'].copy_(conv1_weights)

    avgWs = torch.mean(conv1_weights, dim=1, keepdim=True)
    own_state['audio_conv1.weight'].copy_(avgWs)

    make_temporal_shift(model, rgb_stack_size)

    if arch == 'resnet50':
        own_state = model.state_dict()

        # please download the TSM weights from the official TSM repo: https://github.com/mit-han-lab/temporal-shift-module
        resnet_state_dict = torch.load('TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment16_e45.pth', map_location=torch.device('cpu'))['state_dict']
        for name, param in resnet_state_dict.items():
            name = name.replace('module.base_model.', '')
            if 'v_'+name in own_state:
                own_state['v_'+name].copy_(param)
            else:
                print('No assignation for ', name)


        conv1_weights = resnet_state_dict['module.base_model.conv1.weight']
        own_state['video_conv1.weight'].copy_(conv1_weights)

    print('loaded ws from resnet')
    return model


def _two_stream_resnet(arch, block, layers, pretrained, progress, rgb_stack_size,
                       num_classes, **kwargs):
    model = TwoStreamResNet(block, layers, rgb_stack_size, num_classes, **kwargs)
    if pretrained:
        model = _load_weights_into_two_stream_resnet(model, rgb_stack_size, arch, progress)
    else:
        make_temporal_shift(model, rgb_stack_size)
    return model


def manual_load_state_dict(model, weight_state_dict):
    own_state = model.state_dict()
    for name, param in weight_state_dict.items():
        if 'module.' in name:
            name = name.replace('module.', '')
        own_state[name].copy_(param)

    return model


def resnet18_two_streams(pretrained=False, progress=True, rgb_stack_size=11,
                         num_classes=2, **kwargs):
    return _two_stream_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                               rgb_stack_size, num_classes, **kwargs)

def resnet18_two_streams_forward(pretrained_weights_path, progress=True, rgb_stack_size=11,
                                 num_classes=2, **kwargs):
    model = _two_stream_resnet('resnet18', BasicBlock, [2, 2, 2, 2], False, progress,
                               rgb_stack_size, num_classes, **kwargs)
    model = manual_load_state_dict(model, torch.load(pretrained_weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def resnet50_two_streams(pretrained=False, progress=True, rgb_stack_size=11,
                         num_classes=2, **kwargs):
    return _two_stream_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                               rgb_stack_size, num_classes, **kwargs)

def resnet50_two_streams_forward(pretrained_weights_path, progress=True, rgb_stack_size=11,
                                 num_classes=2, **kwargs):
    model = _two_stream_resnet('resnet50', Bottleneck, [3, 4, 6, 3], False, progress,
                               rgb_stack_size, num_classes, **kwargs)
    model = manual_load_state_dict(model, torch.load(pretrained_weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model
