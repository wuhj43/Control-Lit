import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
from torchvision.models import vgg as vgg
# from torchmetrics.regression import SpearmanCorrCoef

import os
l1=nn.L1Loss()
m=64
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-4

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class L1_Charbonnier_loss_weight(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss_weight, self).__init__()
        self.eps = 1e-4

    def forward(self, X, Y,ori):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)#*torch.abs(ori-Y)
        loss = torch.mean(error)
        return loss


class L1_Charbonnier_loss_surp_var(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss_surp_var, self).__init__()
        self.eps = 1e-4

    def forward(self, X, Y,var):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        var=torch.mean(var.view(X.shape[0],-1),dim=-1,keepdim=True)
        loss = torch.mean(torch.mean(error,dim=[1,2,3])/(2*var)+0.5*torch.log(var))
        return loss

class GradientLoss(nn.Module):
    def __init__(self, eps=1e-6,weight=.1):
        super(GradientLoss, self).__init__()
        # sobel operator
        g_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        g_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        # scharr operator for weak edges
        # g_x = torch.Tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
        # g_y = torch.Tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
        self.g_x = g_x.reshape([1, 1, 3, 3]).repeat(3,1,1,1)
        self.g_y = g_y.reshape([1, 1, 3, 3]).repeat(3,1,1,1)
        self.register_buffer('g_x_kernel', self.g_x)
        self.register_buffer('g_y_kernel', self.g_y)
        self.eps = eps
        self.weight=weight

    def compute_loss(self, prediction, gt):
        diff = prediction - gt
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

    def conv_gradient(self, img):
        n_channels, _, kw, kh = self.g_x_kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.g_x_kernel, groups=n_channels), F.conv2d(img, self.g_y_kernel, groups=n_channels)

    def construct_gt(self, source):
        s_gx, s_gy = self.conv_gradient(source)
        return 0.5*(s_gx * s_gx + s_gy * s_gy)

    def forward(self, prediction, gt):
        gt = self.construct_gt(gt)
        pr_gx, pr_gy = self.conv_gradient(prediction)
        pr = pr_gx * pr_gx + pr_gy * pr_gy
        return self.weight*self.compute_loss(pr, gt)


def position_sampling(k, m, n):
    pos_1 = torch.randint(k, size=(n, m, 2))
    pos_2 = torch.randint(k, size=(n, m, 2))
    return pos_1, pos_2
def collect_samples(x, pos, n):
    _, c, h, w = x.size()
    x = x.view(n, c, -1).permute(1, 0, 2).reshape(c, -1)
    pos = ((torch.arange(n).long().to(pos.device) * h * w).view(n, 1)
    + pos[:, :, 0] * h + pos[:, :, 1]).view(-1)
    return (x[:, pos]).view(c, n, -1).permute(1, 0, 2)

def dense_relative_localization_loss(x):
    n, D, k, k = x.size()
    pos_1, pos_2 = position_sampling(k, m, n)
    deltaxy = abs((pos_1 - pos_2).float()) # [n, m, 2]
    deltaxy /= k
    pts_1 = collect_samples(x, pos_1, n).transpose(1, 2) # [n, m, D]
    pts_2 = collect_samples(x, pos_2, n).transpose(1, 2) # [n, m, D]
    predxy = MLP(torch.cat([pts_1, pts_2], dim=2))
    return l1(predxy, deltaxy)


class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers = self.centers.cuda()

    def forward(self, x):
        x = torch.unsqueeze(x, 1) - torch.unsqueeze(torch.unsqueeze(self.centers, 1),0).repeat(x.shape[0],1,1)
        x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=1)
        # y = x.sum()
        # x = x / (x.sum() + 0.0001)
        return x

    def forward_1(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta
        x = x.sum(dim=-1)
        # x = x / (x.sum() + 0.00001)
        return x

def hist_loss(seg_pred, input_1, input_2, gpu_id=None):
    '''
    1. seg_pred transform to [1,2,3,2,3,1,3...] x batchsize
    2. Get class 1,2,3 index
    3. Use index to get value of img1 and img2
    4. Get hist of img1 and img2
    :return:
    '''
    region_num=25
    N, C, H, W = seg_pred.shape
    bit = 256
    seg_pred_cls = seg_pred.reshape(N, C, -1)
    # seg_pred_cls=seg_pred_cls[:,0].unsqueeze(1)
    # seg_pred_cls = seg_pred.argmax(dim=1)
    input_1 = input_1.reshape(N, 3, -1)
    input_2 = input_2.reshape(N, 3, -1)
    soft_hist = SoftHistogram(bins=bit, min=0, max=1, sigma=400)
    loss = []
    # img:4,3,96,96  hist:4,9,256
    # for n in range(N):
    #     # TODO 简化
    #     cls = seg_pred_cls[n]  # (H * W)
    #     img1 = input_1[n]
    #     img2 = input_2[n]
        # loss1 = soft_hist(img1[0])
    for c in range(region_num):
        cls_index = torch.nonzero(seg_pred_cls==c).squeeze()
        if not cls_index.numel():
            continue
        else:
            if len(cls_index.shape)==1:
                cls_index=cls_index[1]
            else:
                cls_index =cls_index [:, 1]
        img1_index = input_1[:,:, cls_index]
        img2_index = input_2[:,:, cls_index]
        if len(img1_index.shape)==2:
            img1_index=img1_index.unsqueeze(-1)
            img2_index = img2_index.unsqueeze(-1)
        for i in range(input_1.shape[1]):
            img1_hist = soft_hist(img1_index[:,i])
            img2_hist = soft_hist(img2_index[:,i])
            loss.append(F.l1_loss(img1_hist, img2_hist))
    loss = sum(loss) / (N*H*W)#N*H*W
    return loss

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        # target_tensor.to("cuda:0")
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            # loss = 0
            target_tensor=[]
            # for input_i in input:
            pred = input[-1]
            target_tensor= self.get_target_tensor(pred, target_is_real)
                # target_tensor=target_tensor.cuda()
                # loss += self.loss(pred, target_tensor)
            return target_tensor
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)

            return target_tensor


NAMES = {
    'vgg11': [
        'conv1_1', 'relu1_1', 'pool1', 'conv2_1', 'relu2_1', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
        'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
        'pool5'
    ],
    'vgg13': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'
    ],
    'vgg16': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
        'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
        'pool5'
    ],
    'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    ]
}

VGG_PRETRAIN_PATH = '/data/wuhongjun/project/seg/pretrain_ckpt/vgg19-dcbb9e9d.pth'

def insert_bn(names):
    """Insert bn layer after each conv.

    Args:
        names (list): The list of layer names.

    Returns:
        list: The list of layer names with bn layers.
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if 'conv' in name:
            position = name.replace('conv', '')
            names_bn.append('bn' + position)
    return names_bn

class VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): Forward function returns the corresponding
            features according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image. Importantly,
            the input feature must in the range [0, 1]. Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
        remove_pooling (bool): If true, the max pooling operations in VGG net
            will be removed. Default: False.
        pooling_stride (int): The stride of max pooling operation. Default: 2.
    """

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 requires_grad=False,
                 remove_pooling=False,
                 pooling_stride=2):
        super(VGGFeatureExtractor, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        self.names = NAMES[vgg_type.replace('_bn', '')]
        if 'bn' in vgg_type:
            self.names = insert_bn(self.names)

        # only borrow layers that will be used to avoid unused params
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx

        if os.path.exists(VGG_PRETRAIN_PATH):
            vgg_net = getattr(vgg, vgg_type)(pretrained=False)
            state_dict = torch.load(VGG_PRETRAIN_PATH, map_location=lambda storage, loc: storage)
            vgg_net.load_state_dict(state_dict)
        else:
            vgg_net = getattr(vgg, vgg_type)(pretrained=True)

        features = vgg_net.features[:max_idx + 1]

        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if 'pool' in k:
                # if remove_pooling is true, pooling operation will be removed
                if remove_pooling:
                    continue
                else:
                    # in some cases, we may want to change the default stride
                    modified_net[k] = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v

        self.vgg_net = nn.Sequential(modified_net)

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()

        return output

class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

def nor(data):
    mean = torch.mean(data)
    std = torch.std(data)

    # 对数据进行标准化
    normalized_data = (data - mean) / std
    return normalized_data

def pc(x,y):

    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    std_y=torch.std(y)
    std_x=torch.std(x)
    # 计算协方差
    covariance = torch.mean((x-mean_x)  * (y - mean_y))/(std_x*std_y+1e-6)
    return covariance

def codist(x, y):
    # if x.shape == y.shape:
    #     # return (torch.sort(x,dim=0)[0] - torch.sort(y,dim=0)[0]) ** 2
    #     return (x-y) ** 2
    # else:
        # prod=torch.cartesian_prod(torch.var(x, dim=1), torch.var(y, dim=1))
        # prod=(prod[:,0]-prod[:,1])**2
        # factor=prod.view(x.shape[0],y.shape[0])
        # factor=1-torch.exp(-factor)#factor/(factor+1)
        # return torch.sum(x ** 2, dim=1, keepdim=True) + \
        #        torch.sum(y ** 2, dim=1) - 2 * \
        #        torch.matmul(x, y.t())
        return (torch.sum(x ** 2) +torch.sum(y ** 2) - 2 *torch.sum(x*y))


def cca_loss(output1, output2):
    output1 = torch.flatten(output1.permute(1, 0, 2, 3), 1)
    output2 = torch.flatten(output2.permute(1, 0, 2, 3), 1)
    # metric=SpearmanCorrCoef()
    # # 计算视图1和视图2在潜在空间中的均值
    mean1 = torch.mean(output1, dim=0, keepdim=True)
    mean2 = torch.mean(output2, dim=0, keepdim=True)

    # 将视图1和视图2的输出零中心化
    output1_centered = output1 - mean1
    output2_centered = output2 - mean2

    r_num =torch.sum(output1_centered*output2_centered,dim=0)
    r_den = torch.sqrt(torch.sum(output1_centered ** 2,dim=0) * torch.sum(output1_centered ** 2,dim=0))
    r= r_num /( r_den+1e-5)
    # 计算视图1和视图2在潜在空间中的协方差矩阵
    # cov_matrix = torch.matmul(output1_centered.t(), output2_centered)
    #
    # # 计算视图1和视图2在潜在空间中的标准差
    # std1 = torch.std(output1_centered, dim=0, keepdim=True)
    # std2 = torch.std(output2_centered, dim=0, keepdim=True)
    #
    # # 计算典范相关系数
    # cca_corr = torch.mean(cov_matrix / (std1 * std2))

    # 返回损失，即1减去典范相关系数的平方（因为我们的目标是使相关性接近于1）
    return 1 - torch.mean(r)#torch.pow(cca_corr, 2)
