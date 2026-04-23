import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
import torchvision
import math
import copy
# import skimage
from sklearn.cluster import KMeans,MiniBatchKMeans
# from skimage.feature import graycomatrix
from pynverse import inversefunc

class FFTLoss(nn.Module):
    """L1 loss in frequency domain with FFT.
    Args:
    loss_weight (float): Loss weight for FFT loss. Default: 1.0.
    reduction (str): Specifies the reduction to apply to the output.
    Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
        pred (Tensor): of shape (..., C, H, W). Predicted tensor.
        target (Tensor): of shape (..., C, H, W). Ground truth tensor.
        weight (Tensor, optional): of shape (..., C, H, W). Element-wise
        weights. Default: None.
        """
        # pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.fft.rfft2(pred.float(), norm='backward')
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        # target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target.float(), norm='backward')
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        return self.loss_weight * F.l1_loss(pred_fft, target_fft, reduction=self.reduction)

def hdr_loss(pred,gt):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    f = (lambda x: x*(a*x+b)/(x*(c*x+d)+e))
    invcube = inversefunc(f)

    return 0

def get_light_true_channel(image, patch_size):
    """
    计算图像的暗通道先验
    :param image: 输入图像，形状为 (C, H, W)
    :param patch_size: 局部窗口的大小
    :return: 暗通道图像，形状为 (H, W)
    """

    # 取每个通道的最小值
    min_channel = torch.max(image, dim=1,keepdim=True)[0]

    # 对最小值图像应用最小滤波
    pad_size = patch_size // 2
    padded_min_channel = F.pad(min_channel, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    dark_channel = F.max_pool2d(padded_min_channel, kernel_size=patch_size, stride=1)

    return dark_channel.squeeze(dim=1)

def create_gaussian_grid(data, std=1.0, size=100):
    """
    将一维数据升维成二维数据，第二维是以数据为均值的高斯分布
    :param data: 一维张量，包含原始数据
    :param std: 高斯分布的标准差
    :param size: 生成的二维数据的大小
    :return: 二维张量
    """
    # 确保数据为浮点型
    data = data.float()

    # 创建一个一维坐标网格
    x = torch.arange(size).cuda().float()/size

    # 计算高斯分布
    gaussian_2d = torch.exp(-(x.unsqueeze(0) - data.unsqueeze(1)) ** 2 / (2 * std ** 2))

    # 归一化每个分布，使其最大值为1
    # gaussian_2d /= gaussian_2d.max(dim=1, keepdim=True)[0]

    return gaussian_2d


def shift_image(tensor, direction, shift):
    """
    Shift the image tensor in the specified direction.

    Args:
        tensor (torch.Tensor): The input image tensor of shape (B, C, H, W).
        direction (str): Direction to shift ('up', 'down', 'left', 'right').
        shift (int): Number of pixels to shift.

    Returns:
        torch.Tensor: The shifted image tensor.
    """
    # B, C, H, W = tensor.size()

    if direction == 'up':
        padding = (0, 0, shift, 0)  # (left, right, top, bottom)
    elif direction == 'down':
        padding = (0, 0, 0, shift)
    elif direction == 'left':
        padding = (shift, 0, 0, 0)
    elif direction == 'right':
        padding = (0, shift, 0, 0)
    else:
        raise ValueError("Direction should be 'up', 'down', 'left', or 'right'.")

    # Pad the tensor
    padded_tensor = F.pad(tensor, padding, mode='constant', value=0)

    # Crop the tensor to original size
    if direction == 'up':
        return padded_tensor[:, :, :-shift, :]
    elif direction == 'down':
        return padded_tensor[:, :, shift:, :]
    elif direction == 'left':
        return padded_tensor[:, :, :, :-shift]
    elif direction == 'right':
        return padded_tensor[:, :, :, shift:]

def create_grid_5round(data, std=1.0, size=100):
    gaussian_2ds=[]
    if len(data.shape)==3:
        data=data.unsqueeze(1)
    for i in range(5):
        if i==0:
            shift_data=data
        elif i==1:
            shift_data=shift_image(data,'up',1)
        elif i==2:
            shift_data=shift_image(data,'down',1)
        elif i==3:
            shift_data=shift_image(data,'left',1)
        elif i == 4:
            shift_data = shift_image(data, 'right', 1)
        shift_data=shift_data.flatten()
        # data = data.float()

        # 创建一个一维坐标网格
        x = torch.arange(size).cuda().float()/size

        # 计算高斯分布
        gaussian_2d = -(x.unsqueeze(0) - shift_data.unsqueeze(1)) ** 2
        gaussian_2ds.append(gaussian_2d)
    # 归一化每个分布，使其最大值为1
    # gaussian_2d /= gaussian_2d.max(dim=1, keepdim=True)[0]

    return gaussian_2ds


def create_grid(data, std=1.0, size=100):
    """
    将一维数据升维成二维数据，第二维是以数据为均值的高斯分布
    :param data: 一维张量，包含原始数据
    :param std: 高斯分布的标准差
    :param size: 生成的二维数据的大小
    :return: 二维张量
    """
    # 确保数据为浮点型
    data = data.float()

    # 创建一个一维坐标网格
    x = torch.arange(size).cuda().float()/size

    # 计算高斯分布
    gaussian_2d = -(x.unsqueeze(0) - data.unsqueeze(1)) ** 2

    # 归一化每个分布，使其最大值为1
    # gaussian_2d /= gaussian_2d.max(dim=1, keepdim=True)[0]

    return gaussian_2d

def create_dist_grid(data,size=100):
    """
    将一维数据升维成二维数据，第二维是以数据为均值的高斯分布
    :param data: 一维张量，包含原始数据
    :param std: 高斯分布的标准差
    :param size: 生成的二维数据的大小
    :return: 二维张量
    """
    # 确保数据为浮点型
    data = data.float()

    # 创建一个一维坐标网格
    x = torch.arange(size).cuda().float()/size

    # 计算高斯分布
    gaussian_2d = (x.unsqueeze(0) - data.unsqueeze(1)) ** 2

    return gaussian_2d


def get_light_channel(image, patch_size):
    """
    计算图像的暗通道先验
    :param image: 输入图像，形状为 (C, H, W)
    :param patch_size: 局部窗口的大小
    :return: 暗通道图像，形状为 (H, W)
    """

    # 取每个通道的最小值
    min_channel = torch.min(image, dim=1,keepdim=True)[0]

    # 对最小值图像应用最小滤波
    pad_size = patch_size // 2
    padded_min_channel = F.pad(min_channel, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    dark_channel = F.max_pool2d(padded_min_channel, kernel_size=patch_size, stride=1)

    return dark_channel.squeeze(dim=1)

def get_dark_channel(image, patch_size):
    """
    计算图像的暗通道先验
    :param image: 输入图像，形状为 (C, H, W)
    :param patch_size: 局部窗口的大小
    :return: 暗通道图像，形状为 (H, W)
    """

    # 取每个通道的最小值
    min_channel = torch.min(image, dim=1,keepdim=True)[0]

    # 对最小值图像应用最小滤波
    pad_size = patch_size // 2
    padded_min_channel = F.pad(min_channel, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    dark_channel = F.max_pool2d(padded_min_channel, kernel_size=patch_size, stride=1)

    return dark_channel.squeeze(dim=1)


def get_light_channel_real(image, patch_size):
    """
    计算图像的暗通道先验
    :param image: 输入图像，形状为 (C, H, W)
    :param patch_size: 局部窗口的大小
    :return: 暗通道图像，形状为 (H, W)
    """

    # 取每个通道的最小值
    min_channel = torch.max(image, dim=1,keepdim=True)[0]

    # 对最小值图像应用最小滤波
    pad_size = patch_size // 2
    padded_min_channel = F.pad(min_channel, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    dark_channel = F.max_pool2d(padded_min_channel, kernel_size=patch_size, stride=1)

    return dark_channel.squeeze(dim=1)


def rgb2ycbcr(rgb):
    y=rgb[:,0,:,:]*0.299+rgb[:,1,:,:]*0.564+rgb[:,2,:,:]*0.098+16/256
    cb = rgb[:, 0, :, :] *(-0.148) + rgb[:, 1, :, :] * (-0.291) + rgb[:, 2, :, :] * 0.439 + 128/256
    cr = rgb[:, 0, :, :] * 0.439 + rgb[:, 1, :, :] * (-0.368) + rgb[:, 2, :, :] * (-0.071) + 128/256
    return torch.cat((y.unsqueeze(dim=1),cb.unsqueeze(dim=1),cr.unsqueeze(dim=1)),dim=1)

def ycbcr2rgb(y):
    r=1.164*(y[:,0]-16/256)+1.596*(y[:,2]-128/256)
    g=1.164*(y[:,0]-16/256)-0.813*(y[:,2]-128/256)-0.392*(y[:,1]-128/256)
    b = 1.164 * (y[:,0] - 16/256) +2.017 * (y[:,1] - 128/256)
    return torch.cat((r.unsqueeze(dim=1),g.unsqueeze(dim=1),b.unsqueeze(dim=1)),dim=1)

# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
def gram_matrix(feat):
    """
    Calculate gram matrix used in style loss
    https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    """
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)

    return gram

# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1/v2 )  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret
class SSIM_train(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM_train, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size).cuda()

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).cuda().type(img1.dtype)
            self.window = window
            self.channel = channel

        return 1-ssim_train(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size).cuda()

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
def ssim_train(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    # C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1/v2 )  # contrast sensitivity

    # ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    #
    # if size_average:
    #     ret = ssim_map.mean()
    # else:
    #     ret = ssim_map.mean(1).mean(1).mean(1)
    #
    # if full:
    #     return ret, cs
    return cs

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.float64)
    mse = torch.mean((img1 - img2)**2)
    if mse == 0:
        return float(80)
    return 20 * math.log10(255.0 / math.sqrt(mse))


#用于attention的裁剪
def crop_cpu(img,crop_sz,step):
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        c,h, w = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list = []
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[:,x:x + crop_sz, y:y + crop_sz]
            lr_list.append(crop_img)
    # h = x + crop_sz
    # w = y + crop_sz
    return lr_list, num_h, num_w

def attention_combine(att_maps,images,model):
    kernel=20
    step=10
    pics=images.shape[0]
    results = torch.zeros_like(images)
    if torch.cuda.is_available():
        results=results.cuda()
    images=images.cpu().detach().numpy()
    att_maps = att_maps.cpu().detach().numpy()
    for i in range(pics):
        print('第{}个'.format(i))
        image=images[i,...]
        att_map=att_maps[i,...]
        im_list, num_h, num_w=crop_cpu(image,kernel,step)
        att_list, _, _ = crop_cpu(att_map, kernel, step)
        for j in range(num_h):
            for k in range(num_w):
                tem=torch.from_numpy(im_list[j * num_w + k]).unsqueeze(dim=0)
                if torch.cuda.is_available():
                    tem=tem.cuda()
                avg=np.mean(att_list[j*num_w+k])
                if avg>0.85:
                    result=model(tem,3)

                elif avg>0.8:
                    result=model(tem,2)
                else:
                    result=model(tem,1)
                torchvision.utils.save_image(result, 'result_.png')
                results[i,:,j*step:j*step+kernel,k*step:k*step+kernel]=result.squeeze()
                torchvision.utils.save_image(results[0], 'result.png')
    # for j in range(1, num_w):
    #         results[:,:,:, j * step:j * step + (kernel - step) ]/= 2
    # for k in range(1, num_h):
    #         results[:, :,k * step :k * step  + (kernel - step),:]/= 2
    return results

#颜色损失
def color_loss(x,y):
    b, c, h, w = x.shape
    true_reflect_view = x.view(b, c, h * w).permute(0, 2, 1)
    pred_reflect_view = y.view(b, c, h * w).permute(0, 2, 1)  # 16 x (512x512) x 3
    true_reflect_norm = torch.nn.functional.normalize(true_reflect_view, dim=-1)
    pred_reflect_norm = torch.nn.functional.normalize(pred_reflect_view, dim=-1)
    cose_value = true_reflect_norm * pred_reflect_norm
    cose_value = torch.sum(cose_value, dim=-1)  # 16 x (512x512)  # print(cose_value.min(), cose_value.max())
    color_loss = torch.mean(1 - cose_value)
    return color_loss

def weighted_color_loss(pred,gt):
    b, c, h, w = pred.shape
    true_reflect_view = pred.contiguous().view(b, c, h * w).permute(0, 2, 1)
    pred_reflect_view = gt.contiguous().view(b, c, h * w).permute(0, 2, 1)  # 16 x (512x512) x 3
    true_reflect_norm = torch.nn.functional.normalize(true_reflect_view, dim=-1)
    pred_reflect_norm = torch.nn.functional.normalize(pred_reflect_view, dim=-1)
    cose_value = true_reflect_norm * pred_reflect_norm
    cose_value = torch.sum(cose_value, dim=-1)  # 16 x (512x512)  # print(cose_value.min(), cose_value.max())

    weight=torch.mean(gt,dim=1).unsqueeze(1)
    min_w = torch.min(weight)
    max_w = torch.max(weight)
    weight = (weight - min_w) / (max_w - min_w)
    # a=weight.cpu().numpy()
    color_loss = torch.mean((1 - cose_value)*weight.view(b,h * w))
    return color_loss

def cos_loss(pred,gt,diff_weight):
    b, c, h, w = pred.shape
    true_reflect_view = pred.contiguous().view(b, c, h * w).permute(0, 2, 1)
    pred_reflect_view = gt.contiguous().view(b, c, h * w).permute(0, 2, 1)  # 16 x (512x512) x 3
    true_reflect_norm = torch.nn.functional.normalize(true_reflect_view, dim=-1)
    pred_reflect_norm = torch.nn.functional.normalize(pred_reflect_view, dim=-1)
    cose_value = true_reflect_norm * pred_reflect_norm
    cose_value = torch.sum(cose_value, dim=-1)  # 16 x (512x512)  # print(cose_value.min(), cose_value.max())

    # weight=torch.mean(gt,dim=1).unsqueeze(1)
    # min_w = torch.min(weight)
    # max_w = torch.max(weight)
    # weight = (weight - min_w) / (max_w - min_w)
    # a=weight.cpu().numpy()
    color_loss = torch.mean(( cose_value)*diff_weight.view(b,h * w))
    return color_loss

def seg_weighted_color_loss(pred,gt):
    b, c, h, w = pred.shape
    true_reflect_view = pred.contiguous().view(b, c, h * w).permute(0, 2, 1)
    pred_reflect_view = gt.contiguous().view(b, c, h * w).permute(0, 2, 1)  # 16 x (512x512) x 3
    true_reflect_norm = torch.nn.functional.normalize(true_reflect_view, dim=-1)
    pred_reflect_norm = torch.nn.functional.normalize(pred_reflect_view, dim=-1)
    cose_value = true_reflect_norm * pred_reflect_norm
    cose_value = torch.sum(cose_value, dim=-1)  # 16 x (512x512)  # print(cose_value.min(), cose_value.max())

    weight=torch.mean(gt,dim=1).unsqueeze(1)
    min_w = torch.min(weight)
    max_w = torch.max(weight)
    weight = (weight - min_w) / (max_w - min_w)
    # a=weight.cpu().numpy()
    color_loss = torch.mean((1 - cose_value)*weight.view(b,h * w))
    return color_loss

class L_color(torch.nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)


        return torch.mean(k)


def lf_3(data,pred,gt,model):
    lf_1 = torch.nn.L1Loss()
    lf_2 = SSIM()
    return lf_1(pred*model.module.attention_net(data),gt*model.module.attention_net(data))+1-lf_2(pred*model.module.attention_net(data),gt*model.module.attention_net(data))

def lf_3_single_card(data,pred,gt,model):
    lf_1 = torch.nn.L1Loss()
    lf_2 = SSIM()
    return lf_1(pred*model.attention_net(data),gt*model.attention_net(data))+1-lf_2(pred*model.attention_net(data),gt*model.attention_net(data))

def lf_attention(pred_att,gt,data):
    l1 = torch.nn.MSELoss()
    # ln = torch.nn.InstanceNorm2d(num_features=1, eps=0, affine=False, track_running_stats=False)
    # gt,_,_=rgb2ycbcr(gt)
    # data,_,_=rgb2ycbcr(data)
    gt = torch.max(gt, dim=1)
    # pred = torch.max(pred, dim=1)
    data = torch.max(data, dim=1)
    # gt = torch.clamp(gt[0], 1e-7, 1, out=None)
    # z = torch.abs(data[0] - gt) / (gt)
    gt = torch.clamp(gt[0], 1e-4, 1, out=None)
    z = torch.abs(gt-data[0] ) *(1-gt)
    # z = ln(z.unsqueeze(dim=1))
    # z=ln(z)
    # ad=z.cpu().numpy()
    return l1(z, pred_att)

def lf_noise(pred_noise,gt,data):
    lf2=nn.L1Loss()
    # # gt = torch.clamp(gt[0], 1e-7, 1, out=None)
    # # z = torch.abs(data[0] - gt) / (gt)
    # gt = torch.clamp(gt, 1e-7, 1, out=None)
    # z = torch.abs(data - gt) * (1-gt)
    # # z = ln(z.unsqueeze(dim=1))
    # # ad=z.cpu().numpy()
    return lf2(torch.max(torch.abs(gt-data)*(1-torch.clamp(data,min=1e-4,max=1)),dim=1)[0], pred_noise)

    # return lf2(torch.max(torch.abs(gt - data), dim=1)[0], pred_noise)


def cl_loss(p,b,m=3):
    loss=0
    # for i in range(m-1):
    #     for j in range(i+1,m):
    #         loss-=torch.abs(p[:,i]-p[:,j])
    # loss=torch.mean(loss)
    loss+=6*torch.mean(torch.abs(torch.sum(p,dim=1)-b/m))
    # print("cl_loss:{}".format(0.01*loss))
    return loss*1e-2
class BatchRGBToYCbCr(object):
    def __call__(self, imgs):
        return torch.stack((0. / 256. + imgs[:,0, :, :] * 0.299000 +imgs[:,1, :, :] * 0.587000 + imgs[:,2, :, :] * 0.114000,
                           128. / 256. - imgs[:,0, :, :] * 0.168736 - imgs[:,1, :, :] * 0.331264 + imgs[:,2, :, :] * 0.500000,
                           128. / 256. + imgs[:,0, :, :] * 0.500000 - imgs[:,1, :, :] * 0.418688 - imgs[:,2, :, :] * 0.081312),
                          dim=1).unsqueeze(1)
class TVLoss(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.img2Y=BatchRGBToYCbCr()
    def forward(self,x):
        x=self.img2Y(x)
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class TVLoss_l(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss_l,self).__init__()
        self.TVLoss_weight = TVLoss_weight
        # self.img2Y=BatchRGBToYCbCr()
    def forward(self,x):
        # x=self.img2Y(x)
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class TVLoss2(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss2,self).__init__()
        self.TVLoss_weight = TVLoss_weight
        # self.img2Y=BatchRGBToYCbCr()
    def forward(self,x,y):
        # x=self.img2Y(x)
        # batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = (x[:,:,1:,:]-x[:,:,:h_x-1,:])
        w_tv = (x[:,:,:,1:]-x[:,:,:,:w_x-1])
        hy_tv= (y[:,:,1:,:]-y[:,:,:h_x-1,:])
        wy_tv = (y[:,:,:,1:]-y[:,:,:,:w_x-1])
        return self.TVLoss_weight*(torch.abs(h_tv-hy_tv).sum()/count_h+torch.abs(w_tv-wy_tv).sum()/count_w)#2*(h_tv/count_h+w_tv/count_w)/batch_size


class TVLoss_coin(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss_coin,self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.img2Y=BatchRGBToYCbCr()
    def forward(self,x,mask_arr):
        x=self.img2Y(x)[:,0]
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)


        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2)#.sum()
        h_map=torch.clamp(mask_arr[:,:,1:,:]-torch.abs(mask_arr[:,:,1:,:]-mask_arr[:,:,:h_x-1,:]),min=0)   #减掉边界
        h_map=h_map.unsqueeze(1)
        h_tv=h_tv.unsqueeze(2)
        h_seg_tv=(h_map*h_tv/(torch.sum(h_map,dim=[3,4],keepdim=True)+1e-5)).sum()

        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2)#.sum()
        w_map = torch.clamp(mask_arr[:, :, :, 1:] - torch.abs(mask_arr[:, :, :, :w_x-1] - mask_arr[:, :,  :, :w_x-1]),min=0)
        w_map = w_map.unsqueeze(1)
        w_tv = w_tv.unsqueeze(2)
        w_seg_tv = (w_map * w_tv / (torch.sum(w_map, dim=[3,4], keepdim=True) + 1e-5)).sum()
        return self.TVLoss_weight*2*(w_seg_tv+h_seg_tv)/batch_size


def crop_img(img,crop_sz_h,crop_sz_w,step_h,step_w,batch_size):
    [b,c,h, w] = img.shape
    result=torch.zeros([int(b*(math.ceil((h-crop_sz_h)/step_h)+1)*(math.ceil((w-crop_sz_w)/step_w)+1)),c,crop_sz_h,crop_sz_w])
    if torch.cuda.is_available():
        result=result.cuda()
    h_space = np.arange(0, h - crop_sz_h + 1, step_h)
    if h_space[-1]!=h-crop_sz_h:
        h_space=np.append(h_space,h - crop_sz_h)
    w_space = np.arange(0, w - crop_sz_w + 1, step_w)
    if w_space[-1]!=w - crop_sz_w:
        w_space=np.append(w_space,w - crop_sz_w)
    index = 0
    num_h = 0
    num_w=0
    count=0
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            result[count*b:(count+1)*b,:,:,:]=img[:,:,x:x + crop_sz_h, y:y + crop_sz_w]
            count=count+1
    # y = torch.zeros([int(result.shape[0] / num_h / num_w), result.shape[1], num_h * num_w, result.shape[2], result.shape[-1]])
    # for i in range(y.shape[0]):
    #     h_list = np.arange(0, result.shape[0], y.shape[0]) + i
    #     y[i, ...] = result[h_list, ...].permute([1, 0, 2, 3])
    return result,h_space,w_space,b

def crop_img_fix_size(img,crop_size,amp=False):
    [b, c, h, w] = img.shape
    h_space = np.arange(0, h - crop_size + 1, crop_size)
    if h_space[-1] + crop_size  != h:
        h_space=np.append(h_space,h - crop_size )
    w_space = np.arange(0, w - crop_size + 1, crop_size)
    if w_space[-1] + crop_size  != w:
        w_space=np.append(w_space, w - crop_size)

    result = torch.zeros([int(b * len(h_space) * len(w_space)), c, crop_size,crop_size])
    if amp:
        result=result.half()
    if img.device.type !='cpu':
        result=result.cuda()
    count=0
    for x in h_space:
        for y in w_space:
            result[count*b:(count+1)*b,:,:,:]=img[:,:,x:x + crop_size, y:y + crop_size]
            count=count+1
    return result,h_space,w_space
#组合
def combine(sr_list,h_space, w_space,b,patch_h,patch_w,step_h,step_w,out_put):
    index=0
    rem=torch.zeros_like(out_put).cuda()
    for i in h_space:
        for j in w_space:
            out_put[:,:,i:i+patch_h,j:j+patch_w]+=sr_list[index*b:(index+1)*b,...]
            rem[:,:,i:i+patch_h,j:j+patch_w]+=1
            index+=1

    out_put=out_put/rem
    # a=rem.cpu().numpy()
    # sr_img=sr_img.astype('float32')
    # for j in range(1,num_w):
    #     out_put[:,:,:,j*step_w:j*step_w+(patch_w-step_w)]/=2
    #
    # for i in range(1,num_h):
    #     out_put[:,:,i*step_h:i*step_h+(patch_h-step_h),:]/=2
    return out_put

def combine_fix_size(sr_list,h_space,w_space,crop_size,out_put,b,amp=False):
    index = 0
    rem = torch.zeros_like(out_put)
    if amp:
        rem=rem.half()
    if torch.cuda.is_available():
        rem=rem.cuda()
    for i in range(len(h_space)):
        for j in range(len(w_space)):
            out_put[:, :, h_space[i]:h_space[i] + crop_size, w_space[j]:w_space[j] + crop_size] += sr_list[index * b:(index + 1) * b, ...]
            rem[:, :, h_space[i]:h_space[i] + crop_size, w_space[j]:w_space[j] + crop_size] += 1
            index += 1
    z=rem.detach().cpu().numpy()
    out_put = out_put / rem
    # a=rem.cpu().numpy()
    # sr_img=sr_img.astype('float32')
    # for j in range(1,num_w):
    #     out_put[:,:,:,j*step_w:j*step_w+(patch_w-step_w)]/=2
    #
    # for i in range(1,num_h):
    #     out_put[:,:,i*step_h:i*step_h+(patch_h-step_h),:]/=2
    return out_put
#according to psnr
def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2,padding_mode='reflect')

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

@torch.no_grad()
def getGaussianKernel(ksize, sigma=2):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center) # 元素与矩阵中心的横向距离
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2)) # 计算一维卷积核
    # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum() # 归一化
    return kernel

def bilateralFilter(batch_img, ksize, sigmaColor=None, sigmaSpace=None):
    device = batch_img.device
    if sigmaSpace is None:
        sigmaSpace = 0.15 * ksize + 0.35  # 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    if sigmaColor is None:
        sigmaColor = sigmaSpace

    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')

    # batch_img 的维度为 BxcxHxW, 因此要沿着第 二、三维度 unfold
    # patches.shape:  B x C x H x W x ksize x ksize
    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patch_dim = patches.dim()  # 6
    # 求出像素亮度差
    diff_color = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
    # 根据像素亮度差，计算权重矩阵
    weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
    # 归一化权重矩阵
    weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)

    # 获取 gaussian kernel 并将其复制成和 weight_color 形状相同的 tensor
    weights_space = getGaussianKernel(ksize, sigmaSpace).to(device)
    weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
    weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)

    # 两个权重矩阵相乘得到总的权重矩阵
    weights = weights_space * weights_color
    # 总权重矩阵的归一化参数
    weights_sum = weights.sum(dim=(-1, -2))
    # 加权平均
    weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
    return weighted_pix

def get_gt_label(pred,gt,ori_shape):
    lf = nn.MSELoss()
    PSNR = lambda mse: 10 * torch.log10(1. / mse).item() if (mse > 1e-5) else 50
    li_=torch.zeros(gt.shape[0])
    li2_=torch.zeros(gt.shape[0])
    patchs = int(pred.shape[0] / ori_shape)
    idx1 = torch.arange(0, pred.shape[0] - ori_shape, ori_shape).long()
    for i in range(gt.shape[0]):
        # torchvision.utils.save_image(gt[i],'gti.png')
        # torchvision.utils.save_image(pred[i], 'predi.png')
        li2_[i] = PSNR(lf(gt[i], pred[i]))
    for i in range(ori_shape):
        [_,idx]=torch.sort(li2_[idx1],descending=True)
        idx = idx1[idx] + i
        #第一类0.2，第二类0.4，第三类0.4
        li_[idx[int(patchs*0.6):]]=2
        li_[idx[:int(patchs*0.3)]]=0
        li_[idx[int(patchs*0.3):int(patchs*0.6)]] = 1
    return li_

#according to entropy_of_image
def get_entropy_label(x,ori_shape):
    kernel=1/8*torch.FloatTensor([[1,1,1],[1,0,1],[1,1,1]]).unsqueeze(0).unsqueeze(0).cuda()
    pred=x*255
    tem=pred[:,0,...]*0.299+pred[:,1,...]*0.587+pred[:,2,...]*0.114
    tem2=tem.unsqueeze(dim=1)
    tem=F.conv2d(tem2,kernel,padding=1)
    tem = torch.trunc(tem)
    entropy=[]
    li_=torch.zeros(x.shape[0])
    idx1=torch.arange(0,x.shape[0]-ori_shape,ori_shape).long()
    for i in range(tem.shape[0]):
        bins = torch.histc(tem[i,...].float(), 256, min=0, max=255)
        p=bins/torch.sum(bins)
        p[p==0]=1
        entropy.append(-torch.sum(p*torch.log(p)/math.log(2.0)))
    entropy=torch.Tensor(entropy)
    patchs=int(x.shape[0]/ori_shape)
    # k = KMeans(3)
    # k.fit(entropy.numpy().reshape(-1,1))
    for i in range(ori_shape):
        [_, idx] = torch.sort(torch.tensor(entropy[idx1]), descending=True)
        idx=idx1[idx]+i
        li_[idx[int(patchs * 0.75):]] = 0
        li_[idx[:int(patchs * 0.5)]] = 2
        li_[idx[int(patchs * 0.5):int(patchs * 0.75)]] = 1
    return li_

def get_entropy(x):
    kernel = 1 / 8 * torch.FloatTensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).unsqueeze(0).unsqueeze(0).cuda()
    pred = x * 255
    tem = pred[:, 0, ...] * 0.299 + pred[:, 1, ...] * 0.587 + pred[:, 2, ...] * 0.114
    tem2 = tem.unsqueeze(dim=1)
    tem = F.conv2d(tem2, kernel, padding=1)
    tem = torch.trunc(tem)
    entropy = []
    # li_ = torch.zeros(x.shape[0])
    # idx1 = torch.arange(0, x.shape[0] - ori_shape, ori_shape).long()
    for i in range(tem.shape[0]):  # 遍历所有碎片
        bins = torch.histc(tem[i, ...].float(), 256, min=0, max=255)
        p = bins / torch.sum(bins)
        p[p == 0] = 1
        entropy.append(-torch.sum(p * torch.log(p) / math.log(2.0)))
    return torch.Tensor(entropy).cuda()

def get_contrary(x,level=16):
    g_result=torch.zeros(x.shape[0])
    if x.device.type !='cpu':
        g_result=g_result.cuda()
    x = x[:, 0, ...] * 0.299 + x[:, 1, ...] * 0.587 + x[:, 2, ...] * 0.114
    x = torch.clamp(x*level,max=level-1e-5)
    for i in range(x.shape[0]):
        a=x[i,...].cpu().numpy().astype(np.uint8)
        glcm =graycomatrix(a,[2,8,16],[0, np.pi/2,np.pi,np.pi*3/2],level,normed=True,symmetric=True)
        g=0
        for j in range(level):
            for k in range(level):
                g+=(j-k)**2*glcm[j,k,...]
        g_result[i]=np.sum(g)
    return g_result

# def get_contrary(x,level=16):
#     g_result=torch.zeros(x.shape[0])
#     if x.device.type is not'cpu':
#         g_result=g_result.cuda()
#     x = x[:, 0, ...] * 0.299 + x[:, 1, ...] * 0.587 + x[:, 2, ...] * 0.114
#     x = torch.clamp(x*level,max=level-1e-5)
#     for i in range(x.shape[0]):
#         a=x[i,...].cpu().numpy().astype(np.uint8)
#         glcm =graycomatrix(a,[2,8,16],[0, np.pi/2,np.pi,np.pi*3/2],level,normed=True,symmetric=True)
#         g=0
#         for j in range(level):
#             for k in range(level):
#                 g+=(j-k)**2*glcm[j,k,...]
#         g_result[i]=np.sum(g)
#     return g_result

def get_hon(x,level=16):
    g_result = torch.zeros(x.shape[0])
    if x.device.type != 'cpu':
        g_result = g_result.cuda()
    x = x[:, 0, ...] * 0.299 + x[:, 1, ...] * 0.587 + x[:, 2, ...] * 0.114
    x = torch.clamp(x * level, max=level - 1e-5)
    for i in range(x.shape[0]):
        a = x[i, ...].cpu().numpy().astype(np.uint8)
        glcm = graycomatrix(a, [2, 8, 16], [0, np.pi / 2, np.pi, np.pi * 3 / 2], level, normed=True, symmetric=True)
        g = 0
        for j in range(level):
            for k in range(level):
                g += glcm[j, k, ...]/(1+(j - k) ** 2)
        g_result[i] = np.sum(g)
    return g_result

# def get_glcm_entropy(x,level=16):
#     g_result = torch.zeros(x.shape[0])
#     if x.device.type is not 'cpu':
#         g_result = g_result.cuda()
#     x = x[:, 0, ...] * 0.299 + x[:, 1, ...] * 0.587 + x[:, 2, ...] * 0.114
#     x = torch.clamp(x * level, max=level - 1e-5)
#     for i in range(x.shape[0]):
#         a = x[i, ...].cpu().numpy().astype(np.uint8)
#         glcm = graycomatrix(a, [2, 4, 8], [0, np.pi / 2, np.pi, np.pi * 3 / 2], level, normed=True, symmetric=False)
#         glcm[glcm==0]=1
#         g = 0
#         for j in range(level):
#             for k in range(level):
#                 g += -np.log(glcm[j, k, ...]) * glcm[j, k, ...]
#         g_result[i] = np.sum(g)
#     return g_result,glcm

def get_dissim(x,level=16):
    g_result = torch.zeros(x.shape[0])
    if x.device.type != 'cpu':
        g_result = g_result.cuda()
    x = x[:, 0, ...] * 0.299 + x[:, 1, ...] * 0.587 + x[:, 2, ...] * 0.114
    x = torch.clamp(x * level, max=level - 1e-5)
    for i in range(x.shape[0]):
        a = x[i, ...].cpu().numpy().astype(np.uint8)
        glcm = graycomatrix(a, [2, 8, 16], [0, np.pi / 2, np.pi, np.pi * 3 / 2], level, normed=True, symmetric=True)
        g = 0
        for j in range(level):
            for k in range(level):
                g += abs(j-k) * glcm[j, k, ...]
        g_result[i] = np.sum(g)
    return g_result

def get_entropy_label_kmean(x, ori_shape):
    kernel = 1 / 8 * torch.FloatTensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).unsqueeze(0).unsqueeze(0).cuda()
    pred = x * 255
    tem = pred[:, 0, ...] * 0.299 + pred[:, 1, ...] * 0.587 + pred[:, 2, ...] * 0.114
    tem2 = tem.unsqueeze(dim=1)
    tem = F.conv2d(tem2, kernel, padding=1)
    # tem=tem2
    tem = torch.trunc(tem)
    entropy = []
    # li_ = torch.zeros(x.shape[0])
    # idx1 = torch.arange(0, x.shape[0] - ori_shape, ori_shape).long()
    for i in range(tem.shape[0]):#遍历所有碎片
        bins = torch.histc(tem[i, ...].float(), 128, min=0, max=255)
        p = bins / torch.sum(bins)
        p[p == 0] = 1
        entropy.append(-torch.sum(p * torch.log(p) / math.log(2.0)))
    entropy = torch.Tensor(entropy)
    # e_n=entropy.cpu().numpy()
    k=KMeans(3,n_init=30,max_iter=int(3e5),tol=5e-6)
    if len(entropy.shape)==1:
        entropy=entropy.unsqueeze(dim=-1)
    k.fit(entropy.numpy())
    index=np.argsort(k.cluster_centers_.squeeze())
    result=copy.deepcopy(k.labels_)
    for c,i in enumerate(index):
        result[k.labels_==i]=c
    # patchs = int(x.shape[0] / ori_shape)
    return result,entropy
    # for i in range(ori_shape):
    #     [_, idx] = torch.sort(torch.tensor(entropy[idx1]), descending=True)
    #     idx = idx1[idx] + i
    #     li_[idx[int(patchs * 0.75):]] = 0
    #     li_[idx[:int(patchs * 0.5)]] = 2
    #     li_[idx[int(patchs * 0.5):int(patchs * 0.75)]] = 1
    # return li_
#         glcm = graycomatrix(a, [2, 8, 16], [0, np.pi / 2, np.pi, np.pi * 3 / 2], level, normed=True, symmetric=True)
def get_glcm_entropy(x,dist,level=64):
    x = x[:, 0, ...] * 0.299 + x[:, 1, ...] * 0.587 + x[:, 2, ...] * 0.114
    x = torch.clamp(x * level, max=level - 1e-5).int()
    glcm = torch.zeros([x.shape[0],len(dist)*4,level * level]).cuda()
    for idx,i in enumerate(dist):
        for j in range(4):
            if j==0:
                hist = (x[:,:-i,:] * level + x[:,i:,:]).flatten(start_dim=1)
            elif j==1:
                hist = (x[ :, :, :-i] * level + x[ :, :, i:]).flatten(start_dim=1)
            elif j==2:
                hist = (x[ :, i:, :] * level + x[ :, :-i, :]).flatten(start_dim=1)
            else:
                hist = (x[ :, :, i:] * level + x[ :, :, :-i]).flatten(start_dim=1)
            for z in range(hist.shape[0]):
                glcm[z,idx*4+j,:]=torch.histc(hist[z,:],level*level,min=0,max=level*level)
                glcm[z,idx*4+j,:]=glcm[z,idx*4+j,:]/torch.sum(glcm[z,idx*4+j,:])
    glcm[glcm==0]=1
    result=torch.sum(-glcm*torch.log(glcm),dim=(1,2))
    return result

def get_glcm_homogeneity(x,dist,level=64):
    x = x[:, 0, ...] * 0.299 + x[:, 1, ...] * 0.587 + x[:, 2, ...] * 0.114
    x = torch.clamp(x * level, max=level - 1e-5).int()
    glcm = torch.zeros([x.shape[0],len(dist)*4,level * level]).cuda()
    for idx,i in enumerate(dist):
        for j in range(4):
            if j==0:
                hist = (x[:,:-i,:] * level + x[:,i:,:]).flatten(start_dim=1)
            elif j==1:
                hist = (x[ :, :, :-i] * level + x[ :, :, i:]).flatten(start_dim=1)
            elif j==2:
                hist = (x[ :, i:, :] * level + x[ :, :-i, :]).flatten(start_dim=1)
            else:
                hist = (x[ :, :, i:] * level + x[ :, :, :-i]).flatten(start_dim=1)
            for z in range(hist.shape[0]):
                glcm[z,idx*4+j,:]=torch.histc(hist[z,:],level*level,min=0,max=level*level)
                glcm[z,idx*4+j,:]=glcm[z,idx*4+j,:]/torch.sum(glcm[z,idx*4+j,:])
    # glcm[glcm==0]=1
    weight=torch.zeros([level,level]).cuda()
    for i in range(1,level):
        # a=torch.tril(torch.ones(level,level), diagonal=-i).cuda()
        weight+=torch.tril(torch.ones(level,level), diagonal=-i).cuda()+torch.triu(torch.ones(level,level), diagonal=i).cuda()
    weight=weight.view(1,level*level).unsqueeze(0)
    # a=weight.cpu().numpy()
    weight=1/(1+torch.pow(weight,2))
    result=torch.sum(glcm*weight,dim=(1,2))
    return result

def get_glcm_contrast(x,dist,level=64):
    x = x[:, 0, ...] * 0.299 + x[:, 1, ...] * 0.587 + x[:, 2, ...] * 0.114
    x = torch.clamp(x * level, max=level - 1e-5).int()
    glcm = torch.zeros([x.shape[0],len(dist)*4,level * level]).cuda()
    for idx,i in enumerate(dist):
        for j in range(4):
            if j==0:
                hist = (x[:,:-i,:] * level + x[:,i:,:]).flatten(start_dim=1)
            elif j==1:
                hist = (x[ :, :, :-i] * level + x[ :, :, i:]).flatten(start_dim=1)
            elif j==2:
                hist = (x[ :, i:, :] * level + x[ :, :-i, :]).flatten(start_dim=1)
            else:
                hist = (x[ :, :, i:] * level + x[ :, :, :-i]).flatten(start_dim=1)
            for z in range(hist.shape[0]):
                glcm[z,idx*4+j,:]=torch.histc(hist[z,:],level*level,min=0,max=level*level)
                glcm[z,idx*4+j,:]=glcm[z,idx*4+j,:]/torch.sum(glcm[z,idx*4+j,:])
    # glcm[glcm==0]=1
    weight=torch.zeros([level,level]).cuda()
    for i in range(1,level):
        # a=torch.tril(torch.ones(level,level), diagonal=-i).cuda()
        weight+=torch.tril(torch.ones(level,level), diagonal=-i).cuda()+torch.triu(torch.ones(level,level), diagonal=i).cuda()
    weight=weight.view(1,level*level).unsqueeze(0)
    # a=weight.cpu().numpy()
    weight=torch.pow(weight,2)
    result=torch.sum(glcm*weight,dim=(1,2))
    return result

def get_glcm_matrix(x,dist,level):
    x = x[:, 0, ...] * 0.299 + x[:, 1, ...] * 0.587 + x[:, 2, ...] * 0.114
    x = torch.clamp(x * level, max=level - 1e-5).int()
    glcm = torch.zeros([x.shape[0], len(dist) * 4, level * level]).cuda()
    for idx, i in enumerate(dist):
        for j in range(4):
            if j == 0:
                hist = (x[:, :-i, :] * level + x[:, i:, :]).flatten(start_dim=1)
            elif j == 1:
                hist = (x[:, :, :-i] * level + x[:, :, i:]).flatten(start_dim=1)
            elif j == 2:
                hist = (x[:, i:, :] * level + x[:, :-i, :]).flatten(start_dim=1)
            else:
                hist = (x[:, :, i:] * level + x[:, :, :-i]).flatten(start_dim=1)
            for z in range(hist.shape[0]):
                glcm[z, idx * 4 + j, :] = torch.histc(hist[z, :], level * level, min=0, max=level * level)
                glcm[z, idx * 4 + j, :] = glcm[z, idx * 4 + j, :] / torch.sum(glcm[z, idx * 4 + j, :])
    return glcm

def getgreycomatrix(image,dist,level):
    glcm=torch.zeros([4,len(dist),level,level])
    for i in range(len(dist)):
        for j in range(image.shape[-2]-dist[i]):
            for k in range(image.shape[-1]-dist[i]):
                glcm[0,i,image[j,k],image[j+dist[i],k+dist[i]]]+=1
    for i in range(len(dist)):
        for j in range(dist[i],image.shape[-2]):
            for k in range(image.shape[-1]-dist[i]):
                glcm[1,i,image[j,k],image[j-dist[i],k+dist[i]]]+=1
    for i in range(len(dist)):
        for j in range(dist[i],image.shape[-2]):
            for k in range(dist[i],image.shape[-1]):
                glcm[2,i,image[j,k],image[j-dist[i],k-dist[i]]]+=1
    for i in range(len(dist)):
        for j in range(image.shape[-2]-dist[i]):
            for k in range(dist[i],image.shape[-1]):
                glcm[3,i,image[j,k],image[j+dist[i],k-dist[i]]]+=1
    return glcm

def get_G_label_kmean(x):
    level=16
    # pred = torch.clamp(x * level,max=level-1e-5)
    g_result=get_contrary(x,level)
    # tem = pred[:, 0, ...] * 0.299 + pred[:, 1, ...] * 0.587 + pred[:, 2, ...] * 0.114
    # g_result=np.zeros(tem.shape[0])
    # for i in range(tem.shape[0]):
    #     a=tem[i,...].cpu().numpy().astype(np.uint8)
    #     glcm =graycomatrix(a,[2,8,16],[0, np.pi/2,np.pi,np.pi*3/2],level,normed=True,symmetric=True)
    #     g=0
    #     for j in range(level):
    #         for k in range(level):
    #             g+=(j-k)**2*glcm[j,k,...]
    #     g_result[i]=np.sum(g)
    k = KMeans(3, n_init=10, max_iter=int(3e5), tol=1e-5,init='random')
    entropy = g_result[:, np.newaxis]
    k.fit(entropy.cpu().numpy())
    index = np.argsort(k.cluster_centers_.squeeze())
    result = copy.deepcopy(k.labels_)
    for c, i in enumerate(index):
        result[k.labels_ == i] = c
    # patchs = int(x.shape[0] / ori_shape)
    return result, entropy
    # tem=tem2
    # tem = torch.trunc(tem)
    # entropy = []
    # # li_ = torch.zeros(x.shape[0])
    # # idx1 = torch.arange(0, x.shape[0] - ori_shape, ori_shape).long()
    # for i in range(tem.shape[0]):#遍历所有碎片
    #     bins = torch.histc(tem[i, ...].float(), 128, min=0, max=255)
    #     p = bins / torch.sum(bins)
    #     p[p == 0] = 1
    #     entropy.append(-torch.sum(p * torch.log(p) / math.log(2.0)))
    # entropy = torch.Tensor(entropy)
    # # e_n=entropy.cpu().numpy()
    # k=KMeans(3,n_init=30,max_iter=int(3e5),tol=5e-6)
    # if len(entropy.shape)==1:
    #     entropy=entropy.unsqueeze(dim=-1)
    # k.fit(entropy.numpy())
    # index=np.argsort(k.cluster_centers_.squeeze())
    # result=copy.deepcopy(k.labels_)
    # for c,i in enumerate(index):
    #     result[k.labels_==i]=c
    # # patchs = int(x.shape[0] / ori_shape)
    # return result,entropy
    # for i in range(ori_shape):
    #     [_, idx] = torch.sort(torch.tensor(entropy[idx1]), descending=True)
    #     idx = idx1[idx] + i
    #     li_[idx[int(patchs * 0.75):]] = 0
    #     li_[idx[:int(patchs * 0.5)]] = 2
    #     li_[idx[int(patchs * 0.5):int(patchs * 0.75)]] = 1
    # return li_

def get_entropy_label2(x,ori_shape):
    kernel=1/8*torch.FloatTensor([[1,1,1],[1,0,1],[1,1,1]]).unsqueeze(0).unsqueeze(0).cuda()
    pred=x*255
    tem=pred[:,0,...]*0.299+pred[:,1,...]*0.587+pred[:,2,...]*0.114
    tem2=tem.unsqueeze(dim=1)
    tem=F.conv2d(tem2,kernel,padding=1)
    tem = torch.trunc(tem)
    entropy=[]
    li_=torch.zeros(x.shape[0])
    idx1=torch.arange(0,x.shape[0]-ori_shape,ori_shape).long()
    for i in range(tem.shape[0]):
        bins = torch.histc(tem[i,...].float(), 256, min=0, max=255)
        p=bins/torch.sum(bins)
        p[p==0]=1
        entropy.append(-torch.sum(p*torch.log(p)/math.log(2.0)))
    entropy=torch.Tensor(entropy)
    patchs=int(x.shape[0]/ori_shape)
    for i in range(ori_shape):
        [_, idx] = torch.sort(torch.tensor(entropy[idx1]), descending=True)
        idx=idx1[idx]+i
        li_[idx[int(patchs * 0.75):]] = 0
        li_[idx[:int(patchs * 0.5)]] = 1
        li_[idx[int(patchs * 0.5):int(patchs * 0.75)]] = 0
    return li_

def get_entropy_label1(x,ori_shape):
    kernel=1/8*torch.FloatTensor([[1,1,1],[1,0,1],[1,1,1]]).unsqueeze(0).unsqueeze(0).cuda()
    pred=x*255
    tem=pred[:,0,...]*0.299+pred[:,1,...]*0.587+pred[:,2,...]*0.114
    tem2=tem.unsqueeze(dim=1)
    tem=F.conv2d(tem2,kernel,padding=1)
    tem = torch.trunc(tem)
    entropy=[]
    li_=torch.zeros(x.shape[0])
    idx1=torch.arange(0,x.shape[0]-ori_shape,ori_shape).long()
    for i in range(tem.shape[0]):
        bins = torch.histc(tem[i,...].float(), 256, min=0, max=255)
        p=bins/torch.sum(bins)
        p[p==0]=1
        entropy.append(-torch.sum(p*torch.log(p)/math.log(2.0)))
    entropy=torch.Tensor(entropy)
    patchs=int(x.shape[0]/ori_shape)
    for i in range(ori_shape):
        [_, idx] = torch.sort(torch.tensor(entropy[idx1]), descending=True)
        idx=idx1[idx]+i
        li_[idx[int(patchs * 0.75):]] = 2
        li_[idx[:int(patchs * 0.5)]] = 2
        li_[idx[int(patchs * 0.5):int(patchs * 0.75)]] = 2
    return li_

def get_entropy_two_label(x,ori_shape,att_map_seg):
    kernel = 1 / 8 * torch.FloatTensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).unsqueeze(0).unsqueeze(0).cuda()
    pred = x * 255
    tem = pred[:, 0, ...] * 0.299 + pred[:, 1, ...] * 0.587 + pred[:, 2, ...] * 0.114
    tem2 = tem.unsqueeze(dim=1)
    tem = F.conv2d(tem2, kernel, padding=1)
    tem = torch.trunc(tem)
    entropy = []
    li_ = torch.zeros(x.shape[0])
    idx1 = torch.arange(0, x.shape[0] - ori_shape, ori_shape).long()
    for i in range(tem.shape[0]):
        bins = torch.histc(tem[i, ...], 256, min=0, max=255)
        p = bins / torch.sum(bins)
        p[p == 0] = 1
        entropy.append(-torch.sum(p * torch.log(p) / math.log(2.0)))
    entropy = torch.Tensor(entropy).cuda()
    patchs = int(x.shape[0] / ori_shape)
    weight=torch.mean(att_map_seg,dim=[1,2,3])*100
    entropy=weight+entropy
    for i in range(ori_shape):
        [_, j] = torch.sort(torch.tensor(entropy[idx1]), descending=True)
        idx=idx1[j]+i
        # torchvision.utils.save_image(garma(data,-0.3293,1.1258,2.4)[i], 'deserver_full.png')
        # torchvision.utils.save_image(x[idx[0],...],'deserver.png')
        # torchvision.utils.save_image(y[idx[0], ...], 'deserver_gt.png')
        # torchvision.utils.save_image(x[idx[-1], ...], 'deserver_not.png')
        # torchvision.utils.save_image(y[idx[-1], ...], 'deserver_not_gt.png')
        # torchvision.utils.save_image(y[idx[13], ...], 'deserver_mid_gt.png')
        li_[idx[int(patchs * 0.8):]] = 0
        li_[idx[:int(patchs * 0.6)]] = 2
        li_[idx[int(patchs * 0.6):int(patchs * 0.8)]] = 1
    return li_
#gama变换
def garma(img,a,b,k):
    return exp(b*(1-k**a))*torch.pow(img,k**a)
def garma_np(img,a,b,k):
    return exp(b*(1-k**a))*np.float_power(img,k**a)

def get_size(ori_shape,num_of_patch):
    h=ori_shape[-2]
    step_h=np.floor(h/num_of_patch)
    patch_h=h-(num_of_patch-1)*step_h

    w = ori_shape[-1]
    step_w = np.floor(w / num_of_patch)
    patch_w = w - (num_of_patch - 1) * step_w

    while True:
        if (step_h<=patch_h*(num_of_patch)/(num_of_patch+1)):
            break
        patch_h+=(num_of_patch-1)
        step_h-=1

    while True:
        if (step_w<=patch_w*(num_of_patch)/(num_of_patch+1)):
            break
        patch_w+=(num_of_patch-1)
        step_w-=1
    return int(patch_h),int(patch_w),int(step_h),int(step_w)

def check_image_size(window_size, x):
    _, _, h, w = x.size()
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x
def crop_piece(cropsize,x):
    # B,C,H,W=x.shape
    B, C, H, W = x.shape
    x=F.unfold(x,kernel_size=cropsize,stride=cropsize)
    x=x.permute(0, 2, 1).view(B, -1, C, cropsize, cropsize)
    fullshape=x.shape
    L_H=int(H/cropsize)
    L_W=int(W/cropsize)
    return x.contiguous().view(-1,C,cropsize,cropsize),fullshape,[B,C,H,W],[L_H,L_W]

def merge_piece(fullsize,orisize,L_len,x):
    B,L,C,H,W=fullsize
    x=x.view(fullsize).permute(0,2,3,4,1)
    x=x.view(B,-1,L)
    x=F.fold(x,output_size=(orisize[-2],orisize[-1]),stride=(int(orisize[-2]/L_len[-2]),int(orisize[-1]/L_len[-1])),kernel_size=(int(orisize[-2]/L_len[-2]),int(orisize[-1]/L_len[-1])))
    return x

def classify_by_glcm_entropy(img,center,var):
    c_result = torch.zeros([img.shape[0], 3])
    entropy = get_glcm_entropy(img, [2, 4, 8], 64)
    c_total = torch.mean(entropy)
    c_ex = copy.deepcopy(center)
    c_ex += 2e-1 * (c_total - center[1])
    for i in range(3):
        a = torch.pow(entropy - c_ex[i], 2) / (2 * torch.pow(var[i], 2))
        c_result[:, i] = torch.exp(-a) / var[i]
    c_r = torch.argmax(c_result, dim=1)
    return c_r

def classify_by_glcm_entropy_kmeans(img,center):
    c_result = torch.zeros([img.shape[0], 3])
    entropy = get_glcm_entropy(img, [2, 4, 8], 64)
    c_total = torch.mean(entropy)
    c_ex = copy.deepcopy(center)
    c_ex += 2e-1 * (c_total - center[1])
    for i in range(3):
        # a = torch.pow(entropy - c_ex[i], 2) / (2 * torch.pow(var[i], 2))
        # c_result[:, i] = torch.exp(-a) / var[i]
        c_result[:, i] = torch.abs(entropy - c_ex[i])
    c_r = torch.argmin(c_result, dim=1)
    return c_r

def classify_by_glcm_homoge_gauss(img,center,var):
    c_result = torch.zeros([img.shape[0], 3])
    entropy = get_glcm_homogeneity(img, [2, 4, 8], 64)
    c_total = torch.mean(entropy)
    c_ex = copy.deepcopy(center)
    c_ex += 2e-1 * (c_total - center[1])
    for i in range(3):
        a = torch.pow(entropy - c_ex[i], 2) / (2 * torch.pow(var[i], 2))
        c_result[:, i] = torch.exp(-a) / var[i]
        c_result[:, i] = torch.abs(entropy - center[i])
    c_r = torch.argmin(c_result, dim=1)
    return c_r

def classify_by_entropy_gauss(img,center,var):
    c_result = torch.zeros([img.shape[0], 3])
    entropy = get_entropy(img)
    c_total = torch.mean(entropy)
    c_ex = copy.deepcopy(center)
    c_ex += 2e-1 * (c_total - center[1])
    for i in range(3):
        a = torch.pow(entropy - c_ex[i], 2) / (2 * torch.pow(var[i], 2))
        c_result[:, i] = torch.exp(-a) / var[i]
        c_result[:, i] = torch.abs(entropy - center[i])
    c_r = torch.argmin(c_result, dim=1)
    return c_r

def classify_by_glcm_contrast_gauss(img,center,var):
    c_result = torch.zeros([img.shape[0], 3])
    entropy = get_glcm_contrast(img, [2, 4, 8], 64)
    c_total = torch.mean(entropy)
    c_ex = copy.deepcopy(center)
    c_ex += 2e-1 * (c_total - center[1])
    for i in range(3):
        a = torch.pow(entropy - c_ex[i], 2) / (2 * torch.pow(var[i], 2))
        c_result[:, i] = torch.exp(-a) / var[i]
        c_result[:, i] = torch.abs(entropy - center[i])
    c_r = torch.argmin(c_result, dim=1)
    return c_r

def classify_by_glcm_entropy_wi_entropy(img,center,var):
    c_result = torch.zeros([img.shape[0], 3])
    entropy = get_glcm_entropy(img, [2, 4, 8], 64)
    c_total = torch.mean(entropy)
    c_ex = copy.deepcopy(center)
    c_ex += 2e-1 * (c_total - center[1])
    for i in range(3):
        a = torch.pow(entropy - c_ex[i], 2) / (2 * torch.pow(var[i], 2))
        c_result[:, i] = torch.exp(-a) / var[i]
    c_r = torch.argmax(c_result, dim=1)
    return c_r,entropy

def get_diff_graph(img):
    img=img[:, 0, ...] * 0.299 + img[:, 1, ...] * 0.587 + img[:, 2, ...] * 0.114
    img=img.unsqueeze(dim=1)
    img = F.pad(img, [1, 1, 1, 1], mode='replicate')
    h_x = img.size()[2]
    w_x = img.size()[3]
    h_diff_pred = torch.abs((img[:, :, 1:, :] - img[:, :, :h_x - 1, :]))[:, :, :h_x - 2, 1:w_x - 1]
    w_diff_pred = torch.abs((img[:, :, :, 1:] - img[:, :, :, :w_x - 1]))[:, :, 1:h_x - 1,:w_x-2]
    # h_diff=self.l1(h_diff_pred , h_diff_gt)
    # w_diff =self.l1(w_diff_pred , w_diff_gt)
    return h_diff_pred+w_diff_pred#self.weight * self.l1(torch.max(h_diff_pred, w_diff_pred), torch.max(h_diff_gt, w_diff_gt))

def get_high_low_signal(img):
    blur=torchvision.transforms.GaussianBlur(kernel_size=(11,11))
    low_signal=blur(img)
    high_signal=img-low_signal
    return torch.cat([torch.mean(high_signal,dim=1,keepdim=True),torch.mean(low_signal,dim=1,keepdim=True)],dim=1)


# from uti.LUT_loss import *
# class LUT_loss(nn.Module):
#     def __init__(self):
#         super(LUT_loss, self).__init__()
#         # self.tolut=
#         self.LUT0 = Generator3DLUT_identity().cuda()
#         self.LUT1 = Generator3DLUT_zero().cuda()
#         self.LUT2 = Generator3DLUT_zero().cuda()
#         self.classifier=Classifier().cuda()
#         self.L1=nn.L1Loss()
#         self.toimg = TrilinearInterpolation().cuda()
#     def generate_LUT(self,img):
#         pred = self.classifier(img).squeeze()
#
#         LUT = pred[0] * self.LUT0.LUT + pred[1] * self.LUT1.LUT + pred[2] * self.LUT2.LUT  # + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT
#
#         return LUT
#     def __call__(self,pred,gt):
#         LUT_pred = self.generate_LUT(pred)
#         LUT_gt = self.generate_LUT(gt)
#         result_pred = self.toimg(LUT_pred, pred)
#         result_gt = self.toimg(LUT_gt, gt)
#         return self.L1(result_gt,result_pred)
from uti.expandnet import *
class expand_loss(nn.Module):
    def __init__(self):
        super(expand_loss, self).__init__()
        self.net=ExpandNet().cuda()
        self.net.load_state_dict(torch.load('/data1/wuhj/Project/GEDCN/uti/weights.pth'))
        self.L1 = nn.L1Loss()
        for idx, param in enumerate(self.net.parameters()):
            param.requires_grad = False
    def __call__(self, pred, gt):
        ge=self.net(torch.cat([pred,gt],dim=0))
        [ge_pred,ge_gt]=torch.chunk(ge,2)
        # ge_gt = self.net(gt)
        return self.L1(ge_pred,ge_gt)

def get_bbox_list_wo_id(arr):
    # B,H,W=arr.shape
    # arr=torch.flatten(arr, 0, 1)
    ####最正确的版本
    met_x_as = torch.range(0, arr.shape[1] - 1, step=1).unsqueeze(1).repeat(1, arr.shape[2]).cuda()
    met_y_as = torch.range(0, arr.shape[2] - 1, step=1).unsqueeze(0).repeat(arr.shape[1], 1).cuda()

    x_max = torch.max(torch.argmax(arr * met_x_as, dim=1, ), dim=1)[0].unsqueeze(-1)
    y_max = torch.max(torch.argmax(arr * met_y_as, dim=2, ), dim=1)[0].unsqueeze(-1)

    # met_x_ds = torch.range(arr.shape[1] - 1, 0, step=-1).unsqueeze(1).repeat(1, arr.shape[2]).cuda()
    # met_y_ds = torch.range(arr.shape[2] - 1, 0, step=-1).unsqueeze(0).repeat(arr.shape[1], 1).cuda()
    x_min = arr.shape[1]-torch.max(torch.argmax(torch.flip(arr,dims=[1]) * met_x_as, dim=1, ), dim=1)[0].unsqueeze(-1)
    y_min = arr.shape[2]-torch.max(torch.argmax(torch.flip(arr,dims=[2]) * met_y_as, dim=2, ), dim=1)[0].unsqueeze(-1)
    bbox = torch.cat([y_min, x_min, y_max, x_max ], dim=-1)
    # ids=torch.range(0,B - 1,step=1).unsqueeze(1).repeat(1,C).cuda()
    # ids=torch.flatten(ids,0,1).unsqueeze(1)
    # ids=torch.range(0,bbox.shape[0]-1).unsqueeze(-1).cuda()
    # ids =torch.zeros(bbox.shape[0],1).cuda()
    return bbox

# def get_user_map(arr):
#
#     y_idx = torch.range(0, end=arr.shape[-2] - 1)
#     x_idx = torch.range(0, end=arr.shape[-1] - 1)
#     y_idxs, x_idxs = torch.meshgrid(y_idx, x_idx)
#     x_idxs = x_idxs.unsqueeze(0).repeat(arr.shape[0], 1, 1).cuda()
#     y_idxs = y_idxs.unsqueeze(0).repeat(arr.shape[0], 1, 1).cuda()
#     # idx_map=
#     if len(arr.shape)==4:
#         arr=arr.squeeze(dim=1)
#     xyxy = get_bbox_list_wo_id(arr.float())
#     center = [torch.floor((xyxy[:, 2] + xyxy[:, 0]) / 2).unsqueeze(-1).unsqueeze(-1),
#               torch.floor((xyxy[:, 1] + xyxy[:, 3]) / 2).unsqueeze(-1).unsqueeze(-1)]
#     user_map = torch.sqrt(torch.square(center[0] - x_idxs) + torch.square(center[1] - y_idxs))
#     user_map = user_map / (torch.max(torch.max(user_map, dim=1, )[0], dim=1)[0]).unsqueeze(-1).unsqueeze(-1)
#     return (1-user_map).unsqueeze(1)

def get_user_map(arr):
    alpha=0.5
    y_idx = torch.range(0, end=arr.shape[-2] - 1)
    x_idx = torch.range(0, end=arr.shape[-1] - 1)
    y_idxs, x_idxs = torch.meshgrid(y_idx, x_idx)
    x_idxs = x_idxs.unsqueeze(0).repeat(arr.shape[0], 1, 1).cuda()
    y_idxs = y_idxs.unsqueeze(0).repeat(arr.shape[0], 1, 1).cuda()
    # idx_map=
    if len(arr.shape) == 4:
        arr=arr.squeeze(dim=1)
    ##索引先第三维，后第二维
    xyxy = get_bbox_list_wo_id(arr)
    center = [torch.floor((xyxy[:, 2] + xyxy[:, 0]) / 2).unsqueeze(-1).unsqueeze(-1),
              torch.floor((xyxy[:, 1] + xyxy[:, 3]) / 2).unsqueeze(-1).unsqueeze(-1)]
    # y_len = (xyxy[:, 3] - xyxy[:, 1]) / arr.shape[-1]
    # x_len = (xyxy[:, 2] - xyxy[:, 0]) / arr.shape[-2]
    # x_ratio =( y_len / (y_len + x_len)).unsqueeze(-1).unsqueeze(-1)
    # y_ratio =( x_len / (y_len + x_len)).unsqueeze(-1).unsqueeze(-1)
    user_map = torch.sqrt(torch.square(center[0] - x_idxs) +  torch.square(center[1] - y_idxs))
    user_map = user_map / (torch.max(torch.max(user_map, dim=1, )[0], dim=1)[0]).unsqueeze(-1).unsqueeze(-1)
    user_map=1 - user_map
    user_map = user_map * alpha
    user_map[arr == 1] = 1
    if len(user_map.shape)==3:
        user_map=user_map.unsqueeze(1)
    return user_map


def stretch(map):
    mean = 0.5
    # x_len=center[:,3]-center[:,1]/map.shape[-1]
    # y_len=center[:,2]-center[:,0]/map.shape[-2]
    # blur = get_gaussian_kernel(15, channels=1).cuda()
    alaph = 1.5
    map[map >= mean] = torch.exp(alaph * (map[map >= mean] - mean)) - (1 - mean)
    map[map < mean] = 1 + mean - torch.exp(alaph * (mean - map[map < mean]))
    map = torch.clamp(map, min=0, max=1)
    return map

def get_gray(x):
    return (x[:, 0:1, ...] * 0.299 + x[:, 1:2, ...] * 0.587 + x[:, 2:3, ...] * 0.114).repeat(1,3,1,1)


def rgb2xyz(img):
    """
    RGB from 0 to 255
    :param img:
    :return:
    """
    # img=img*255.
    r, g, b = torch.split(img, 1, dim=1)

    r = torch.where(r > 0.04045, torch.pow((r+0.055) / 1.055, 2.4), r / 12.92)
    g = torch.where(g > 0.04045, torch.pow((g+0.055) / 1.055, 2.4), g / 12.92)
    b = torch.where(b > 0.04045, torch.pow((b+0.055) / 1.055, 2.4), b / 12.92)

    # r = r * 100
    # g = g * 100
    # b = b * 100

    x = r * 0.412453 + g * 0.357580 + b * 0.180423
    y = r * 0.212671 + g * 0.715160 + b * 0.072169
    z = r * 0.019334 + g * 0.119193 + b * 0.950227
    return torch.cat([x,y,z], dim=1)


def xyz2lab(xyz):
    x, y, z = torch.split(xyz, 1, dim=1)
    ref_x, ref_y, ref_z = 95.047, 100.000, 108.883
    # ref_x, ref_y, ref_z = 0.95047, 1., 1.08883
    x = x / ref_x
    y = y / ref_y
    z = z / ref_z

    x = torch.where(x > 0.008856, torch.pow( x , 1/3 ), (7.787 * x) + (16 / 116.))
    y = torch.where(y > 0.008856, torch.pow( y , 1/3 ), (7.787 * y) + (16 / 116.))
    z = torch.where(z > 0.008856, torch.pow( z , 1/3 ), (7.787 * z) + (16 / 116.))

    l = (116. * y) - 16.
    a = 500. * (x - y)
    b = 200. * (y - z)
    return torch.cat([l,a,b], dim=1)

def lab2xyz(lab):
    ref_x, ref_y, ref_z = 95.047, 100.000, 108.883
    l, a, b = torch.split(lab, 1, dim=1)
    y = (l + 16) / 116.
    x = a / 500. + y
    z = y - b / 200.

    y = torch.where(torch.pow( y , 3 ) > 0.008856, torch.pow( y , 3 ), ( y - 16 / 116. ) / 7.787)
    x = torch.where(torch.pow( x , 3 ) > 0.008856, torch.pow( x , 3 ), ( x - 16 / 116. ) / 7.787)
    z = torch.where(torch.pow( z , 3 ) > 0.008856, torch.pow( z , 3 ), ( z - 16 / 116. ) / 7.787)

    x = ref_x * x
    y = ref_y * y
    z = ref_z * z
    return torch.cat([x,y,z],dim=1)


def xyz2rgb(xyz):
    x, y, z = torch.split(xyz, 1, dim=1)

    # x = x / 100.
    # y = y / 100.
    # z = z / 100.

    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    r = torch.where(r > 0.0031308, 1.055 * torch.pow( r , ( 1 / 2.4 ) ) - 0.055,  12.92 * r)
    g = torch.where(g > 0.0031308, 1.055 * torch.pow( g , ( 1 / 2.4 ) ) - 0.055,  12.92 * g)
    b = torch.where(b > 0.0031308, 1.055 * torch.pow( b , ( 1 / 2.4 ) ) - 0.055,  12.92 * b)

    # r = torch.round(r)
    # g = torch.round(g)
    # b = torch.round(b)

    return torch.cat([r,g,b], dim=1)

def RandomMask(s, hole_range=[0,1]):
    coef = min(hole_range[0] + hole_range[1], 1.0)
    if np.random.random() < 0:
        return torch.ones(1, s, s).float()
    while True:
        mask = np.ones((s, s), np.uint8)
        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
            mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0
        def MultiFill(max_tries, max_size):
            for _ in range(np.random.randint(max_tries)):
                Fill(max_size)
        MultiFill(int(10 * coef), s // 2)
        MultiFill(int(5 * coef), s)
        mask = np.logical_and(mask, 1 - RandomBrush(int(20 * coef), s))
        hole_ratio = 1 - np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return torch.from_numpy(mask[np.newaxis, ...].astype(np.float32)).float()

def cal_kl_loss(mu_poster=None, sigma_poster=None, mu_prior=None, sigma_prior=None,eps=1e-5):
    kl_total=0.5*(torch.log(sigma_poster/(sigma_prior+ eps)+eps)+(sigma_prior+(mu_prior-mu_poster)**2)/(sigma_poster+eps)-1)
    # print(torch.mean(kl_total))
    return torch.mean(kl_total)
def cal_kl_loss_v2(mu_poster=None, sigma_poster=None, mu_prior=None, sigma_prior=None,eps=1e-5):
    # sigma_poster = sigma_poster ** 2
    # sigma_prior = sigma_prior ** 2
    sigma_poster_matrix_det = torch.abs(torch.prod(sigma_poster, dim=1))
    sigma_prior_matrix_det = torch.abs(torch.prod(sigma_prior, dim=1))

    sigma_prior_matrix_inv = 1.0 / (sigma_prior+eps)
    delta_u = (mu_prior - mu_poster)
    term1 = torch.sum(sigma_poster / (sigma_prior+eps), dim=1)
    term2 = torch.sum(delta_u * sigma_prior_matrix_inv * delta_u, 1)
    term3 = - mu_poster.shape[-1]
    term4 = torch.log(sigma_prior_matrix_det + eps) - torch.log(
        sigma_poster_matrix_det + eps)
    kl_loss = 0.5 * (term1 + term2 + term3 + term4)
    kl_loss = torch.clamp(kl_loss, 0, 10)
    # print(torch.mean(kl_loss))
    return torch.mean(kl_loss)

def gaussian_kl_divergence(mu1, sigma1, mu2, sigma2):
    # KL 散度的计算公式
    kl_div = 0.5 * (0.5*torch.log(torch.abs(sigma2 / (sigma1+1e-5))) + (sigma1 + (mu1 - mu2) ** 2) / (2 * sigma2 ) - 0.5)
    return kl_div.mean()

def gaussian_js_divergence(mu1, sigma1, mu2, sigma2):
    # 计算两个方向的 KL 散度
    kl1 = cal_kl_loss_v2(mu1, sigma1, mu2, sigma2)
    kl2 = cal_kl_loss_v2(mu2, sigma2, mu1, sigma1)

    # 计算 JS 散度
    js_divergence = 0.5 * (kl1 + kl2)

    return js_divergence


def cal_wasserstein(mu1, sigma1, mu2, sigma2):
    # 根据公式计算 Wasserstein 距离
    sigma1=torch.sqrt(sigma1)
    sigma2 = torch.sqrt(sigma2)
    distance =  torch.abs(sigma1 - sigma2)#torch.abs(mu1 - mu2) +

    return torch.mean(distance)

def cal_wasserstein_w(mu1, sigma1, mu2, sigma2):
    # 根据公式计算 Wasserstein 距离
    sigma1=torch.sqrt(torch.abs(sigma1))
    sigma2 = torch.sqrt(torch.abs(sigma2))
    distance = torch.abs(mu1 - mu2) +torch.abs(sigma1 - sigma2)

    return torch.mean(distance)

def cal_wasserstein_mean(mu1, sigma1, mu2, sigma2):
    # 根据公式计算 Wasserstein 距离
    # sigma1=torch.sqrt(torch.abs(sigma1))
    # sigma2 = torch.sqrt(torch.abs(sigma2))
    distance = torch.abs(mu1 - mu2)# +torch.abs(sigma1 - sigma2)

    return torch.mean(distance)

def cal_wasserstein_w_l2(mu1, sigma1, mu2, sigma2,loss):
    # 根据公式计算 Wasserstein 距离
    sigma1=torch.sqrt(torch.abs(sigma1.squeeze()))
    sigma2 = torch.sqrt(torch.abs(sigma2.squeeze()))
    distance = loss(mu1 .squeeze(), mu2.squeeze()) +loss(sigma1 , sigma2)

    return distance


def cal_wasserstein_w_sigma(mu1, sigma1, mu2, sigma2):
    # 根据公式计算 Wasserstein 距离
    sigma1=torch.sqrt(torch.abs(sigma1))
    sigma2 = torch.sqrt(torch.abs(sigma2))
    distance = torch.abs(mu1 - mu2) +torch.abs(sigma1 - sigma2)

    return torch.mean(distance)

def kl_divergence(mu1, sigma1, mu2, sigma2):
    """
    计算两个高斯分布之间的KL散度
    """
    mu1=mu1.squeeze()
    mu2 = mu2.squeeze()
    sigma1 = sigma1.squeeze()
    sigma2 = sigma2.squeeze()
    var_ratio = torch.pow(sigma1, 2) / torch.pow(sigma2, 2)
    kl = 0.5 * (torch.pow(mu2 - mu1, 2) / torch.pow(sigma2, 2) + var_ratio - 1 + torch.log(var_ratio))
    return kl.sum()

def softened_kl_divergence(mu2, sigma2,mu1, sigma1,  temperature=10):
    """
    计算两个高斯分布之间的软化KL散度
    """
    mu1 = mu1.squeeze()
    mu2 = mu2.squeeze()
    sigma1 = sigma1.squeeze()
    sigma2 = sigma2.squeeze()
    # 计算两个高斯分布的方差之比
    var_ratio = (torch.pow(sigma1, 2) + 1e-8) / (torch.pow(sigma2, 2) + 1e-8)
    # 计算两个高斯分布的均值之差的平方
    mean_diff_square = torch.pow(mu2 - mu1, 2)
    # 计算软化的KL散度
    kl = 0.5 * (var_ratio + mean_diff_square / (torch.pow(sigma2, 2) + 1e-8) - 1 - torch.log(var_ratio + 1e-8))
    # 对KL散度进行温度缩放
    kl = kl / temperature
    return kl.mean()

def schmidt(A):
    Q,_=torch.qr(A)
    return Q