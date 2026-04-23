from torch.utils.data import Dataset
import torchvision
import os
import numpy as np
from uti.argument import augment_img
import cv2
import PIL.Image as Image
import random
from skimage import io,filters,exposure
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as tr
from utils import RandomMask
from skimage.filters import gaussian

from torch.nn import functional as F

def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    img=img/255.
    # img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    """USM sharpening.

    Input image: I; Blurry image: B.
    1. sharp = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * sharp + (1 - Mask) * I


    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharp = img + weight * residual
    sharp = np.clip(sharp, 0, 1)
    result=soft_mask * sharp + (1 - soft_mask) * img
    # result=cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
    return result*255.

def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)

def uint2tensor3_nodiv(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()

class Dataset_train(Dataset):
    def __init__(self, input_root, label_root, fis=256,use_mixup=False):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.full_img_size=fis
        self.use_mixup=use_mixup

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        if self.use_mixup:
            idx_add=torch.randint(0,self.__len__(),size=[1])
            lam = np.random.beta(1,1)

            input_img = lam * input_img + (1-lam) * io.imread(os.path.join(self.input_root, self.input_files[idx_add]))
            label_img = lam * label_img + (1 - lam) * io.imread(os.path.join(self.label_root, self.label_files[idx_add]))

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]

        mode = random.randint(0, 7)
        input_img = augment_img(input_img, mode=mode)
        label_img = augment_img(label_img, mode=mode)

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis = self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
        # torchvision.utils.save_image(label_img,'cv2.png')
        return (input_img, label_img)


class Dataset_train_clemask(Dataset):
    def __init__(self, input_root, label_root, fis=256,use_mixup=False):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.full_img_size=fis
        self.use_mixup=use_mixup

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        if self.use_mixup:
            idx_add=torch.randint(0,self.__len__(),size=[1])
            lam = np.random.beta(1,1)

            input_img = lam * input_img + (1-lam) * io.imread(os.path.join(self.input_root, self.input_files[idx_add]))
            label_img = lam * label_img + (1 - lam) * io.imread(os.path.join(self.label_root, self.label_files[idx_add]))

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]

        mode = random.randint(0, 7)
        input_img = augment_img(input_img, mode=mode)
        label_img = augment_img(label_img, mode=mode)

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)

        mask = RandomMask(input_img.shape[-1], hole_range=[0, 0.6])
        mask = cv2.GaussianBlur(mask[0].numpy(), (25, 25), 0)
        mask = uint2tensor3(mask)

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis = self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
            mask = TF.pad(mask, (0, 0, padw, padh), padding_mode='reflect')
        # torchvision.utils.save_image(label_img,'cv2.png')
        return (input_img, label_img,mask)

class Dataset_train_baid(Dataset):
    def __init__(self, input_root, label_root,mask_root, fis=256,use_mixup=False):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()
        self.mask_root=mask_root
        self.mask_files = os.listdir(mask_root)
        self.mask_files.sort()
        self.full_img_size=fis
        self.use_mixup=use_mixup

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        mask_img_path = os.path.join(self.mask_root, self.label_files[index])
        mask_img = io.imread(mask_img_path)[:,:,np.newaxis]
        if self.use_mixup:
            idx_add=torch.randint(0,self.__len__(),size=[1])
            lam = np.random.beta(1,1)

            input_img = lam * input_img + (1-lam) * io.imread(os.path.join(self.input_root, self.input_files[idx_add]))
            label_img = lam * label_img + (1 - lam) * io.imread(os.path.join(self.label_root, self.label_files[idx_add]))

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        mask_img = mask_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size,:]


        mode = random.randint(0, 7)
        input_img = augment_img(input_img, mode=mode)
        label_img = augment_img(label_img, mode=mode)
        mask_img = augment_img(mask_img, mode=mode)

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        mask_img = uint2tensor3(mask_img)

        # mask = RandomMask(input_img.shape[-1], hole_range=[0, 0.6])
        # mask = cv2.GaussianBlur(mask[0].numpy(), (25, 25), 0)
        # mask = uint2tensor3(mask)

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis = self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
            mask_img = TF.pad(mask_img, (0, 0, padw, padh), padding_mode='reflect')
        # torchvision.utils.save_image(label_img,'cv2.png')
        return (input_img, label_img,mask_img)

class Dataset_train_baid_nomask(Dataset):
    def __init__(self, input_root, label_root, fis=256,use_mixup=False):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()
        self.full_img_size=fis
        self.use_mixup=use_mixup

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        if self.use_mixup:
            idx_add=torch.randint(0,self.__len__(),size=[1])
            lam = np.random.beta(1,1)

            input_img = lam * input_img + (1-lam) * io.imread(os.path.join(self.input_root, self.input_files[idx_add]))
            label_img = lam * label_img + (1 - lam) * io.imread(os.path.join(self.label_root, self.label_files[idx_add]))

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]


        mode = random.randint(0, 7)
        input_img = augment_img(input_img, mode=mode)
        label_img = augment_img(label_img, mode=mode)

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)

        # mask = RandomMask(input_img.shape[-1], hole_range=[0, 0.6])
        # mask = cv2.GaussianBlur(mask[0].numpy(), (25, 25), 0)
        # mask = uint2tensor3(mask)

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis = self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
        # torchvision.utils.save_image(label_img,'cv2.png')
        return (input_img, label_img)


class Dataset_train_map(Dataset):
    def __init__(self, input_root, label_root, input_map_root,label_map_root,fis=256,use_mixup=False):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()

        self.input_map_root=input_map_root
        self.label_map_root=label_map_root

        self.full_img_size=fis
        self.use_mixup=use_mixup

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        input_mask_img=io.imread(os.path.join(self.input_map_root, self.input_files[index]))


        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        label_mask_img = io.imread(os.path.join(self.label_map_root, self.input_files[index]))

        if self.use_mixup:
            idx_add=torch.randint(0,self.__len__(),size=[1])
            lam = np.random.beta(1,1)

            input_img = lam * input_img + (1-lam) * io.imread(os.path.join(self.input_root, self.input_files[idx_add]))
            label_img = lam * label_img + (1 - lam) * io.imread(os.path.join(self.label_root, self.label_files[idx_add]))

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        input_mask_img = input_mask_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_mask_img = label_mask_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]

        mode = random.randint(0, 7)
        input_img = augment_img(input_img, mode=mode)
        label_img = augment_img(label_img, mode=mode)
        input_mask_img = augment_img(input_mask_img, mode=mode)
        label_mask_img = augment_img(label_mask_img, mode=mode)

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        input_mask_img = uint2tensor3(input_mask_img)
        label_mask_img = uint2tensor3(label_mask_img)

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis = self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
            input_mask_img = TF.pad(input_mask_img, (0, 0, padw, padh), padding_mode='reflect')
            label_mask_img = TF.pad(label_mask_img, (0, 0, padw, padh), padding_mode='reflect')
        # torchvision.utils.save_image(label_img,'cv2.png')
        return (input_img, label_img, input_mask_img, label_mask_img)

class Dataset_train_map_array(Dataset):
    def __init__(self, input_root, label_root, input_map_root,label_map_root,array_root,fis=256,use_mixup=False,region=50):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()
        self.array_root=array_root
        self.array_files=os.listdir(self.array_root)
        # self.array_files.sort()


        self.input_map_root=input_map_root
        self.label_map_root=label_map_root

        self.full_img_size=fis
        self.use_mixup=use_mixup
        self.region=region
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        input_mask_img=io.imread(os.path.join(self.input_map_root, self.input_files[index]))


        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        label_mask_img = io.imread(os.path.join(self.label_map_root, self.input_files[index]))

        arr_path=os.path.join(self.array_root, self.array_files[index])
        arr=np.load(arr_path)[:,:,0]


        if self.use_mixup:
            idx_add=torch.randint(0,self.__len__(),size=[1])
            lam = np.random.beta(1,1)

            input_img = lam * input_img + (1-lam) * io.imread(os.path.join(self.input_root, self.input_files[idx_add]))
            label_img = lam * label_img + (1 - lam) * io.imread(os.path.join(self.label_root, self.label_files[idx_add]))

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        input_mask_img = input_mask_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_mask_img = label_mask_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        arr=arr[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size]

        mode = random.randint(0, 7)
        input_img = augment_img(input_img, mode=mode)
        label_img = augment_img(label_img, mode=mode)
        input_mask_img = augment_img(input_mask_img, mode=mode)
        label_mask_img = augment_img(label_mask_img, mode=mode)
        arr=augment_img(arr, mode=mode)

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        input_mask_img = uint2tensor3(input_mask_img)
        label_mask_img = uint2tensor3(label_mask_img)
        arr=torch.from_numpy(np.ascontiguousarray(arr)).to(torch.int64)
        # arr[arr>=self.region]=self.region
        # arr=F.one_hot(arr, num_classes = self.region+1).permute(2,0,1)
        # arr=arr[:-1]

        arr[arr >= self.region] = self.region-1
        arr_forloss=arr.unsqueeze(0)
        arr = F.one_hot(arr, num_classes=self.region).permute(2, 0, 1)
        # arr = arr[:-1]

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis = self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
            input_mask_img = TF.pad(input_mask_img, (0, 0, padw, padh), padding_mode='reflect')
            label_mask_img = TF.pad(label_mask_img, (0, 0, padw, padh), padding_mode='reflect')
            arr=TF.pad(arr, (0, 0, padw, padh), padding_mode='reflect')
            arr_forloss = TF.pad(arr_forloss, (0, 0, padw, padh), padding_mode='reflect')
        # torchvision.utils.save_image(label_img,'cv2.png')
        return (input_img, label_img, input_mask_img, label_mask_img,arr,arr_forloss)


class Dataset_train_map_array_msam(Dataset):
    def __init__(self, input_root, label_root, input_map_root,label_map_root,array_root,fis=256,use_mixup=False,region=50):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()
        self.array_root=array_root
        self.array_files=os.listdir(self.array_root)

        self.input_map_root=input_map_root
        self.label_map_root=label_map_root

        self.full_img_size=fis
        self.use_mixup=use_mixup
        self.region=region
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        input_mask_img=io.imread(os.path.join(self.input_map_root, self.input_files[index]))


        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        label_mask_img = io.imread(os.path.join(self.label_map_root, self.input_files[index]))

        # arr_path=os.path.join(self.array_root, self.array_files[index])
        # arr=np.load(arr_path)[:,:,0]


        if self.use_mixup:
            idx_add=torch.randint(0,self.__len__(),size=[1])
            lam = np.random.beta(1,1)

            input_img = lam * input_img + (1-lam) * io.imread(os.path.join(self.input_root, self.input_files[idx_add]))
            label_img = lam * label_img + (1 - lam) * io.imread(os.path.join(self.label_root, self.label_files[idx_add]))

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        input_mask_img = input_mask_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_mask_img = label_mask_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        # arr=arr[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size]

        mode = random.randint(0, 7)
        input_img = augment_img(input_img, mode=mode)
        label_img = augment_img(label_img, mode=mode)
        input_mask_img = augment_img(input_mask_img, mode=mode)
        label_mask_img = augment_img(label_mask_img, mode=mode)
        # arr=augment_img(arr, mode=mode)

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        input_mask_img = uint2tensor3(input_mask_img)
        label_mask_img = uint2tensor3(label_mask_img)
        # arr=torch.from_numpy(np.ascontiguousarray(arr)).to(torch.int64)
        # arr[arr>=self.region]=self.region
        # arr=F.one_hot(arr, num_classes = self.region+1).permute(2,0,1)
        # arr=arr[:-1]

        # arr[arr >= self.region] = self.region-1
        # arr_forloss=arr.unsqueeze(0)
        # arr = F.one_hot(arr, num_classes=self.region).permute(2, 0, 1)
        # arr = arr[:-1]

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis = self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
            input_mask_img = TF.pad(input_mask_img, (0, 0, padw, padh), padding_mode='reflect')
            label_mask_img = TF.pad(label_mask_img, (0, 0, padw, padh), padding_mode='reflect')
            # arr=TF.pad(arr, (0, 0, padw, padh), padding_mode='reflect')
            # arr_forloss = TF.pad(arr_forloss, (0, 0, padw, padh), padding_mode='reflect')
        input_img_np=(input_img.permute(1,2, 0).numpy()*255.).astype(np.uint8).copy()
        # torchvision.utils.save_image(label_img,'cv2.png')
        return [input_img, label_img, input_mask_img, label_mask_img,input_img_np]

class Dataset_train_map_array_box(Dataset):
    def __init__(self, input_root, label_root, input_map_root,label_map_root,array_root,fis=256,use_mixup=False,region=50):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()
        self.array_root=array_root
        self.array_files=os.listdir(self.array_root)

        self.input_map_root=input_map_root
        self.label_map_root=label_map_root

        self.full_img_size=fis
        self.use_mixup=use_mixup
        self.region=region
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        input_mask_img=io.imread(os.path.join(self.input_map_root, self.input_files[index]))


        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        label_mask_img = io.imread(os.path.join(self.label_map_root, self.input_files[index]))

        arr_path=os.path.join(self.array_root, self.array_files[index])
        arr=np.load(arr_path)[:,:,0]


        if self.use_mixup:
            idx_add=torch.randint(0,self.__len__(),size=[1])
            lam = np.random.beta(1,1)

            input_img = lam * input_img + (1-lam) * io.imread(os.path.join(self.input_root, self.input_files[idx_add]))
            label_img = lam * label_img + (1 - lam) * io.imread(os.path.join(self.label_root, self.label_files[idx_add]))

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        input_mask_img = input_mask_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_mask_img = label_mask_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        arr=arr[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size]

        mode = random.randint(0, 7)
        input_img = augment_img(input_img, mode=mode)
        label_img = augment_img(label_img, mode=mode)
        input_mask_img = augment_img(input_mask_img, mode=mode)
        label_mask_img = augment_img(label_mask_img, mode=mode)
        arr=augment_img(arr, mode=mode)

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        input_mask_img = uint2tensor3(input_mask_img)
        label_mask_img = uint2tensor3(label_mask_img)
        arr=torch.from_numpy(np.ascontiguousarray(arr)).to(torch.int64)
        # arr[arr>=self.region]=self.region
        # arr=F.one_hot(arr, num_classes = self.region+1).permute(2,0,1)
        # arr=arr[:-1]

        arr[arr >= self.region] = self.region-1
        arr_forloss=arr.unsqueeze(0)
        arr = F.one_hot(arr, num_classes=self.region).permute(2, 0, 1)
        # arr = arr[:-1]

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis = self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
            input_mask_img = TF.pad(input_mask_img, (0, 0, padw, padh), padding_mode='reflect')
            label_mask_img = TF.pad(label_mask_img, (0, 0, padw, padh), padding_mode='reflect')
            arr=TF.pad(arr, (0, 0, padw, padh), padding_mode='reflect')
            arr_forloss = TF.pad(arr_forloss, (0, 0, padw, padh), padding_mode='reflect')
        # torchvision.utils.save_image(label_img,'cv2.png')
        return (input_img, label_img, input_mask_img, label_mask_img,arr,arr_forloss)

class Dataset_train_map_array_gt(Dataset):
    def __init__(self, input_root, label_root, input_map_root,label_map_root,array_root,fis=256,use_mixup=False,region=50):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()
        self.array_root=array_root
        self.array_files=os.listdir(self.array_root)

        self.input_map_root=input_map_root
        self.label_map_root=label_map_root

        self.full_img_size=fis
        self.use_mixup=use_mixup
        self.region=region
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        input_mask_img=io.imread(os.path.join(self.input_map_root, self.input_files[index]))


        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        label_mask_img = io.imread(os.path.join(self.label_map_root, self.input_files[index]))

        arr_path=os.path.join(self.array_root, self.array_files[index])
        arr=np.load(arr_path)[:,:,0]


        if self.use_mixup:
            idx_add=torch.randint(0,self.__len__(),size=[1])
            lam = np.random.beta(1,1)

            input_img = lam * input_img + (1-lam) * io.imread(os.path.join(self.input_root, self.input_files[idx_add]))
            label_img = lam * label_img + (1 - lam) * io.imread(os.path.join(self.label_root, self.label_files[idx_add]))

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        input_mask_img = input_mask_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_mask_img = label_mask_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        arr=arr[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size]

        mode = random.randint(0, 7)
        input_img = augment_img(input_img, mode=mode)
        label_img = augment_img(label_img, mode=mode)
        input_mask_img = augment_img(input_mask_img, mode=mode)
        label_mask_img = augment_img(label_mask_img, mode=mode)
        arr=augment_img(arr, mode=mode)

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        input_mask_img = uint2tensor3(input_mask_img)
        label_mask_img = uint2tensor3(label_mask_img)
        arr=torch.from_numpy(np.ascontiguousarray(arr)).to(torch.int64)
        # arr[arr>=self.region]=self.region
        # arr=F.one_hot(arr, num_classes = self.region+1).permute(2,0,1)
        # arr=arr[:-1]

        arr[arr >= self.region] = self.region-1
        arr_forloss=arr.unsqueeze(0)
        arr = F.one_hot(arr, num_classes=self.region).permute(2, 0, 1)
        # arr = arr[:-1]

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis = self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
            input_mask_img = TF.pad(input_mask_img, (0, 0, padw, padh), padding_mode='reflect')
            label_mask_img = TF.pad(label_mask_img, (0, 0, padw, padh), padding_mode='reflect')
            arr=TF.pad(arr, (0, 0, padw, padh), padding_mode='reflect')
            arr_forloss = TF.pad(arr_forloss, (0, 0, padw, padh), padding_mode='reflect')
        # torchvision.utils.save_image(label_img,'cv2.png')
        return (input_img, label_img, input_mask_img, label_mask_img,arr,arr_forloss)

class Dataset_train_memory(Dataset):
    def __init__(self, input_root, label_root, fis=256, use_mixup=False):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.full_img_size = fis
        self.use_mixup = use_mixup
        self.input_imgs=[]
        self.label_imgs = []
        for i in self.input_files:
            path=os.path.join(self.input_root, i)
            self.input_imgs.append(io.imread(path))
        print('input done')
        for i in self.input_files:
            path = os.path.join(self.label_root, i)
            self.label_imgs.append(io.imread(path))
        print('gt done')
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img=self.input_imgs[index]
        label_img = self.label_imgs[index]
        # input_img_path = os.path.join(self.input_root, self.input_files[index])
        # input_img = io.imread(input_img_path)
        # label_img_path = os.path.join(self.label_root, self.label_files[index])
        # label_img = io.imread(label_img_path)
        if self.use_mixup:
            idx_add = torch.randint(0, self.__len__(), size=[1])
            lam = np.random.beta(1, 1)

            input_img = lam * input_img + (1 - lam) * io.imread(
                os.path.join(self.input_root, self.input_files[idx_add]))
            label_img = lam * label_img + (1 - lam) * io.imread(
                os.path.join(self.label_root, self.label_files[idx_add]))

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]

        mode = random.randint(0, 7)
        input_img = augment_img(input_img, mode=mode)
        label_img = augment_img(label_img, mode=mode)

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis = self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
        # torchvision.utils.save_image(label_img,'cv2.png')
        return (input_img, label_img)

class Dataset_train_map_array_sort(Dataset):
    def __init__(self, input_root, label_root, input_map_root,label_map_root,array_root,fis=256,use_mixup=False,region=50):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()
        self.array_root=array_root
        self.array_files=os.listdir(self.array_root)
        self.array_files.sort()

        self.input_map_root=input_map_root
        self.label_map_root=label_map_root

        self.full_img_size=fis
        self.use_mixup=use_mixup
        self.region=region
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        input_mask_img=io.imread(os.path.join(self.input_map_root, self.input_files[index]))


        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        label_mask_img = io.imread(os.path.join(self.label_map_root, self.input_files[index]))

        arr_path=os.path.join(self.array_root, self.array_files[index])
        arr=np.load(arr_path)[:,:,0]


        if self.use_mixup:
            idx_add=torch.randint(0,self.__len__(),size=[1])
            lam = np.random.beta(1,1)

            input_img = lam * input_img + (1-lam) * io.imread(os.path.join(self.input_root, self.input_files[idx_add]))
            label_img = lam * label_img + (1 - lam) * io.imread(os.path.join(self.label_root, self.label_files[idx_add]))

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        input_mask_img = input_mask_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_mask_img = label_mask_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        arr=arr[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size]

        mode = random.randint(0, 7)
        input_img = augment_img(input_img, mode=mode)
        label_img = augment_img(label_img, mode=mode)
        input_mask_img = augment_img(input_mask_img, mode=mode)
        label_mask_img = augment_img(label_mask_img, mode=mode)
        arr=augment_img(arr, mode=mode)

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        input_mask_img = uint2tensor3(input_mask_img)
        label_mask_img = uint2tensor3(label_mask_img)
        arr=torch.from_numpy(np.ascontiguousarray(arr).copy()).to(torch.int64)
        # arr[arr>=self.region]=self.region
        # arr=F.one_hot(arr, num_classes = self.region+1).permute(2,0,1)
        # arr=arr[:-1]

        arr[arr >= self.region] = self.region-1
        arr_forloss=arr.unsqueeze(0)
        arr = F.one_hot(arr, num_classes=self.region).permute(2, 0, 1)
        arr = arr[torch.sort(torch.sum(arr, dim=[1, 2]))[1]]
        # arr = arr[:-1]

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis = self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
            input_mask_img = TF.pad(input_mask_img, (0, 0, padw, padh), padding_mode='reflect')
            label_mask_img = TF.pad(label_mask_img, (0, 0, padw, padh), padding_mode='reflect')
            arr=TF.pad(arr, (0, 0, padw, padh), padding_mode='reflect')
            arr_forloss = TF.pad(arr_forloss, (0, 0, padw, padh), padding_mode='reflect')
        # torchvision.utils.save_image(label_img,'cv2.png')
        return (input_img, label_img, input_mask_img, label_mask_img,arr,arr_forloss)

class Dataset_test_map_array_sort(Dataset):
    def __init__(self, input_root, label_root,input_map_root,label_map_root,arr_root,region=50):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()

        self.input_map_root = input_map_root
        self.input_map_files = os.listdir(input_map_root)
        self.input_map_files.sort()
        self.label_map_root = label_map_root
        self.label_map_files = os.listdir(label_map_root)
        self.label_map_files.sort()

        self.arr_root=arr_root
        self.arr_files=os.listdir(self.arr_root)
        self.arr_files.sort()

        # self.train=train
        self.transforms = tr.Compose([tr.ToTensor()])
        self.region=region
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        input_map_path = os.path.join(self.input_map_root, self.input_map_files[index])
        input_map = io.imread(input_map_path)
        label_map_path = os.path.join(self.label_map_root, self.label_map_files[index])
        label_map = io.imread(label_map_path)
        arr_pth=os.path.join(self.arr_root, self.arr_files[index])
        arr=np.load(arr_pth)[:,:,0]


        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        input_map = uint2tensor3(input_map)
        label_map = uint2tensor3(label_map)
        arr = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.int64)
        # arr[arr >= self.region] = 0
        # arr = F.one_hot(arr, num_classes=self.region).permute(2, 0, 1)
        # arr[arr >= self.region] = self.region
        # arr = F.one_hot(arr, num_classes=self.region + 1).permute(2, 0, 1)
        # arr = arr[:-1]

        arr[arr >= self.region] = self.region-1
        arr = F.one_hot(arr, num_classes=self.region).permute(2, 0, 1)
        arr = arr[torch.sort(torch.sum(arr, dim=[1, 2]))[1]]
        # torchvision.utils.save_image(label_img,'gt.png')
        # torchvision.utils.save_image(input_img, 'inpt.png')
        # return (input_img, label_img, self.files[index])
        return (input_img,label_img,input_map,label_map,arr,self.input_files[index])

class Dataset_train_map_array_free_msk(Dataset):
    def __init__(self, input_root, label_root, input_map_root,label_map_root,array_root,fis=256,use_mixup=False,region=50):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()
        self.array_root=array_root
        self.array_files=os.listdir(self.array_root)
        self.array_files.sort()

        self.input_map_root=input_map_root
        self.label_map_root=label_map_root

        self.full_img_size=fis
        self.use_mixup=use_mixup
        self.region=region
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        input_mask_img=io.imread(os.path.join(self.input_map_root, self.input_files[index]))


        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        label_mask_img = io.imread(os.path.join(self.label_map_root, self.input_files[index]))

        arr_path=os.path.join(self.array_root, self.array_files[index])
        arr=np.load(arr_path)[:,:,0]


        if self.use_mixup:
            idx_add=torch.randint(0,self.__len__(),size=[1])
            lam = np.random.beta(1,1)

            input_img = lam * input_img + (1-lam) * io.imread(os.path.join(self.input_root, self.input_files[idx_add]))
            label_img = lam * label_img + (1 - lam) * io.imread(os.path.join(self.label_root, self.label_files[idx_add]))

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        input_mask_img = input_mask_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_mask_img = label_mask_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        arr=arr[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size]

        mode = random.randint(0, 7)
        input_img = augment_img(input_img, mode=mode)
        label_img = augment_img(label_img, mode=mode)
        input_mask_img = augment_img(input_mask_img, mode=mode)
        label_mask_img = augment_img(label_mask_img, mode=mode)
        arr=augment_img(arr, mode=mode)

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        input_mask_img = uint2tensor3(input_mask_img)
        label_mask_img = uint2tensor3(label_mask_img)
        arr=torch.from_numpy(np.ascontiguousarray(arr).copy()).to(torch.int64)
        # arr[arr>=self.region]=self.region
        # arr=F.one_hot(arr, num_classes = self.region+1).permute(2,0,1)
        # arr=arr[:-1]

        arr[arr >= self.region] = self.region-1
        # arr_forloss=arr.unsqueeze(0)
        arr = F.one_hot(arr, num_classes=self.region).permute(2, 0, 1)
        # arr = arr[:-1]

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis = self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
            input_mask_img = TF.pad(input_mask_img, (0, 0, padw, padh), padding_mode='reflect')
            label_mask_img = TF.pad(label_mask_img, (0, 0, padw, padh), padding_mode='reflect')
            arr=TF.pad(arr, (0, 0, padw, padh), padding_mode='reflect')
            # arr_forloss = TF.pad(arr_forloss, (0, 0, padw, padh), padding_mode='reflect')
        # torchvision.utils.save_image(label_img,'cv2.png')
        masks_p = torch.sum(arr, dim=[1, 2]) / (arr.shape[1] * arr.shape[2])
        free_msk_num = torch.multinomial(masks_p, 1)
        free_msk=arr[free_msk_num]

        return (input_img, label_img, input_mask_img, label_mask_img,arr,free_msk)

class Dataset_train_noise(Dataset):
    def __init__(self, input_root, label_root, fis=256,use_mixup=False):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.full_img_size=fis
        self.use_mixup=use_mixup
    def __len__(self):
        return len(self.input_files)
    def gasuss_noise(self,image, mean, var,shape):
        noise = np.random.normal(mean, var, [shape[0], shape[1], 1])
        out = image + np.tile(noise, 3)
        return np.clip(out, 0, 255).astype(np.uint8)
    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        if self.use_mixup:
            idx_add=torch.randint(0,self.__len__(),size=[1])
            lam = np.random.beta(1,1)

            input_img = lam * input_img + (1-lam) * io.imread(os.path.join(self.input_root, self.input_files[idx_add]))
            label_img = lam * label_img + (1 - lam) * io.imread(os.path.join(self.label_root, self.label_files[idx_add]))

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]

        mode = random.randint(0, 7)
        input_img = augment_img(input_img, mode=mode)
        label_img = augment_img(label_img, mode=mode)
        input_img = self.gasuss_noise(input_img, 0, 16, [input_img.shape[0], input_img.shape[1]])
        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis = self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
        # torchvision.utils.save_image(label_img,'cv2.png')
        return (input_img, label_img)

class Dataset_train_syc(Dataset):
    def __init__(self, input_root, label_root, fis=256,use_mixup=False):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.full_img_size=fis
        self.use_mixup=use_mixup
    def __len__(self):
        return len(self.input_files)

    def gasuss_noise(self,image, mean, var,shape):
        noise = np.random.normal(mean, var, [shape[0], shape[1], 1])
        out = image + np.tile(noise, 3)
        return np.clip(out, 0, 255).astype(np.uint8)
    def __getitem__(self, index):
        # input_img_path = os.path.join(self.input_root, self.input_files[index])
        # input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        if self.use_mixup:
            idx_add=torch.randint(0,self.__len__(),size=[1])
            lam = np.random.beta(1,1)

            # input_img = lam * input_img + (1-lam) * io.imread(os.path.join(self.input_root, self.input_files[idx_add]))
            label_img = lam * label_img + (1 - lam) * io.imread(os.path.join(self.label_root, self.label_files[idx_add]))

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        # input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]

        mode = random.randint(0, 7)
        # input_img = augment_img(input_img, mode=mode)
        label_img = augment_img(label_img, mode=mode)

        input_img=self.gasuss_noise(label_img,0, 16,[label_img.shape[0],label_img.shape[1]])
        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis = self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
        # torchvision.utils.save_image(label_img,'cv2.png')
        return (input_img, label_img)
# class Dataset_train_usm(Dataset):
#     def __init__(self, input_root, label_root, fis=256,use_mixup=False):
#         self.input_root = input_root
#         self.input_files = os.listdir(input_root)
#         self.label_root = label_root
#         self.label_files = os.listdir(label_root)
#         self.full_img_size=fis
#         self.use_mixup=use_mixup
#     def __len__(self):
#         return len(self.input_files)
#
#     def __getitem__(self, index):
#         input_img_path = os.path.join(self.input_root, self.input_files[index])
#         input_img = io.imread(input_img_path)
#         label_img_path = os.path.join(self.label_root, self.label_files[index])
#         label_img = io.imread(label_img_path)
#         if self.use_mixup:
#             idx_add=torch.randint(0,self.__len__(),size=[1])
#             lam = np.random.beta(1,1)
#
#             input_img = lam * input_img + (1-lam) * io.imread(os.path.join(self.input_root, self.input_files[idx_add]))
#             label_img = lam * label_img + (1 - lam) * io.imread(os.path.join(self.label_root, self.label_files[idx_add]))
#
#         H, W, _ = label_img.shape
#         rnd_h = random.randint(0, max(0, H - self.full_img_size))
#         rnd_w = random.randint(0, max(0, W - self.full_img_size))
#         input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
#         label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
#
#         #usm
#         # alpha=0.5
#         # label_img=label_img / 255.0
#         # gauss_out = cv2.GaussianBlur(label_img,(0,0),5)
#         # label_img=(label_img - gauss_out)*alpha + label_img
#         # label_img=label_img*255.
#
#         label_img=usm_sharp(label_img)
#
#         mode = random.randint(0, 7)
#         input_img = augment_img(input_img, mode=mode)
#         label_img = augment_img(label_img, mode=mode)
#
#         input_img = uint2tensor3(input_img)
#         label_img = uint2tensor3(label_img)
#
#         w = label_img.shape[-2]
#         h = label_img.shape[-1]
#         fis = self.full_img_size
#         padw = fis - w if w < fis else 0
#         padh = fis - h if h < fis else 0
#
#         # Reflect Pad in case image is smaller than patch_size
#         if padw != 0 or padh != 0:
#             input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
#             label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
#         # torchvision.utils.save_image(label_img,'cv2.png')
#         return (input_img, label_img)

class Dataset_train_enhance(Dataset):
    def __init__(self, input_root, label_root, fis=256,use_mixup=False):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.full_img_size=fis
        self.use_mixup=use_mixup
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        label_img = label_img / 255.
        label_img = filters.gaussian(label_img, sigma=0.7)
        label_img=np.clip(label_img,a_max=1,a_min=0)
        # label_img = exposure.equalize_adapthist(label_img)
        if self.use_mixup:
            idx_add=torch.randint(0,self.__len__(),size=[1])
            lam = np.random.beta(1,1)

            input_img = lam * input_img + (1-lam) * io.imread(os.path.join(self.input_root, self.input_files[idx_add]))
            label_img = lam * label_img + (1 - lam) * io.imread(os.path.join(self.label_root, self.label_files[idx_add]))

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]

        mode = random.randint(0, 7)
        input_img = augment_img(input_img, mode=mode)
        label_img = augment_img(label_img, mode=mode)

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3_nodiv(label_img)

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis = self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
        # torchvision.utils.save_image(label_img,'cv2.png')
        return (input_img, label_img)

class Dataset_train_cv2_mit5k(Dataset):
    def __init__(self, input_root, label_root, files,fis=256):
        self.input_root = input_root
        # self.input_files = os.listdir(input_root)
        self.label_root = label_root
        self.files = []
        with open(files) as f:
            line = f.readline().split('\n')[0]
            while line:
                self.files.append(line)
                line = f.readline().split('\n')[0]
        # self.label_files = os.listdir(label_root)
        self.full_img_size = fis

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.files[index])
        label_img = io.imread(label_img_path)

        H, W, _ = label_img.shape
        rnd_h = random.randint(0, max(0, H - self.full_img_size))
        rnd_w = random.randint(0, max(0, W - self.full_img_size))
        input_img = input_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]
        label_img = label_img[rnd_h:rnd_h + self.full_img_size, rnd_w:rnd_w + self.full_img_size, :]

        mode = random.randint(0, 7)
        input_img = augment_img(input_img, mode=mode)
        label_img = augment_img(label_img, mode=mode)

        input_img=uint2tensor3(input_img)
        label_img=uint2tensor3(label_img)

        w = label_img.shape[-2]
        h = label_img.shape[-1]
        fis=self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
        # torchvision.utils.save_image(label_img,'cv2.png')
        return (input_img,label_img)

class Dataset_train_mit5k(Dataset):
    def __init__(self, input_root, label_root,files ,fis=256):
        self.input_root = input_root
        # self.input_files = os.listdir(input_root)
        self.label_root = label_root
        self.files =[]
        with open(files) as f:
            line = f.readline().split('\n')[0]
            while line:
                self.files.append(line)
                line = f.readline().split('\n')[0]
        # self.label_files = os.listdir(label_root)
        self.full_img_size=fis

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.files[index])
        input_img = Image.open(input_img_path)#.convert('RGB')
        # input_img = cv2.imread(input_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        label_img_path = os.path.join(self.label_root, self.files[index])
        label_img = Image.open(label_img_path)#.convert('RGB')
        # label_img = cv2.imread(label_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        w, h = label_img.size
        fis=self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')

        input_img = TF.to_tensor(input_img)
        label_img = TF.to_tensor(label_img)

        hh, ww = label_img.shape[1], label_img.shape[2]

        rr = random.randint(0, hh - fis)
        cc = random.randint(0, ww - fis)
        aug = random.randint(0, 8)

        # Crop patch
        input_img = input_img[:, rr:rr + fis, cc:cc + fis]
        label_img = label_img[:, rr:rr + fis, cc:cc + fis]

        # Data Augmentations
        if aug == 1:
            input_img = input_img.flip(1)
            label_img = label_img.flip(1)
        elif aug == 2:
            input_img = input_img.flip(2)
            label_img = label_img.flip(2)
        elif aug == 3:
            input_img = torch.rot90(input_img, dims=(1, 2))
            label_img = torch.rot90(label_img, dims=(1, 2))
        elif aug == 4:
            input_img = torch.rot90(input_img, dims=(1, 2), k=2)
            label_img = torch.rot90(label_img, dims=(1, 2), k=2)
        elif aug == 5:
            input_img = torch.rot90(input_img, dims=(1, 2), k=3)
            label_img = torch.rot90(label_img, dims=(1, 2), k=3)
        elif aug == 6:
            input_img = torch.rot90(input_img.flip(1), dims=(1, 2))
            label_img = torch.rot90(label_img.flip(1), dims=(1, 2))
        elif aug == 7:
            input_img = torch.rot90(input_img.flip(2), dims=(1, 2))
            label_img = torch.rot90(label_img.flip(2), dims=(1, 2))


        return (input_img,label_img)

class Dataset_train_extend(Dataset):
    def __init__(self, input_root, label_root, fis=256):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.full_img_size=fis

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = Image.open(input_img_path)#.convert('RGB')
        # input_img = cv2.imread(input_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = Image.open(label_img_path)#.convert('RGB')
        # label_img = cv2.imread(label_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        # w, h = label_img.size
        fis=self.full_img_size
        # padw = fis - w if w < fis else 0
        # padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        # if padw != 0 or padh != 0:
        #     input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
        #     label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')

        input_img = TF.to_tensor(input_img)
        label_img = TF.to_tensor(label_img)

        hh, ww = label_img.shape[1], label_img.shape[2]

        rr = random.randint(0, hh - fis)
        cc = random.randint(0, ww - fis)
        aug = random.randint(0, 8)

        # Crop patch
        input_img = input_img[:, rr:rr + fis, cc:cc + fis]
        label_img = label_img[:, rr:rr + fis, cc:cc + fis]

        # Data Augmentations
        if aug == 1:
            input_img = input_img.flip(1)
            label_img = label_img.flip(1)
        elif aug == 2:
            input_img = input_img.flip(2)
            label_img = label_img.flip(2)
        elif aug == 3:
            input_img = torch.rot90(input_img, dims=(1, 2))
            label_img = torch.rot90(label_img, dims=(1, 2))
        elif aug == 4:
            input_img = torch.rot90(input_img, dims=(1, 2), k=2)
            label_img = torch.rot90(label_img, dims=(1, 2), k=2)
        elif aug == 5:
            input_img = torch.rot90(input_img, dims=(1, 2), k=3)
            label_img = torch.rot90(label_img, dims=(1, 2), k=3)
        elif aug == 6:
            input_img = torch.rot90(input_img.flip(1), dims=(1, 2))
            label_img = torch.rot90(label_img.flip(1), dims=(1, 2))
        elif aug == 7:
            input_img = torch.rot90(input_img.flip(2), dims=(1, 2))
            label_img = torch.rot90(label_img.flip(2), dims=(1, 2))


        return (input_img,label_img)

class Dataset_train_velol(Dataset):
    def __init__(self, input_root, label_root, fis=256):
        self.input_root = input_root
        self.files = os.listdir(input_root)
        self.label_root = label_root
        # self.label_files = os.listdir(label_root)
        self.full_img_size=fis

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.files[index])
        input_img = Image.open(input_img_path).convert('RGB')
        # input_img =cv2.imread(input_img_path,cv2.IMREAD_COLOR).astype(np.float32) / 255.
        if self.files[index].startswith('low'):
            high_file=self.files[index].replace('low','normal')
        else:
            high_file = self.files[index]
        label_img_path = os.path.join(self.label_root, high_file)
        label_img =Image.open(label_img_path).convert('RGB')
        # label_img = cv2.imread(label_img_path,cv2.IMREAD_COLOR).astype(np.float32) / 255.
        # w, h = label_img.size
        fis=self.full_img_size
        # padw = fis - w if w < fis else 0
        # padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        # if padw != 0 or padh != 0:
        #     input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
        #     label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')

        input_img = TF.to_tensor(input_img)
        label_img = TF.to_tensor(label_img)

        hh, ww = label_img.shape[1], label_img.shape[2]

        rr = random.randint(0, hh - fis)
        cc = random.randint(0, ww - fis)
        aug = random.randint(0, 8)

        # Crop patch
        input_img = input_img[:, rr:rr + fis, cc:cc + fis]
        label_img = label_img[:, rr:rr + fis, cc:cc + fis]

        # Data Augmentations
        if aug == 1:
            input_img = input_img.flip(1)
            label_img = label_img.flip(1)
        elif aug == 2:
            input_img = input_img.flip(2)
            label_img = label_img.flip(2)
        elif aug == 3:
            input_img = torch.rot90(input_img, dims=(1, 2))
            label_img = torch.rot90(label_img, dims=(1, 2))
        elif aug == 4:
            input_img = torch.rot90(input_img, dims=(1, 2), k=2)
            label_img = torch.rot90(label_img, dims=(1, 2), k=2)
        elif aug == 5:
            input_img = torch.rot90(input_img, dims=(1, 2), k=3)
            label_img = torch.rot90(label_img, dims=(1, 2), k=3)
        elif aug == 6:
            input_img = torch.rot90(input_img.flip(1), dims=(1, 2))
            label_img = torch.rot90(label_img.flip(1), dims=(1, 2))
        elif aug == 7:
            input_img = torch.rot90(input_img.flip(2), dims=(1, 2))
            label_img = torch.rot90(label_img.flip(2), dims=(1, 2))


        return (input_img,label_img)

class Dataset_train_o(Dataset):
    def __init__(self, input_root, label_root,noise_root, fis=256):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.noise_root=noise_root
        self.noise_files = os.listdir(noise_root)
        self.full_img_size=fis

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = Image.open(input_img_path).convert('RGB')

        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = Image.open(label_img_path).convert('RGB')
        noise_img_path = os.path.join(self.noise_root, self.label_files[index])
        noise_img = Image.open(noise_img_path).convert('RGB')

        w, h = label_img.size
        fis=self.full_img_size
        padw = fis - w if w < fis else 0
        padh = fis - h if h < fis else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            input_img = TF.pad(input_img, (0, 0, padw, padh), padding_mode='reflect')
            label_img = TF.pad(label_img, (0, 0, padw, padh), padding_mode='reflect')
            noise_img = TF.pad(noise_img, (0, 0, padw, padh), padding_mode='reflect')

        input_img = TF.to_tensor(input_img)
        label_img = TF.to_tensor(label_img)
        noise_img = TF.to_tensor(noise_img)

        hh, ww = label_img.shape[1], label_img.shape[2]

        rr = random.randint(0, hh - fis)
        cc = random.randint(0, ww - fis)
        aug = random.randint(0, 8)

        # Crop patch
        input_img = input_img[:, rr:rr + fis, cc:cc + fis]
        label_img = label_img[:, rr:rr + fis, cc:cc + fis]
        noise_img = noise_img[:, rr:rr + fis, cc:cc + fis]
        # Data Augmentations
        if aug == 1:
            input_img = input_img.flip(1)
            label_img = label_img.flip(1)
            noise_img = noise_img.flip(1)
        elif aug == 2:
            input_img = input_img.flip(2)
            label_img = label_img.flip(2)
            noise_img = noise_img.flip(2)
        elif aug == 3:
            input_img = torch.rot90(input_img, dims=(1, 2))
            label_img = torch.rot90(label_img, dims=(1, 2))
            noise_img = torch.rot90(noise_img, dims=(1, 2))
        elif aug == 4:
            input_img = torch.rot90(input_img, dims=(1, 2), k=2)
            label_img = torch.rot90(label_img, dims=(1, 2), k=2)
            noise_img = torch.rot90(noise_img, dims=(1, 2),k=2)
        elif aug == 5:
            input_img = torch.rot90(input_img, dims=(1, 2), k=3)
            label_img = torch.rot90(label_img, dims=(1, 2), k=3)
            noise_img = torch.rot90(noise_img, dims=(1, 2),k=3)
        elif aug == 6:
            input_img = torch.rot90(input_img.flip(1), dims=(1, 2))
            label_img = torch.rot90(label_img.flip(1), dims=(1, 2))
            noise_img = torch.rot90(noise_img.flip(1), dims=(1, 2))
        elif aug == 7:
            input_img = torch.rot90(input_img.flip(2), dims=(1, 2))
            label_img = torch.rot90(label_img.flip(2), dims=(1, 2))
            noise_img = torch.rot90(noise_img.flip(2), dims=(1, 2))

        return (input_img,label_img,noise_img)

class Dataset_val(Dataset):
    def __init__(self, input_root, label_root):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.transforms = tr.Compose([tr.CenterCrop(256),tr.ToTensor()])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = Image.open(input_img_path).convert('RGB')

        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = Image.open(label_img_path).convert('RGB')
        input_img=self.transforms(input_img)
        label_img = self.transforms(label_img)

        return (input_img,label_img)

class Dataset_test(Dataset):
    def __init__(self, input_root, label_root):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()
        # self.train=train
        self.transforms = tr.Compose([tr.ToTensor()])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)

        return (input_img,label_img,self.input_files[index])

class Dataset_test_baid(Dataset):
    def __init__(self, input_root, label_root,mask_root):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()
        self.mask_root = mask_root
        self.mask_files = os.listdir(mask_root)
        self.mask_files.sort()
        # self.train=train
        self.transforms = tr.Compose([tr.ToTensor()])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        mask_img_path = os.path.join(self.mask_root, self.mask_files[index])
        mask_img = io.imread(mask_img_path)[:,:,np.newaxis]

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        mask_img = uint2tensor3(mask_img)

        return (input_img,label_img,mask_img,self.input_files[index])

class Dataset_test_clediff(Dataset):
    def __init__(self, input_root, label_root,mask_root):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()
        self.mask_root = mask_root
        self.mask_files = os.listdir(mask_root)
        self.mask_files.sort()
        # self.train=train
        self.transforms = tr.Compose([tr.ToTensor()])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        mask_img_path = os.path.join(self.mask_root, self.mask_files[index])
        mask_img = io.imread(mask_img_path)#[:,:,np.newaxis]

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        mask_img = uint2tensor3(mask_img)[:1]

        return (input_img,label_img,mask_img,self.input_files[index])

class Dataset_test_baid_nomask(Dataset):
    def __init__(self, input_root, label_root,mask_root):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()
        # self.mask_root = mask_root
        # self.mask_files = os.listdir(mask_root)
        # self.mask_files.sort()
        # self.train=train
        self.transforms = tr.Compose([tr.ToTensor()])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        # mask_img_path = os.path.join(self.mask_root, self.mask_files[index])
        # mask_img = io.imread(mask_img_path)[:,:,np.newaxis]

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        # mask_img = uint2tensor3(mask_img)

        return (input_img,label_img,self.input_files[index])


class Dataset_test_map(Dataset):
    def __init__(self, input_root, label_root,input_map_root,label_map_root,arr_root):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()

        self.input_map_root = input_map_root
        self.input_map_files = os.listdir(input_map_root)
        self.input_map_files.sort()
        self.label_map_root = label_map_root
        self.label_map_files = os.listdir(label_map_root)
        self.label_map_files.sort()

        self.arr_root=arr_root
        self.arr_files=os.listdir(self.arr_root)
        self.arr_files.sort()

        # self.train=train
        self.transforms = tr.Compose([tr.ToTensor()])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        input_map_path = os.path.join(self.input_map_root, self.input_map_files[index])
        input_map = io.imread(input_map_path)
        label_map_path = os.path.join(self.label_map_root, self.label_map_files[index])
        label_map = io.imread(label_map_path)
        arr_pth=os.path.join(self.arr_root, self.arr_files[index])
        arr=np.load(arr_pth)


        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        input_map = uint2tensor3(input_map)
        label_map = uint2tensor3(label_map)

        # torchvision.utils.save_image(label_img,'gt.png')
        # torchvision.utils.save_image(input_img, 'inpt.png')
        # return (input_img, label_img, self.files[index])
        return (input_img,label_img,input_map,label_map,self.input_files[index])

class Dataset_test_map_array(Dataset):
    def __init__(self, input_root, label_root,input_map_root,label_map_root,arr_root,region=50):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()

        self.input_map_root = input_map_root
        self.input_map_files = os.listdir(input_map_root)
        self.input_map_files.sort()
        self.label_map_root = label_map_root
        self.label_map_files = os.listdir(label_map_root)
        self.label_map_files.sort()

        self.arr_root=arr_root
        self.arr_files=os.listdir(self.arr_root)
        self.arr_files.sort()

        # self.train=train
        self.transforms = tr.Compose([tr.ToTensor()])
        self.region=region
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        input_map_path = os.path.join(self.input_map_root, self.input_map_files[index])
        input_map = io.imread(input_map_path)
        label_map_path = os.path.join(self.label_map_root, self.label_map_files[index])
        label_map = io.imread(label_map_path)
        arr_pth=os.path.join(self.arr_root, self.arr_files[index])
        arr=np.load(arr_pth)[:,:,0]


        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        input_map = uint2tensor3(input_map)
        label_map = uint2tensor3(label_map)
        arr = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.int64)
        # arr[arr >= self.region] = 0
        # arr = F.one_hot(arr, num_classes=self.region).permute(2, 0, 1)
        # arr[arr >= self.region] = self.region
        # arr = F.one_hot(arr, num_classes=self.region + 1).permute(2, 0, 1)
        # arr = arr[:-1]

        arr[arr >= self.region] = self.region-1
        arr = F.one_hot(arr, num_classes=self.region).permute(2, 0, 1)

        # torchvision.utils.save_image(label_img,'gt.png')
        # torchvision.utils.save_image(input_img, 'inpt.png')
        # return (input_img, label_img, self.files[index])
        return (input_img,label_img,input_map,label_map,arr,self.input_files[index])

class Dataset_infer(Dataset):
    def __init__(self, input_root):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        # self.label_root = label_root
        # self.label_files = os.listdir(label_root)
        # self.label_files.sort()
        # self.train=train
        self.transforms = tr.Compose([tr.ToTensor()])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        # label_img_path = os.path.join(self.label_root, self.label_files[index])
        # label_img = io.imread(label_img_path)

        input_img = uint2tensor3(input_img)
        # label_img = uint2tensor3(label_img)
        # torchvision.utils.save_image(label_img,'gt.png')
        # torchvision.utils.save_image(input_img, 'inpt.png')
        # return (input_img, label_img, self.files[index])
        return (input_img,self.input_files[index])

class Dataset_motivation(Dataset):
    def __init__(self, input_root, label_root,ori_root,gt_root):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        self.label_files.sort()
        self.ori_root = ori_root
        self.ori_files = os.listdir(ori_root)
        self.ori_files.sort()
        self.gt_root = gt_root
        self.gt_files = os.listdir(gt_root)
        self.gt_files.sort()
        self.transforms = tr.Compose([tr.ToTensor()])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        # input_img=np.asarray(Image.open(os.path.join(self.input_root, self.input_files[index])), dtype=np.uint8)[..., :3].astype(np.float32)
        # input_img = input_img.transpose((2, 0, 1))
        # input_img=  input_img / 255
        # input_img = torch.from_numpy(input_img).type(torch.FloatTensor)
        #
        # label_img = np.asarray(Image.open(os.path.join(self.label_root, self.label_files[index])), dtype=np.uint8)[...,
        #             :3].astype(np.float32)
        # label_img = label_img.transpose((2, 0, 1))
        # label_img = label_img /255
        # label_img = torch.from_numpy(label_img).type(torch.FloatTensor)
        #
        # ori_img = np.asarray(Image.open(os.path.join(self.ori_root, self.ori_files[index])), dtype=np.uint8)[...,
        #             :3].astype(np.float32)
        # ori_img = ori_img.transpose((2, 0, 1))
        # ori_img = ori_img /255
        # ori_img = torch.from_numpy(ori_img).type(torch.FloatTensor)
        #
        # gt_img = np.asarray(Image.open(os.path.join(self.gt_root, self.gt_files[index])), dtype=np.uint8)[...,
        #             :3].astype(np.float32)
        # gt_img = gt_img.transpose((2, 0, 1))
        # gt_img = gt_img /255
        # gt_img = torch.from_numpy(gt_img).type(torch.FloatTensor)
        
        
        
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        ori_img_path = os.path.join(self.ori_root, self.ori_files[index])
        ori_img = io.imread(ori_img_path)
        gt_img_path = os.path.join(self.gt_root, self.gt_files[index])
        gt_img = io.imread(gt_img_path)
        # 
        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        ori_img = uint2tensor3(ori_img)
        gt_img=uint2tensor3(gt_img)
        # torchvision.utils.save_image(label_img,'gt.png')
        # torchvision.utils.save_image(input_img, 'inpt.png')
        # return (input_img, label_img, self.files[index])
        return (input_img,label_img,ori_img,gt_img,self.input_files[index])

class Dataset_test_unpair(Dataset):
    def __init__(self, input_root):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.input_files.sort()
        # self.label_root = label_root
        # self.label_files = os.listdir(label_root)
        # self.label_files.sort()
        # self.train=train
        self.transforms = tr.Compose([tr.ToTensor()])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        # label_img_path = os.path.join(self.label_root, self.label_files[index])
        # label_img = io.imread(label_img_path)

        input_img = uint2tensor3(input_img)
        # label_img = uint2tensor3(label_img)
        # torchvision.utils.save_image(label_img,'gt.png')
        # torchvision.utils.save_image(input_img, 'inpt.png')
        # return (input_img, label_img, self.files[index])
        return (input_img,self.input_files[index])

class Dataset_test_syc(Dataset):
    def __init__(self, input_root, label_root):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        # self.train=train
        self.transforms = tr.Compose([tr.ToTensor()])

    def __len__(self):
        return len(self.input_files)
    def gasuss_noise(self,image, mean, var,shape):
        noise = np.random.normal(mean, var, [shape[0], shape[1], 1])
        out = image + np.tile(noise, 3)
        return np.clip(out, 0, 255).astype(np.uint8)
    def __getitem__(self, index):
        # input_img_path = os.path.join(self.input_root, self.input_files[index])
        # input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        input_img = self.gasuss_noise(label_img, 0, 16, [label_img.shape[0], label_img.shape[1]])
        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        # torchvision.utils.save_image(label_img,'gt.png')
        # torchvision.utils.save_image(input_img, 'inpt.png')
        # return (input_img, label_img, self.files[index])
        return (input_img,label_img,self.input_files[index])

class Dataset_test_usm(Dataset):
    def __init__(self, input_root, label_root):
        self.input_root = input_root
        self.input_files = os.listdir(input_root)
        self.label_root = label_root
        self.label_files = os.listdir(label_root)
        # self.train=train
        self.transforms = tr.Compose([tr.ToTensor()])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = io.imread(label_img_path)
        label_img=usm_sharp(label_img)
        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        # torchvision.utils.save_image(label_img,'gt.png')
        # torchvision.utils.save_image(input_img, 'inpt.png')
        # return (input_img, label_img, self.files[index])
        return (input_img,label_img,self.input_files[index])

class Dataset_test_mit5k(Dataset):
    def __init__(self, input_root, label_root,files ,train=True):
        self.input_root = input_root
        # self.input_files = os.listdir(input_root)
        self.label_root = label_root
        # self.label_files = os.listdir(label_root)
        self.files = []
        with open(files) as f:
            line = f.readline().split('\n')[0]
            while line:
                self.files.append(line)
                line = f.readline().split('\n')[0]
        self.train=train
        self.transforms = tr.Compose([tr.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.files[index])
        input_img = Image.open(input_img_path).convert('RGB')
        # input_img =cv2.imread(input_img_path,cv2.IMREAD_COLOR).astype(np.float32) / 255.

        label_img_path = os.path.join(self.label_root, self.files[index])
        label_img = Image.open(label_img_path).convert('RGB')
        # label_img = cv2.imread(label_img_path,cv2.IMREAD_COLOR).astype(np.float32) / 255.

        input_img=self.transforms(input_img)
        label_img = self.transforms(label_img)
        # torchvision.utils.save_image(label_img,'gt.png')
        # torchvision.utils.save_image(input_img, 'inpt.png')
        return (input_img,label_img,self.files[index])

class Dataset_test_cv2_mit5k(Dataset):
    def __init__(self, input_root, label_root,files ,train=True):
        self.input_root = input_root
        # self.input_files = os.listdir(input_root)
        self.label_root = label_root
        # self.label_files = os.listdir(label_root)
        self.files = []
        with open(files) as f:
            line = f.readline().split('\n')[0]
            while line:
                self.files.append(line)
                line = f.readline().split('\n')[0]
        self.train=train
        self.transforms = tr.Compose([tr.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        input_img_path = os.path.join(self.input_root, self.files[index])
        input_img = io.imread(input_img_path)
        label_img_path = os.path.join(self.label_root, self.files[index])
        label_img = io.imread(label_img_path)

        input_img = uint2tensor3(input_img)
        label_img = uint2tensor3(label_img)
        # torchvision.utils.save_image(label_img,'gt.png')
        # torchvision.utils.save_image(input_img, 'inpt.png')
        return (input_img,label_img,self.files[index])

