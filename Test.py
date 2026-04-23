import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import warnings
from utils import *
warnings.filterwarnings('ignore')
from torchvision import models
import csv
from models import Control_Lit
from uti.loss_fuc import L1_Charbonnier_loss
from dataset.load_data import Dataset_test_baid
import os
import argparse
torch.backends.cudnn.enabled = False
from torch.cuda.amp import autocast,GradScaler
from uti.loss_fuc import GANLoss
from utils import *
import contextlib # any random number
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  #
    torch.backends.cudnn.deterministic = True #


parser = argparse.ArgumentParser(description='')
parser.add_argument('--mode',default='Test')
parser.add_argument('--num_of_patch',default=5,type=int)
parser.add_argument('--random_seed',default=42,type=int)

parser.add_argument('--mgpu_in',default=True)
parser.add_argument('--mgpu_train',default=False)

parser.add_argument('--test_root',default='/data/wuhj/Dataset/BAID_clip_test/resize_input')
parser.add_argument('--test_gt_root',default='/data/wuhj/Dataset/BAID_clip_test/resize_gt')

#save setting
parser.add_argument('--psnr_st',default=25,help='psnr standard for saving model')
parser.add_argument('--val_img_size',default=256)
parser.add_argument('--saveimg_gap',default=1)
parser.add_argument('--test_img_num',default=368)
parser.add_argument('--test_freq',default=20)
parser.add_argument('--contine',default=True)
parser.add_argument('--contine_path',default="./ckpt/model.pth")
parser.add_argument('--warmup_ep',default=5)
parser.add_argument('--boundary',default=-1)

def test(t_loader,model,pth,args):
    print('testing......')
    model = model.eval()
    S=SSIM()
    os.makedirs(pth,exist_ok=True)

    with torch.no_grad():
        all_psnr=0
        all_ssim=0
        with tqdm(total=args.test_img_num) as tq:
            for idx,(data,gt,mask,name) in enumerate(t_loader):
                if idx>=args.test_img_num:
                    break
                data = data.cuda()
                # data_map=data_map.cuda()
                gt = gt.cuda()
                mask=mask.cuda()
                mask=F.interpolate(mask, size=[data.shape[-2],data.shape[-1]], mode='bicubic')
                _, C, W, H = data.shape
                data = check_image_size(16, data)
                gt= check_image_size(16, gt)
                mask = check_image_size(16, mask)
                # data_l, data_ab = torch.split(xyz2lab(rgb2xyz(data)), [1, 2], dim=1)
                # gt_l, gt_ab = torch.split(xyz2lab(rgb2xyz(gt)), [1, 2], dim=1)
                # data_l = data_l.repeat(1, 3, 1, 1)
                # gt_l = gt_l.repeat(1, 3, 1, 1)
                # out_put = torch.zeros_like(data).cuda()
                # flops, params = profile(model, (data,gt,mask))
                # # summary(model, [(3, 1368, 2048),(3, 1368, 2048),(1, 1368, 2048)])
                # print('flops:{}G,params:{}M'.format(flops / (1e9), params / (1000 ** 2)))
                # with amp_cm():
                pred=model(data,gt,mask)
                rgb_pred=pred
                rgb_pred = rgb_pred[:, :, :W, :H]
                gt=gt[:, :, :W, :H]
                if (idx+1)%args.saveimg_gap==0:
                #     # for i in range(1):
                    torchvision.utils.save_image(rgb_pred,pth + '/{}'.format(name[0].replace('JPG','png')))
                lf=nn.MSELoss()
                PSNR = lambda mse: 10 * torch.log10(1. / mse).item() if (mse > 1e-5) else 50
                psnr=PSNR(lf(gt,rgb_pred))
                ssim_=S(gt,rgb_pred)
                all_psnr+=psnr
                all_ssim+=ssim_
                tq.update()
        print(all_psnr/args.test_img_num,all_ssim/args.test_img_num)
        return all_psnr/args.test_img_num,all_ssim/args.test_img_num

if __name__=='__main__':

    args=parser.parse_args()
    set_seed(args.random_seed)

    dataset_test = Dataset_test_baid(args.test_root, args.test_gt_root,args.mask_test_root)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8,drop_last=False)

    model=Control_Lit(3,3,stage=2,depth=8,weight=1,n_e=256)
    if torch.cuda.is_available():
        model = model.cuda()
    if args.contine:
        params = torch.load(args.contine_path)
        if args.mgpu_in:
            if args.mgpu_train:

                new_dict1 = {k: v for k, v in params.items() }
            else:

                new_dict1 = {k[7:]: v for k, v in params.items() }
            result=model.load_state_dict(new_dict1,strict=False)

        else:
            if args.mgpu_train:

                new_dict1 = {'module.' + k: v for k, v in params.items() }
            else:
                new_dict1 = {k: v for k, v in params.items() }

            result = model.load_state_dict(new_dict1,strict=False)
        print(result)

    save_path="./{}".format(args.mode)
    model=model.train()
    test(test_loader, model,save_path, args)