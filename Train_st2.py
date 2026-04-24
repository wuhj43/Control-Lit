import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torch.utils.data import DataLoader

from tqdm import tqdm
import warnings
from utils import *
warnings.filterwarnings('ignore')
# from model_2 import *
from torchvision import models
import csv
from models import Control_Lit
from uti.loss_fuc import L1_Charbonnier_loss
import torch.nn.functional as F
from dataset.load_data import  Dataset_train_baid,Dataset_test_baid

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
parser.add_argument('--mode',default='Train_st2')
parser.add_argument('--num_of_patch',default=5,type=int)
parser.add_argument('--random_seed',default=42,type=int)
#training setting
parser.add_argument('--gpu',default='6',type=str)
parser.add_argument('--lr',default=5e-4,help='learning weight',type=float)
parser.add_argument('--batch_size',default=[32,16,8,4],type=int)
parser.add_argument('--epochs',default=10000,type=int)
parser.add_argument('--use_amp',default=False)
parser.add_argument('--mgpu_in',default=True)
parser.add_argument('--mgpu_train',default=False)
parser.add_argument('--input_root',default='/data/wuhj/Dataset/BAID_380/input')
parser.add_argument('--label_root',default='/data/wuhj/Dataset/BAID_380/gt')
parser.add_argument('--mask_train_root',default='/data/wuhj/Dataset/BAID_380/mask')


parser.add_argument('--test_root',default='/data/wuhj/Dataset/BAID_clip_test/resize_input')
parser.add_argument('--test_gt_root',default='/data/wuhj/Dataset/BAID_clip_test/resize_gt')
parser.add_argument('--mask_test_root',default='/data/wuhj/Dataset/BAID_clip_test/mask_test')


# swinir
parser.add_argument('--croped_img_size',default=64)
parser.add_argument('--img_size',default=[192,256,320,384])
parser.add_argument('--patch_size',default=2)
parser.add_argument('--upscale',default=1)
parser.add_argument('--window_size',default=4)
parser.add_argument('--ape',default=False)
parser.add_argument('--n_feats',default=72)
parser.add_argument('--mlp_ratio',default=2.)
parser.add_argument('--num_heads',default=6)
parser.add_argument('--drop_path',default=0.5)
parser.add_argument('--attn_drop',default=0.5)

#save setting
parser.add_argument('--psnr_st',default=25,help='psnr standard for saving model')
parser.add_argument('--val_img_size',default=256)
parser.add_argument('--saveimg_gap',default=50)
parser.add_argument('--test_img_num',default=368)
parser.add_argument('--test_freq',default=20)
parser.add_argument('--contine',default=False)
parser.add_argument('--contine_path',default="")
parser.add_argument('--warmup_ep',default=5)
parser.add_argument('--boundary',default=-635)


def get_vgg19():
    with torch.no_grad():
        vgg19=models.vgg19(pretrained=True)
        if torch.cuda.is_available():
            vgg19=vgg19.cuda()
        vgg19=vgg19.features
    for idx,param in enumerate(vgg19.parameters()):
        param.requires_grad = False

    vgg19_model_new = list(vgg19.children())[:17]
    vgg19 = nn.Sequential(*vgg19_model_new)
    return vgg19
def test(t_loader,model,ep,pth,max_psnr,args,amp_cm):
    print('testing......')
    model = model.eval()
    S=SSIM()


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
                data = F.interpolate(data, scale_factor=0.5, mode='bicubic')
                gt = F.interpolate(gt, scale_factor=0.5, mode='bicubic')
                mask=F.interpolate(mask, size=[data.shape[-2],data.shape[-1]], mode='bicubic')
                _, C, W, H = data.shape
                data = check_image_size(32, data)
                gt= check_image_size(32, gt)
                mask = check_image_size(32, mask)
                with amp_cm():
                    pred=model(data,gt,mask)

                    rgb_pred=pred

                    rgb_pred = rgb_pred[:, :, :W, :H]
                    gt=gt[:, :, :W, :H]
                    if (idx+1)%args.saveimg_gap==0:
                    #     # for i in range(1):
                        torchvision.utils.save_image(pred,pth + '/layer0_{}'.format(name[0]))
                    lf=nn.MSELoss()
                    PSNR = lambda mse: 10 * torch.log10(1. / mse).item() if (mse > 1e-5) else 50
                    psnr=PSNR(lf(gt,rgb_pred))
                    ssim_=S(gt,rgb_pred)
                    all_psnr+=psnr
                    all_ssim+=ssim_
                tq.update()
        return all_psnr/args.test_img_num,all_ssim/args.test_img_num

def train(args,test_loader,model,lf_1,lf_2,opt,scheduler1,scheduler2,save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    L1=nn.L1Loss()
    chan=L1_Charbonnier_loss()
    # tv_loss = TVLoss_l(0.01)
    vgg19=get_vgg19()
    max_psnr=0
    t=1
    S = SSIM()
    fftloss=FFTLoss().cuda()
    scaler = GradScaler()
    amp_cm = autocast if args.use_amp else contextlib.nullcontext
    headers = ['epoch', 'psnr','ssim']
    it = 0
    dataset_train = Dataset_train_baid(args.input_root, args.label_root, args.mask_train_root, fis=args.img_size[it],
                                       use_mixup=False)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size[it], shuffle=True,
                                               num_workers=8,
                                               pin_memory=True,
                                               drop_last=False)

    with open('./record/{}.csv'.format(args.mode), 'a', newline='') as f:
        record = csv.writer(f)
        record.writerow(headers)
    for ep in range(args.epochs):
        if (ep+1)%1000==0 and it<4:
            it+=1
            torch.cuda.empty_cache()
            dataset_train = Dataset_train_baid(args.input_root, args.label_root, args.mask_train_root, fis=args.img_size[it],
                                               use_mixup=False)
            train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size[it], shuffle=True,
                                                       num_workers=8,
                                                       pin_memory=True,
                                                       drop_last=False)
        # len_dataloader = len(train_loader)
        model = model.train()
        running_loss=0.0
        running_content_loss = 0.
        running_pred_loss = torch.tensor(0.)
        with tqdm(total=len(train_loader)) as tq2:
            for train_batch,(data,gt,mask) in enumerate(train_loader):
                if torch.cuda.is_available():
                    if args.use_amp:
                        data=data.half()
                        gt=gt.half()
                        # loc_label=loc_label.half()
                    data=data.cuda()
                    gt=gt.cuda()
                    mask=mask.cuda()
                with amp_cm():
                    # a=0.5 + torch.rand(1).cuda()
                    ori_pred = model(data, gt, mask)

                    feature_pred = vgg19(ori_pred)
                    feature_gt = vgg19(gt)
                    loss = 0
                    # loss = chan(ori_pred, gt)+code_loss+20 * (L1(feature_pred, feature_gt) / (ori_pred.shape[-1] * ori_pred.shape[-2]))#+ 0.5 * cl(ori_pred, gt)+0.5 * weighted_color_loss(ori_pred, gt)
                    loss_content = chan(ori_pred, gt) + 20 * (
                                L1(feature_pred, feature_gt) / (ori_pred.shape[-1] * ori_pred.shape[-2]))
                    # print(fftloss(ori_pred, gt))
                    loss += loss_content +10*( 1 - S(ori_pred, gt)) + 5 * fftloss(ori_pred, gt)#+10*(1-S(ori_pred, gt))+0.1*fftloss(ori_pred, gt)#+code_loss

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                # for name, parms in model.named_parameters():
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
                #           ' -->grad_value:', torch.mean(parms.grad) if parms.grad!=None else 0)
                opt.zero_grad()
                running_loss += loss.item()
                running_content_loss+=loss_content.item()
                # running_pred_loss += loss_lcp.item()
                tq2.set_description(desc="epoch:{}".format(ep+1),refresh=False)
                tq2.update(1)
        now_loss = running_loss / len(train_loader)/t
        now_loss_con = running_content_loss / len(train_loader) / t
        now_loss_pred = running_pred_loss / len(train_loader) / t
        if ep<args.boundary:
            scheduler1.step()
        else:
            scheduler2.step()
        # val_psnr=val(val_loader, model,args,amp_cm)
        test_psnr=0
        if ep==args.boundary:
            torch.save(model.state_dict(), save_path + '/ep_{}.pth'.format(ep + 1))
        if (ep+1)%args.test_freq==0 and now_loss_con<0.055:
            test_psnr,test_ssim = test(test_loader, model, ep, save_path, max_psnr,args,amp_cm)
            # if test_psnr>23.5:
            #     max_psnr=test_psnr
            if test_psnr>args.psnr_st:
                torch.save(model.state_dict(), save_path  + '/ep_{}.pth'.format(ep + 1))
            torch.save(model.state_dict(), save_path + '/newest.pth')
            with open('record/{}.csv'.format(args.mode), 'a', newline='') as f:
                record = csv.writer(f)
                record.writerow([ep+1, test_psnr,test_ssim])

        output_infos = '\rTrain===> [epoch {}/{}] [loss {:.4f}] [loss_content {:.4f}],[loss_falign {:.4f}] [lr: {:.7f}] [val_psnr {:.4f}] [test_psnr {:.4f}] [best_psnr:{:.4f}]'.format(
            ep + 1, args.epochs, now_loss,now_loss_con,now_loss_pred,opt.param_groups[0]['lr'],0,test_psnr,max_psnr)

        print(output_infos)
        print('-----------------------------------------------')

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
                # new_dict = {k: v for k, v in params.items()}
                new_dict1 = {k: v for k, v in params.items() }
            else:
                # new_dict = {k[7:]: v for k, v in params.items()}
                new_dict1 = {k[7:]: v for k, v in params.items() if 'est_lcp' not in k }
            result=model.load_state_dict(new_dict1,strict=False)
            # result2 = aux_model.load_state_dict(new_dict, strict=False)
        else:
            if args.mgpu_train:
                # new_dict = {'module.'+k: v for k, v in params.items()}
                new_dict1 = {'module.' + k: v for k, v in params.items() }
            else:
                new_dict1 = {k: v for k, v in params.items() }
                # new_dict = {k: v for k, v in params.items() }
            result = model.load_state_dict(new_dict1,strict=False)
            # result2 = aux_model.load_state_dict(new_dict, strict=False)
        print(result)
    # if args.contine:
    #     result=model.load_state_dict(torch.load(args.contine_path))
    #     print(result)
    # lf_1=nn.L1Loss()
    lf_1=nn.MSELoss()
    # lf_1=L1_Charbonnier_loss()
    lf_2=SSIM()
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(0.9,0.999),eps=1e-7)
    # scheduler2 = torch.optim.lr_scheduler.MultiStepLR(opt,milestones=list(range(0, args.epochs, 1)), gamma=0.98)
    # scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True)
    # scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,
    #                          [30,60,90],
    #                          0.1
    #                          )
    model=nn.DataParallel(model)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 5, 2)

    scheduler2 =torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,320,1)

    save_path="./{}".format(args.mode)
    train(args,test_loader, model, lf_1, lf_2, opt, scheduler1,scheduler2,save_path)