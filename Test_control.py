import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from utils import *
warnings.filterwarnings('ignore')
from models import Control_Lit
from dataset.load_data import  Dataset_test_baid
import os
import argparse
torch.backends.cudnn.enabled = False
from utils import *
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  #
    torch.backends.cudnn.deterministic = True #

parser = argparse.ArgumentParser(description='')
parser.add_argument('--mode',default='Test_control')
parser.add_argument('--control_degree',default=[-5,-2,0,2,5],help="set values for controlling image illumination")
parser.add_argument('--mgpu_in',default=True)
parser.add_argument('--mgpu_train',default=False)

parser.add_argument('--test_root',default='/data/wuhj/project/seg/demo_mask_v3')
parser.add_argument('--mask_test_root',default='/data/wuhj/project/seg/demo_mask_select_v3',help="provide the region mask needed to be controled")

# swinir
parser.add_argument('--random_seed',default=42)

parser.add_argument('--contine',default=True)
parser.add_argument('--contine_path',default="./ckpt/model.pth")


def test(t_loader,model,pth,args):
    print('testing......')
    model = model.eval()

    with torch.no_grad():
        with tqdm(total=len(t_loader)) as tq:
            for idx,(data,gt,mask,name) in enumerate(t_loader):
                data = data.cuda()
                mask=mask.cuda()
                _, C, W, H = data.shape
                data = check_image_size(8, data)
                mask = check_image_size(8, mask)
                os.makedirs(pth + '/{}'.format(name[0].replace('.JPG','')),exist_ok=True)
                for j in args.control_degree:
                    pred = model.module.forward_vector_sam(data, mask,degree=j * 0.1)
                    pred=pred[:,:,:W,:H]

                    torchvision.utils.save_image(pred,pth + '/{}/deg_{}_{}'.format(name[0].replace('.JPG',''),j,name[0].replace('JPG','png')))
                tq.update()
        return

if __name__=='__main__':
    args=parser.parse_args()
    set_seed(args.random_seed)

    dataset_test = Dataset_test_baid(args.test_root, args.test_root,args.mask_test_root)
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
    model=nn.DataParallel(model)

    save_path="./{}".format(args.mode)
    test(test_loader, model, save_path, args)