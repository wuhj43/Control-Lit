from torch import nn
from VQ.VM import *
from utils import get_gaussian_kernel,get_light_channel,create_dist_grid
from uti.loss_fuc import GradientLoss
from VQ.fsq import *
class Concate_shuffle(nn.Module):
    def __init__(self,dim,out_dim=3):
        super(Concate_shuffle, self).__init__()
        self.adjust=nn.Conv2d(dim,dim,1)
        self.conv1=nn.Sequential(nn.Conv2d(dim, dim,1,padding=0,padding_mode='replicate'))
        # self.conv2=nn.Sequential(nn.Conv2d(dim, dim,3,padding=1,padding_mode='replicate'))#原版只有这里是shuffle
        self.conv3 = nn.Conv1d(1,1,7,padding=3,padding_mode='replicate')
        # self.conv3=nn.Linear(int(out_channel *branch_num),int(out_channel *branch_num))
        if dim==out_dim:
            self.conv4 = nn.Sequential(nn.Conv2d(dim, dim,3,padding=1,padding_mode='replicate'), nn.GELU())
        else:
            self.conv4=nn.Sequential(nn.Conv2d(dim,out_dim,3,padding=1,padding_mode='replicate'),nn.GELU())
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.sig=nn.Sigmoid()
        self.relu=nn.GELU()
    def forward(self, x):
        x=self.adjust(x)
        z=self.pool(x).squeeze()
        z=z.unsqueeze(dim=-2)
        if len(z.shape)==2:
            z=z.unsqueeze(dim=0)
        z=self.conv3(z).transpose(-1, -2).unsqueeze(dim=-1)
        # if len(z.shape) == 1:
        #     z.unsqueeze(0)
        # z=self.conv3(z).unsqueeze(-1).unsqueeze(-1)
        return self.conv4(self.relu((self.conv1(x))*self.sig(z)))

class Diff_cproc_vss_ir_all(nn.Module):
    def __init__(self,in_c,out_c,midc=64,patch_size=2,depth=4):
        super(Diff_cproc_vss_ir_all, self).__init__()
        self.patch=PatchEmbed2D(patch_size, in_c, midc, nn.LayerNorm)
        self.vsslayer=VSSLayer_ir(
                          dim=midc,  # 96
                          depth=depth ,  # 2
                          d_state=16,  # 16
                          drop=0,  # 0
                          attn_drop=0,  # 0
                          drop_path=0.2,  # ，每一个模块传一个概率值
                          norm_layer=nn.LayerNorm,  # nn.LN
                          downsample=None,  # PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                          use_checkpoint=False,
                      )
        self.reconstruct=nn.Sequential(Final_PatchExpand2D(midc, patch_size, nn.LayerNorm),
                                       Concate_shuffle(midc,out_c))

    def forward(self,x):
        x=self.patch(x)
        x=self.vsslayer(x)
        return self.reconstruct(x)

class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        x=x.permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x

class VSSLayer_ir(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(  # 以第一个为例
            self,
            dim,  # # 96
            depth,  # 2
            d_state=16,
            drop=0.,
            attn_drop=0.,
            drop_path=0.2,  # 每一个模块都有一个drop
            norm_layer=nn.LayerNorm,
            downsample=None,  # PatchMergin2D
            use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            nn.Sequential(VSSBlock(
                hidden_dim=dim,  # 96
                drop_path=drop_path,  # 0.2
                norm_layer=norm_layer,  # nn.LN
                attn_drop_rate=attn_drop,  # 0
                d_state=d_state,  # 16
            ),
                VSSBlock(
                    hidden_dim=dim,  # 96
                    drop_path=drop_path,  # 0.2
                    norm_layer=norm_layer,  # nn.LN
                    attn_drop_rate=attn_drop,  # 0
                    d_state=d_state,  # 16
                )
            )
            for i in range(depth)])
        self.convs=nn.ModuleList([ nn.Conv2d(dim,dim,3,1,1) for i in range(int(depth))])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for idx,blk in enumerate(self.blocks):
            x_1 = blk(x)
            x_1=x_1.permute(0,3,1,2)
            x_1=self.convs[idx](x_1).permute(0,2,3,1)
            x=x+x_1
        if self.downsample is not None:
            x = self.downsample(x)

        return x

def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv=default_conv, n_feats=64, kernel_size=3,
        bias=True, bn=False, act=nn.PReLU(), res_scale=1,depth=2):

        super(ResBlock, self).__init__()
        m = []
        for i in range(depth):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Concate_shuffle_res_dcp(nn.Module):
    def __init__(self,dim,out_dim=3):
        super(Concate_shuffle_res_dcp, self).__init__()
        self.adjust=nn.Conv2d(dim,dim,1)
        self.conv1=nn.Sequential(nn.Conv2d(dim, dim,1,padding=0,padding_mode='replicate'))
        self.conv2=nn.Sequential(nn.Conv2d(dim, dim,3,padding=1,padding_mode='replicate'))#原版只有这里是shuffle
        self.conv3 = nn.Conv1d(1,1,7,padding=3,padding_mode='replicate')
        # self.conv3=nn.Linear(int(out_channel *branch_num),int(out_channel *branch_num))
        if dim==out_dim:
            self.conv4 = nn.Sequential(nn.Conv2d(dim, dim,3,padding=1,padding_mode='replicate'))
        else:
            self.conv4=nn.Sequential(nn.Conv2d(dim,out_dim,3,padding=1,padding_mode='replicate'))
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.sig=nn.Sigmoid()
        self.relu=nn.GELU()
    def forward(self, x):
        x=self.adjust(x)
        z=self.pool(x).squeeze()
        z=z.unsqueeze(dim=-2)
        if len(z.shape)==2:
            z=z.unsqueeze(dim=0)
        z=self.conv3(z).transpose(-1, -2).unsqueeze(dim=-1)
        return self.conv4(self.relu((self.conv1(x)+self.conv2(x))*self.sig(z)))


class DCP_light_pred_algo_vss_3in_mscale(nn.Module):
    def __init__(self):
        super(DCP_light_pred_algo_vss_3in_mscale, self).__init__()
        # self.trans_for = nn.Sequential(nn.Conv2d(4, 64, 3, 1, 1), nn.GELU(),MSAB(dim=64, num_blocks=1, dim_head=64 // 4, heads=4))
        self.trans = nn.Sequential(PatchEmbed2D(2, 4, 32, nn.LayerNorm),
                                   # nn.Conv2d(in_c, self.channel_numb*4 , (1, 1), padding=0),
                                   VSSLayer_ir(
                                       dim=32,  # 96
                                       depth=1,  # 2
                                       d_state=8,  # 16
                                       drop=0,  # 0
                                       attn_drop=0,  # 0
                                       drop_path=0.2,  # ，每一个模块传一个概率值
                                       norm_layer=nn.LayerNorm,  # nn.LN
                                       downsample=None,
                                       # PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                                       use_checkpoint=False,
                                   ),
                                   Final_PatchExpand2D(32, 2, nn.LayerNorm)
                                   )
        self.trans2 = nn.Sequential(PatchEmbed2D(2, 32 + 3, 48, nn.LayerNorm),
                                    # nn.Conv2d(in_c, self.channel_numb*4 , (1, 1), padding=0),
                                    VSSLayer_ir(
                                        dim=48,  # 96
                                        depth=1,  # 2
                                        d_state=8,  # 16
                                        drop=0,  # 0
                                        attn_drop=0,  # 0
                                        drop_path=0.2,  # ，每一个模块传一个概率值
                                        norm_layer=nn.LayerNorm,  # nn.LN
                                        downsample=None,
                                        # PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                                        use_checkpoint=False,
                                    ),
                                    Final_PatchExpand2D(48, 2, nn.LayerNorm)
                                    )
        self.trans3 = nn.Sequential(PatchEmbed2D(2, 48 + 3, 64, nn.LayerNorm),
                                    # nn.Conv2d(in_c, self.channel_numb*4 , (1, 1), padding=0),
                                    VSSLayer_ir(
                                        dim=64,  # 96
                                        depth=1,  # 2
                                        d_state=8,  # 16
                                        drop=0,  # 0
                                        attn_drop=0,  # 0
                                        drop_path=0.2,  # ，每一个模块传一个概率值
                                        norm_layer=nn.LayerNorm,  # nn.LN
                                        downsample=None,
                                        # PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                                        use_checkpoint=False,
                                    ),
                                    Final_PatchExpand2D(64, 2, nn.LayerNorm)
                                    )
        self.down = nn.MaxPool2d((2, 2), 2, return_indices=True)
        self.gauss_k = get_gaussian_kernel(kernel_size=5)
        self.out = nn.Sequential(
            Concate_shuffle_res_dcp(64 + 32 + 48, 1))  # nn.Sequential(nn.Conv2d(128,1,3,1,1),nn.GELU())
        self.pool = nn.Sequential(nn.Conv2d(1, 64, 1, 1), nn.GELU())
        self.fc = nn.Sequential(nn.Linear(64, 1), nn.GELU())
        self.up = nn.MaxUnpool2d((2, 2), 2)
        self.weig = nn.Parameter(torch.Tensor([1.]).cuda(), requires_grad=True)
        # self.up2 = nn.MaxUnpool2d((4, 4), 4)
        # self.conv=nn.Sequential(nn.Conv2d(1,64,3,1,1),nn.GELU(),nn.BatchNorm2d(64),nn.Conv2d(64,1,3,1,1),nn.GELU())
        # self.out2 = nn.Sequential(nn.Conv2d(128, 1, 1, 1), nn.GELU())

    def down_lplas(self, img):
        img_d1 = F.interpolate(self.gauss_k(img), scale_factor=0.5, mode='bicubic')
        img_d2 = F.interpolate(self.gauss_k(img_d1), scale_factor=0.5, mode='bicubic')
        r1 = img - F.interpolate(self.gauss_k(img_d1), scale_factor=2, mode='bicubic')
        r2 = img_d1 - F.interpolate(self.gauss_k(img_d2), scale_factor=2, mode='bicubic')
        return r1, r2, img_d2

    def forward(self, x, gt=None, img_short=None):
        B, C, H, W = x.shape
        if x.shape[1] == 1:
            lcp = x
            # lcp = F.interpolate(lcp, (128, 128))
            B, C, H, W = x.shape
            img_d2 = img_short
            # lcp = torch.cat([lcp, img_short, 1 - lcp], dim=1)
        else:
            if img_short == None:
                r1, r2, img_d2 = self.down_lplas(x)
                B, C, H, W = img_d2.shape

                lcp = get_light_channel(img_d2, 5).unsqueeze(1)
                # lcp_large = get_light_true_channel(img_d2, 33).unsqueeze(1)
                # lcp_large=F.interpolate(lcp_large, (128, 128))
                lcp = F.interpolate(lcp, (128, 128))
                # lcp2 = lcp / (lcp_large + 1e-6)
                # lcp =torch.cat([lcp,img_d2,1-lcp],dim=1)


            else:
                B, C, H, W = img_short.shape
                lcp = get_light_channel(img_short, 5).unsqueeze(1)
                # lcp_large = get_light_true_channel(img_short, 33).unsqueeze(1)
                # lcp_large = F.interpolate(lcp_large, (128, 128))
                lcp = F.interpolate(lcp, (128, 128))
                img_d2 = img_short
        lcp_tmp = lcp
        # lcp_tmp = strech(lcp, al)
        out1 = self.trans(torch.cat([lcp_tmp, img_d2], dim=1))
        out1_d, idx1 = self.down(out1)
        out2 = self.trans2(torch.cat([out1_d, F.interpolate(img_d2, scale_factor=0.5)], dim=1))
        out2_d, idx2 = self.down(out2)
        out3 = self.trans3(torch.cat([out2_d, F.interpolate(img_d2, scale_factor=0.25)], dim=1))
        out2_up = F.interpolate(out2, scale_factor=2, mode='nearest')
        out3_up = F.interpolate(out3, scale_factor=4, mode='nearest')
        # trans_for_out = self.trans_for(torch.cat([lcp_tmp, img_d2], dim=1))
        # lcp=torch.pow(lcp,al.unsqueeze(-1).unsqueeze(-1))
        # lcp_pred =lcp_tmp*(1+self.out(mam_out*(1-mask)+trans_for_out*(mask)))

        # lcp_pred =strech(lcp,self.out(torch.cat([out1,out2_up,out3_up],dim=1)))
        lcp_pred = self.weig * lcp + self.out(torch.cat([out1, out2_up, out3_up], dim=1))
        # lcp_pred
        # lcp_pred = F.interpolate(lcp_pred, (H, W))
        if gt == None:
            return lcp_pred
        else:
            gt_r1, gt_r2, gt_img_d2 = self.down_lplas(gt)
            lcp_gt = get_light_channel(gt_img_d2, 5).unsqueeze(1)
            # lcp_gt_large = get_light_true_channel(gt_img_d2, 33).unsqueeze(1)
            B, C, H, W = lcp_gt.shape

            # lcp_gt = F.interpolate(lcp_gt, (128, 128))

            return lcp_pred, lcp_gt


class Control_Lit(nn.Module):
    def __init__(self, in_c,out_c,stage,region=25,num_blocks1=4,num_blocks2=2,channel_numb=32,depth=16,n_e=1024,rank=4,weight=10):
        super(Control_Lit, self).__init__()
        self.channel_numb=channel_numb
        # self.conv_l1 = nn.Sequential(nn.Conv2d(in_c*2, self.channel_numb, (3, 3), padding=1),nn.GELU(),nn.Conv2d(self.channel_numb, out_c,(3, 3), padding=1))
        # self.conv_l2 = nn.Sequential(nn.Conv2d(in_c*2, self.channel_numb, (3, 3), padding=1),nn.GELU(),nn.Conv2d(self.channel_numb, out_c,(3, 3), padding=1))
        self.conv_l1 = Diff_cproc_vss_ir_all(in_c*3,in_c,64,patch_size=4,depth=4)#nn.Sequential(nn.Conv2d(in_c*2, self.channel_numb, (3, 3), padding=1),nn.GELU(),nn.Conv2d(self.channel_numb, out_c,(3, 3), padding=1))
        self.conv_l2 = Diff_cproc_vss_ir_all(in_c*3,in_c,64,patch_size=4,depth=4)#nn.Sequential(nn.Conv2d(in_c*2, self.channel_numb, (3, 3), padding=1),nn.GELU(),nn.Conv2d(self.channel_numb, out_c,(3, 3), padding=1))
        self.conv_l3 = nn.Sequential(PatchEmbed2D(2,in_c, self.channel_numb*4,nn.LayerNorm),#nn.Conv2d(in_c, self.channel_numb*4 , (1, 1), padding=0),
                                     VSSLayer_ir(
                                         dim=self.channel_numb*4,  # 96
                                         depth=num_blocks1,  # 2
                                         d_state=8,  # 16
                                         drop=0,  # 0
                                         attn_drop=0,  # 0
                                         drop_path=0.2,  # ，每一个模块传一个概率值
                                         norm_layer=nn.LayerNorm,  # nn.LN
                                         downsample=None,#PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=False,
                                     ),
                                     Final_PatchExpand2D(self.channel_numb*4,2,nn.LayerNorm)
                                     )
        self.istnorm=nn.InstanceNorm2d(self.channel_numb*4)

        self.conv_r3 = nn.Sequential(#nn.Conv2d(self.channel_numb* 2, self.channel_numb * 2, (1, 1), padding=0),
                                     # MSAB(dim=self.channel_numb*4, num_blocks=num_blocks, dim_head=self.channel_numb*4 // 4, heads=4),
                                     # ResBlock(n_feats=self.channel_numb * 4, act=nn.GELU()),
                                     ResBlock(n_feats=self.channel_numb * 4, act=nn.GELU(),conv=default_conv),
                                     ResBlock(n_feats=self.channel_numb * 4, act=nn.GELU(),conv=default_conv),
                                     nn.Conv2d(self.channel_numb*4, out_c,1))
        self.one = nn.Parameter(torch.eye(channel_numb*4).cuda(), requires_grad=False)
        self.gauss_k=get_gaussian_kernel(kernel_size=5)
        self.n_e=n_e
        self.depth=depth
        self.vq3= FSQ(levels=[8,5,5,5],dim=self.channel_numb*4)#VQModule_norm_sparse_sample_reinit_vm_QR_lcp_spreadv3_v2(64 * 4 * 2, self.channel_numb*4, depth=depth,n_e=n_e,rank=rank,weight=weight)  #(64 * 4 * 2, self.channel_numb*4, depth=dep,stage=stage)
        self.stage=stage
        self.gradient=GradientLoss()
        self.unfold = nn.Unfold(kernel_size=(2, 2), stride=2)
        # self.est_lcp = nn.Sequential(nn.Conv2d(1, self.channel_numb * 4, (1, 1), padding=0),nn.BatchNorm2d(self.channel_numb * 4),nn.GELU(),
        #                              MSAB(dim=self.channel_numb * 4, num_blocks=num_blocks,
        #                                   dim_head=self.channel_numb * 4 // 4, heads=4),
        #                              nn.Conv2d(self.channel_numb * 4,1, (1, 1), padding=0),nn.GELU(),
        #                              )
        # self.est_lcp=Diff_cproc_se(1,1,128)
        self.est_lcp = DCP_light_pred_algo_vss_3in_mscale()
        # params = torch.load('/data/wuhj/project/seg/DCP_light_pred_algo_3in_mscale/ep_1480.pth')
        # new_dict = {k: v for k, v in params.items()}
        # load = self.est_lcp.load_state_dict(new_dict)
        # print(load)
        # if self.stage==1:
        #     self.adj_m =nn.Sequential(PatchEmbed2D(2,self.channel_numb*4, self.channel_numb*4,nn.LayerNorm),#nn.Conv2d(in_c, self.channel_numb*4 , (1, 1), padding=0),
        #                               VSSLayer_ir(
        #                                   dim=self.channel_numb * 4,  # 96
        #                                   depth=num_blocks2,  # 2
        #                                   d_state=8,  # 16
        #                                   drop=0,  # 0
        #                                   attn_drop=0,  # 0
        #                                   drop_path=0.2,  # ，每一个模块传一个概率值
        #                                   norm_layer=nn.LayerNorm,  # nn.LN
        #                                   downsample=None,
        #                                   # PatchMerging2D if (i_layer < self.num_layers - 1) else None,
        #                                   use_checkpoint=False,
        #                               ),
        #                              Final_PatchExpand2D(self.channel_numb*4,2,nn.LayerNorm)
        #                              )
        #     self.need_idx=False
        #     for name,param in self.named_parameters():
        #         if ('conv_l3' in name) or ('vq' in name) or ('conv_r' in name) or ('gauss_k' in name):#('vq' in name) or
        #             param.requires_grad=False

        if self.stage==1:
            self.adj_m = nn.Sequential(PatchEmbed2D(2, self.channel_numb * 4, self.channel_numb * 4, nn.LayerNorm),
                                       # nn.Conv2d(in_c, self.channel_numb*4 , (1, 1), padding=0),
                                       VSSLayer_ir(
                                           dim=self.channel_numb * 4,  # 96
                                           depth=num_blocks2,  # 2
                                           d_state=8,  # 16
                                           drop=0,  # 0
                                           attn_drop=0,  # 0
                                           drop_path=0.2,  # ，每一个模块传一个概率值
                                           norm_layer=nn.LayerNorm,  # nn.LN
                                           downsample=None,
                                           # PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                                           use_checkpoint=False,
                                       ),
                                       Final_PatchExpand2D(self.channel_numb * 4, 2, nn.LayerNorm)
                                       )
            # for name,param in self.named_parameters():
            #     if ('conv_l3' in name) or ('vq' in name) or ('conv_r' in name) or ('gauss_k' in name):#('vq' in name) or
            #         param.requires_grad=False
            self.light_vector=nn.Parameter(torch.randn(4,1).cuda(), requires_grad=True)
            self.light_bias = nn.Parameter(torch.randn(1, 1).cuda(), requires_grad=True)

        if self.stage==2:
            self.adj_m = nn.Sequential(PatchEmbed2D(2, self.channel_numb * 4, self.channel_numb * 4, nn.LayerNorm),
                                       # nn.Conv2d(in_c, self.channel_numb*4 , (1, 1), padding=0),
                                       VSSLayer_ir(
                                           dim=self.channel_numb * 4,  # 96
                                           depth=num_blocks2,  # 2
                                           d_state=8,  # 16
                                           drop=0,  # 0
                                           attn_drop=0,  # 0
                                           drop_path=0.2,  # ，每一个模块传一个概率值
                                           norm_layer=nn.LayerNorm,  # nn.LN
                                           downsample=None,
                                           # PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                                           use_checkpoint=False,
                                       ),
                                       Final_PatchExpand2D(self.channel_numb * 4, 2, nn.LayerNorm)
                                       )
            for name,param in self.named_parameters():
                if ('conv_l3' in name) or ('vq' in name) or ('conv_r' in name) or ('gauss_k' in name):#('vq' in name) or
                    param.requires_grad=False
            self.light_vector=nn.Parameter(torch.randn(4,1).cuda(), requires_grad=False)
            self.light_bias = nn.Parameter(torch.randn(1, 1).cuda(), requires_grad=False)


    def forward_swap_pixel(self, x,gt,mask,test=False):

        r1,r2,img_d2=self.down_lplas(x)
        B, C, H, W = img_d2.shape
        gt_r1, gt_r2, gt_img_d2 = self.down_lplas(gt)

        var_map = torch.var(self.unfold(torch.mean(img_d2, dim=1, keepdim=True)), dim=1, keepdim=True).view(B, 1,int(H / 2),int(W / 2))
        mean_map = torch.mean(img_d2, dim=1, keepdim=True)
        layer_f3 = self.conv_l3(img_d2)  #
        gt_layer_f3 = self.conv_l3(gt_img_d2)
        b, c, h, w = layer_f3.shape
        # layer_f3=gt_layer_f3/torch.mean(gt_layer_f3,dim=[1],keepdim=True)*torch.mean(layer_f3,dim=[1],keepdim=True)
        layer_f3 = layer_f3 * torch.mean(gt_layer_f3, dim=[1], keepdim=True) / torch.mean(layer_f3, dim=[1],
                                                                                             keepdim=True)
        # layer_f3[:,:,:, 0::2]=gt_layer_f3[:,:,:, 0::2]
        # layer_f3[:,:,:, 1::2] = gt_layer_f3[:,:,:, 1::2]
        # layer_f3[:,int(c/4):int(c/4)*3]=gt_layer_f3[:,int(c/4):int(c/4)*3]
        # layer_f3=gt_layer_f3
        fq3, codebook_loss3, distance_map, est_m_v, idex = self.vq3(layer_f3, [var_map, mean_map], tau=1e-4,test=test)

        # gt_fq3, gt_codebook_loss3, gt_distance_map, gt_est_m_v, gt_idex = self.vq3(gt_layer_f3, [var_map, mean_map], tau=1e-4, test=test)

        r3_rec = self.conv_r3(fq3)

        result = self.up_lplas(gt_r1, gt_r2, r3_rec)

        return result, codebook_loss3, [r3_rec], layer_f3, idex
    def get_ref_conv_l3(self):
        self.conv_l3_ref=nn.ModuleList([self.copy_module()])
        for name, param in self.conv_l3_ref.named_parameters():
            # if ('conv_l3' in name) or ('vq' in name) or ('conv_r' in name) or ('gauss_k' in name):  # ('vq' in name) or
                param.requires_grad = False
    def copy_module(self):
        copy = self.conv_l3.__class__(*self.conv_l3.parameters())
        return copy
    def down_lplas(self,img):
        img_d1=F.interpolate(self.gauss_k(img),scale_factor=0.5,mode='bicubic')
        img_d2=F.interpolate(self.gauss_k(img_d1),scale_factor=0.5,mode='bicubic')
        r1=img-F.interpolate(self.gauss_k(img_d1),scale_factor=2,mode='bicubic')
        r2 = img_d1 - F.interpolate(self.gauss_k(img_d2), scale_factor=2, mode='bicubic')
        return r1,r2,img_d2
    def up_lplas(self,r1,r2,img_d2):
        img_d1=F.interpolate(self.gauss_k(img_d2), scale_factor=2, mode='bicubic')+r2
        img = F.interpolate(self.gauss_k(img_d1), scale_factor=2, mode='bicubic') + r1
        return img
    def get_encoder_feature(self,x):
        with torch.no_grad():
            r1, r2, img_d2 = self.down_lplas(x)
            # layer_f1 = self.conv_l1(r1)
            # layer_f2 = self.conv_l2(r2)
            layer_f3 = self.conv_l3(img_d2)
            return [layer_f3]
    def get_fq_feature(self,x):
        with torch.no_grad():
            r1, r2, img_d2 = self.down_lplas(x)
            # layer_f1 = self.conv_l1(r1)
            # layer_f1,_,_=self.vq1(layer_f1)
            # layer_f2 = self.conv_l2(r2)
            # layer_f2,_,_ = self.vq2(layer_f2)
            layer_f3 = self.conv_l3(img_d2)
            layer_f3, _, _ = self.vq3(layer_f3)
            return [layer_f3]
    def pred_mask(self,ori_pred,gt):
        return self.predictor(torch.cat([ori_pred,gt],dim=1))
    def forward(self, x,gt,mask,test=False,use_gt_lcp=False,use_gt_high=False,need_reconstruct=False):
        r1,r2,img_d2=self.down_lplas(x)
        B, C, H, W = img_d2.shape
        gt_r1, gt_r2, gt_img_d2 = self.down_lplas(gt)
        if self.stage==-1:
            c=get_light_channel(img_d2, 7).unsqueeze(1)
            est_c=self.est_lcp(c)+c
            gt_c=get_light_channel(gt_img_d2, 7).unsqueeze(1)
            return est_c,gt_c,c
        if self.stage==0:
            # var_map = torch.var(self.unfold(torch.mean(img_d2, dim=1, keepdim=True)), dim=1, keepdim=True).view(B, 1,int(H / 2),int(W / 2))
            # mean_map = torch.mean(img_d2, dim=1, keepdim=True)
            layer_f3 = self.conv_l3(img_d2)  #
            # lcp_backlit = get_light_channel(img_d2,7)
            # lcp_light = get_light_channel(gt_img_d2, 7)
            # if not use_gt_lcp:
            #     lcp= create_dist_grid(lcp_backlit.flatten(),self.n_e)
            # else:
            #     lcp = create_dist_grid(lcp_light.flatten(), self.n_e)

            fq3,ids= self.vq3(layer_f3)
            r3_rec = self.conv_r3(fq3)
            if use_gt_high:
                result = self.up_lplas(gt_r1, gt_r2, r3_rec)
            else:
                result = self.up_lplas(r1, r2, r3_rec)

            return result, [r3_rec],
        elif self.stage==1:
            layer_f3 = self.conv_l3(img_d2)
            # mask = F.interpolate(mask, size=[layer_f3.shape[-2], layer_f3.shape[-1]], mode='bicubic')
            fq3=layer_f3
            # tv.utils.save_image(get_light_channel(img_d2, 7), '/data/wuhongjun/project/segv2/lcp_extract/lcp.png')
            # lcp_pred=self.est_lcp(get_light_channel(img_d2, 7).unsqueeze(1))
            # lcp_pred_flatten = create_dist_grid(lcp_pred.flatten(), self.n_e)

            fq3_after, idexs = self.vq3(fq3)

            # il_map=self.proj(torch.max(img_d2,dim=1,keepdim=True)[0])
            # il_map_inv=self.proj(1-torch.max(img_d2,dim=1,keepdim=True)[0])


            # m_shift=self.est(torch.cat([fq3_after,il_map],dim=1))
            # v_shift=self.est_m(torch.cat([fq3_after,il_map],dim=1))
            #
            # m_shift_inv = self.est(torch.cat([fq3_after, il_map_inv], dim=1))
            # v_shift_inv = self.est_m(torch.cat([fq3_after, il_map_inv], dim=1))


            # rot_input=self.pool(fq3_after).squeeze(-1).squeeze(-1)

            fq3_after = fq3_after + ((self.adj_m(fq3_after)))# * (1-lcp_pred)

            # fq3_after = fq3_after+((self.adj_m(fq3_after.permute(0, 2, 3, 1))).permute(0, 3, 1, 2))*mask+((self.adj_m_inv(fq3_after.permute(0, 2, 3, 1))).permute(0, 3, 1, 2))*(1-mask)
            # fq3_after=(fq3_after*(1+v_shift)+m_shift)*mask+(fq3_after*(1+v_shift_inv)+m_shift_inv)*(1-mask)
            r3_rec = self.conv_r3(fq3_after)
            #
            gra1 = self.gradient.construct_gt(x)
            gra2 = self.gradient.construct_gt(F.interpolate(x, scale_factor=0.5, mode='bicubic'))
            #
            r3_rec_up2=F.interpolate(img_d2,scale_factor=2,mode='bicubic')
            r2_rec=self.conv_l2(torch.cat([r3_rec_up2,r2,gra2],dim=1))

            r3_rec_up4 = F.interpolate(img_d2,scale_factor=4,mode='bicubic')
            r1_rec = self.conv_l1(torch.cat([r3_rec_up4, r1,gra1],dim=1))

            result = self.up_lplas(r1_rec,r2_rec,r3_rec)
            # mask_pred = self.pred_mask(result, gt)

            if self.training:
                # gt_layer_f3 = self.conv_l3(gt_img_d2)
                # gt_fq3_after, codebook_loss3_gt, _, _, idgt = self.vq3(gt_layer_f3,[0,0], tau=1, test=False)
                # gt_r3_rec = self.conv_r3(gt_fq3_after)
                #
                # gt_m=torch.mean(gt_layer_f3,dim=1).flatten()
                # gt_v = torch.var(gt_layer_f3, dim=1).flatten()
                # est_m = torch.mean(fq3_after, dim=1).flatten()
                # est_v = torch.var(fq3_after, dim=1).flatten()
                #
                # # [gt_m,gt_v]=self.vq3.get_ref_m_v(gt_layer_f3)
                # # [est_m,est_v] = self.vq3.get_ref_m_v(fq3)
                # # est_m = est_m.squeeze() * (1 + torch.flatten(v_shift)) + torch.flatten(m_shift)
                # # est_v = est_v.squeeze() * (1 + torch.flatten(v_shift)) * (1 + torch.flatten(v_shift))
                #
                # # A=self.adj_m(self.one)
                # # A_inv = self.adj_m_inv(self.one)
                # # F_loss=torch.norm(A-A.T)+torch.norm(A_inv-A_inv.T)
                # F_loss=torch.zeros(1).cuda()
                # lcp_gt=get_light_channel(gt_img_d2, 7).unsqueeze(1)
                return result
                # return result, F_loss, [fq3, gt_layer_f3], [fq3_after, gt_fq3_after], [
                #     r3_rec, gt_r3_rec],[est_m,est_v,gt_m,gt_v]

            else:
                # if self.need_idx:
                #     [gt_m,gt_v] = self.vq3.get_ref_m_v(gt_layer_f3)
                #     return result, codebook_loss3+codebook_loss3_gt,[fq3,gt_layer_f3],[fq3_after,gt_fq3_after],[r3_rec,gt_r3_rec],[est_m,est_v,gt_m,gt_v],idexs
                # else:
                return result
        elif self.stage==2:

            layer_f3 = self.conv_l3(img_d2)  #
            # lcp_backlit = get_light_channel(img_d2,7)
            # lcp_light = get_light_channel(gt_img_d2, 7)
            # if not use_gt_lcp:
            #     lcp= create_dist_grid(lcp_backlit.flatten(),self.n_e)
            # else:
            #     lcp = create_dist_grid(lcp_light.flatten(), self.n_e)

            if need_reconstruct:
                fq3, ids = self.vq3(layer_f3)
                r3_rec = self.conv_r3(fq3)
                if use_gt_high:
                    result = self.up_lplas(gt_r1, gt_r2, r3_rec)
                else:
                    result = self.up_lplas(r1, r2, r3_rec)

                # idx=self.vq3.get_idx(layer_f3)
                # c = get_light_channel(img_d2, 7).unsqueeze(1)
                # # gt_c=get_light_channel(gt_img_d2, 7).unsqueeze(1)
                # pred_c=idx@self.light_vector#+self.light_bias

                return result#,pred_c,(c).permute(0,2,3,1).view(c.shape[0],-1,1)
            else:
                c = get_light_channel(img_d2, 7).unsqueeze(1)
                gt_c=get_light_channel(gt_img_d2, 7).unsqueeze(1)
                diff_c=(gt_c - c).permute(0,2,3,1).view(c.shape[0],-1,1)
                feature,_ = self.vq3(layer_f3, (diff_c.repeat(1,1,4)* self.light_vector.squeeze()))
                r3_rec = self.conv_r3(feature)
                if use_gt_high:
                    result = self.up_lplas(gt_r1, gt_r2, r3_rec)
                else:
                    result = self.up_lplas(r1, r2, r3_rec)
                return result#pred_c,(c).permute(0,2,3,1).view(c.shape[0],-1,1)
        elif self.stage==3:
            layer_f3 = self.conv_l3(img_d2)  #
            c = get_light_channel(img_d2, 7).unsqueeze(1)
            gt_c=self.est_lcp(c,img_short=img_d2)
            diff_c=(gt_c - c).permute(0,2,3,1).view(c.shape[0],-1,1)
            fq3_after,_ = self.vq3(layer_f3, (diff_c.repeat(1,1,4)* self.light_vector.squeeze()))
            # r3_rec = self.conv_r3(feature)
            fq3_after = fq3_after + ((self.adj_m(fq3_after)))
            r3_rec = self.conv_r3(fq3_after)
            #
            gra1 = self.gradient.construct_gt(x)
            gra2 = self.gradient.construct_gt(F.interpolate(x, scale_factor=0.5, mode='bicubic'))
            #
            r3_rec_up2 = F.interpolate(img_d2, scale_factor=2, mode='bicubic')
            r2_rec = self.conv_l2(torch.cat([r3_rec_up2, r2, gra2], dim=1))

            r3_rec_up4 = F.interpolate(img_d2, scale_factor=4, mode='bicubic')
            r1_rec = self.conv_l1(torch.cat([r3_rec_up4, r1, gra1], dim=1))

            result = self.up_lplas(r1_rec, r2_rec, r3_rec)
            return result
        elif self.stage==45:
            layer_f3 = self.conv_l3(img_d2)  #
            c = get_light_channel(img_d2, 7).unsqueeze(1)
            gt_c=self.est_lcp(c,img_short=img_d2)
            diff_c=(gt_c - c).permute(0,2,3,1).view(c.shape[0],-1,1)
            fq3_after,_ = self.vq3(layer_f3, (diff_c.repeat(1,1,4)* self.light_vector.squeeze()))
            # r3_rec = self.conv_r3(feature)
            # fq3_after = fq3_after + ((self.adj_m(fq3_after)))
            r3_rec = self.conv_r3(fq3_after)
            #
            # gra1 = self.gradient.construct_gt(x)
            # gra2 = self.gradient.construct_gt(F.interpolate(x, scale_factor=0.5, mode='bicubic'))
            #
            # r3_rec_up2 = F.interpolate(img_d2, scale_factor=2, mode='bicubic')
            # r2_rec = self.conv_l2(torch.cat([r3_rec_up2, r2, gra2], dim=1))
            #
            # r3_rec_up4 = F.interpolate(img_d2, scale_factor=4, mode='bicubic')
            # r1_rec = self.conv_l1(torch.cat([r3_rec_up4, r1, gra1], dim=1))

            result = self.up_lplas(r1, r2, r3_rec)
            return result

    def visual(self, x):
        r1, r2, img_d2 = self.down_lplas(x)
        layer_f3 = self.conv_l3(img_d2)
        fq3_after, idxs = self.vq3(layer_f3)
        return fq3_after, idxs


    def dilate_mask(self,mask, kernel_size=3):
        # 创建一个全1的卷积核
        kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32).cuda()

        # 将输入的二值mask升维，以适应卷积操作的输入要求
        # mask = mask.unsqueeze(0).unsqueeze(0).float()  # (B, C, H, W) -> (1, 1, H, W)

        # 对mask应用卷积操作
        dilated_mask = F.conv2d(mask, kernel, padding=kernel_size // 2)

        # 将膨胀后的mask转为二值
        dilated_mask = (dilated_mask > 0).float()

        # 去掉多余的维度
        # dilated_mask = dilated_mask.squeeze(0).squeeze(0)

        return dilated_mask
    def forward_vector_sam(self, x,mask,degree=0.,):
        r1,r2,img_d2=self.down_lplas(x)

        layer_f3 = self.conv_l3(img_d2)
        fq3 = layer_f3
        mask = F.interpolate(mask, size=[layer_f3.shape[-2], layer_f3.shape[-1]], mode='bicubic')
        # mask[mask==0]=-0.1
        fq3_after, idexs = self.vq3.forward_vector_sam(fq3,self.light_vector*degree,mask)

        fq3_after = fq3_after + ((self.adj_m(fq3_after)))

        r3_rec = self.conv_r3(fq3_after)
        #
        gra1 = self.gradient.construct_gt(x)
        gra2 = self.gradient.construct_gt(F.interpolate(x, scale_factor=0.5, mode='bicubic'))
        #
        r3_rec_up2 = F.interpolate(img_d2, scale_factor=2, mode='bicubic')
        r2_rec = self.conv_l2(torch.cat([r3_rec_up2, r2, gra2], dim=1))

        r3_rec_up4 = F.interpolate(img_d2, scale_factor=4, mode='bicubic')
        r1_rec = self.conv_l1(torch.cat([r3_rec_up4, r1, gra1], dim=1))

        result = self.up_lplas(r1_rec, r2_rec, r3_rec)
        # mask_pred = self.pred_mask(result, gt)

        return result

    def forward_vector_sam_darkenbg(self, x,mask,maskbg,degree=0.,):
        r1,r2,img_d2=self.down_lplas(x)

        layer_f3 = self.conv_l3(img_d2)
        fq3 = layer_f3
        mask = F.interpolate(mask, size=[layer_f3.shape[-2], layer_f3.shape[-1]], mode='bicubic')
        maskbg = F.interpolate(maskbg, size=[layer_f3.shape[-2], layer_f3.shape[-1]], mode='bicubic')
        # mask[mask==0]=-0.1
        fq3_after, idexs = self.vq3.forward_vector_sam_darkenbg(fq3,self.light_vector,mask*degree,maskbg)

        fq3_after = fq3_after + ((self.adj_m(fq3_after)))*mask

        r3_rec = self.conv_r3(fq3_after)
        #
        gra1 = self.gradient.construct_gt(x)
        gra2 = self.gradient.construct_gt(F.interpolate(x, scale_factor=0.5, mode='bicubic'))
        #
        r3_rec_up2 = F.interpolate(img_d2, scale_factor=2, mode='bicubic')
        r2_rec = self.conv_l2(torch.cat([r3_rec_up2, r2, gra2], dim=1))

        r3_rec_up4 = F.interpolate(img_d2, scale_factor=4, mode='bicubic')
        r1_rec = self.conv_l1(torch.cat([r3_rec_up4, r1, gra1], dim=1))

        result = self.up_lplas(r1_rec, r2_rec, r3_rec)
        # mask_pred = self.pred_mask(result, gt)

        return result


    def forward_vector(self, x,degree=0.):
        r1,r2,img_d2=self.down_lplas(x)

        layer_f3 = self.conv_l3(img_d2)
        fq3 = layer_f3

        fq3_after, idexs = self.vq3.forward_vector(fq3,self.light_vector*degree)

        fq3_after = fq3_after + ((self.adj_m(fq3_after)))

        r3_rec = self.conv_r3(fq3_after)
        #
        gra1 = self.gradient.construct_gt(x)
        gra2 = self.gradient.construct_gt(F.interpolate(x, scale_factor=0.5, mode='bicubic'))
        #
        r3_rec_up2 = F.interpolate(img_d2, scale_factor=2, mode='bicubic')
        r2_rec = self.conv_l2(torch.cat([r3_rec_up2, r2, gra2], dim=1))

        r3_rec_up4 = F.interpolate(img_d2, scale_factor=4, mode='bicubic')
        r1_rec = self.conv_l1(torch.cat([r3_rec_up4, r1, gra1], dim=1))

        result = self.up_lplas(r1_rec, r2_rec, r3_rec)
        # mask_pred = self.pred_mask(result, gt)

        return result

    def forward_vector_stage1(self, x,degree=0.):
        r1,r2,img_d2=self.down_lplas(x)
        # B, C, H, W = img_d2.shape
        # gt_r1, gt_r2, gt_img_d2 = self.down_lplas(gt)

        layer_f3 = self.conv_l3(img_d2)
        # mask = F.interpolate(mask, size=[layer_f3.shape[-2], layer_f3.shape[-1]], mode='bicubic')
        # mask[:]=0
        fq3 = layer_f3
        # lcp_ori=get_light_channel(img_d2, 7).unsqueeze(1)
        # lcp_pred = self.est_lcp(get_light_channel(img_d2, 7).unsqueeze(1))
        # for i in range(50):
        #     mask=self.dilate_mask(mask)
        # lcp_mix=mask*degree#(1-(lcp_ori+(lcp_pred-lcp_ori)*degree))*(mask)+lcp_ori*(1-mask)#mask#mask+lcp_ori*(1-mask)
        # lcp_mix=lcp_ori*(1-mask*degree)+lcp_pred*mask*degree
        # mask=torch.ones_like(lcp_ori).cuda()
        # b,c,h,w=mask.shape
        # mask[:,:,int(h/5*4):]=0
        # lcp_mix_c=lcp_mix.clone()
        # lcp_mix_c[:]=0.515
        # lcp_pred_flatten = create_dist_grid((lcp_mix).flatten(),  self.n_e)
        # lcp_ori_pred_flatten = create_dist_grid((lcp_ori).flatten(), self.n_e)
        # tmp=self.light_vector.clone()
        # tmp[:]=0
        # tmp[0]=(degree)

        fq3_after, idexs = self.vq3.forward_vector(fq3,self.light_vector*degree)
        # tv.utils.save(fq3_after)
        # il_map=self.proj(torch.max(img_d2,dim=1,keepdim=True)[0])
        # il_map_inv=self.proj(1-torch.max(img_d2,dim=1,keepdim=True)[0])

        # m_shift=self.est(torch.cat([fq3_after,il_map],dim=1))
        # v_shift=self.est_m(torch.cat([fq3_after,il_map],dim=1))
        #
        # m_shift_inv = self.est(torch.cat([fq3_after, il_map_inv], dim=1))
        # v_shift_inv = self.est_m(torch.cat([fq3_after, il_map_inv], dim=1))

        # rot_input=self.pool(fq3_after).squeeze(-1).squeeze(-1)
        # tv.utils.save_image(torch.sum(lcp_ori,dim=1,keepdim=True),'/data/wuhongjun/project/segv2/Test_seg_lcp_2illu_idex_mask_enhanced_and_nomask_ori/lcp_ori6.png')
        # tv.utils.save_image(torch.sum(lcp_pred,dim=1,keepdim=True),'/data/wuhongjun/project/segv2/Test_seg_lcp_2illu_idex_mask_enhanced_and_nomask_ori/lcp_pred6.png')
        # tv.utils.save_image(torch.sum(lcp_mix+mask,dim=1,keepdim=True),'/data/wuhongjun/project/segv2/Test_seg_lcp_2illu_idex_mask_enhanced_and_nomask_ori/lcp_mix6.png')

        # fq3_after_ori, codebook_loss3, distance_map, [_, _], idexs = self.vq3(fq3, lcp_ori_pred_flatten, [0, 0], tau=1,test=False)

        # fq3_after = fq3_after_ori#fq3_after+((self.adj_m(fq3_after))) * (1-lcp_pred) #+ self.adj_m(fq3_after) * (lcp_ori*mask)

        # fq3_after = fq3_after+((self.adj_m(fq3_after.permute(0, 2, 3, 1))).permute(0, 3, 1, 2))*mask+((self.adj_m_inv(fq3_after.permute(0, 2, 3, 1))).permute(0, 3, 1, 2))*(1-mask)
        # fq3_after=(fq3_after*(1+v_shift)+m_shift)*mask+(fq3_after*(1+v_shift_inv)+m_shift_inv)*(1-mask)
        # fq3_after = fq3_after + ((self.adj_m(fq3_after))) * (1 - lcp_pred)

        # fq3_after = fq3_after + ((self.adj_m(fq3_after)))

        r3_rec = self.conv_r3(fq3_after)
        #
        # gra1 = self.gradient.construct_gt(x)
        # gra2 = self.gradient.construct_gt(F.interpolate(x, scale_factor=0.5, mode='bicubic'))
        #
        # r3_rec_up2 = F.interpolate(img_d2, scale_factor=2, mode='bicubic')
        # r2_rec = self.conv_l2(torch.cat([r3_rec_up2, r2, gra2], dim=1))
        #
        # r3_rec_up4 = F.interpolate(img_d2, scale_factor=4, mode='bicubic')
        # r1_rec = self.conv_l1(torch.cat([r3_rec_up4, r1, gra1], dim=1))

        result = self.up_lplas(r1, r2, r3_rec)
        # mask_pred = self.pred_mask(result, gt)

        return result


    def forward_pixel(self, x,gt,mask,test=False,use_gt_lcp=False,use_gt_high=False,degree=0.):

        r1, r2, img_d2 = self.down_lplas(x)
        B, C, H, W = img_d2.shape
        gt_r1, gt_r2, gt_img_d2 = self.down_lplas(gt)

        layer_f3 = self.conv_l3(img_d2)
        mask = F.interpolate(mask, size=[layer_f3.shape[-2], layer_f3.shape[-1]], mode='bicubic')
        # mask[:]=0
        fq3 = layer_f3
        lcp_ori = get_light_channel(img_d2, 7).unsqueeze(1)
        lcp_pred = self.est_lcp(get_light_channel(img_d2, 7).unsqueeze(1))
        # for i in range(50):
        #     mask=self.dilate_mask(mask)
        # lcp_mix=mask*degree#(1-(lcp_ori+(lcp_pred-lcp_ori)*degree))*(mask)+lcp_ori*(1-mask)#mask#mask+lcp_ori*(1-mask)
        lcp_mix = lcp_pred * (1 - mask * degree) +  lcp_ori* mask * degree
        # mask=torch.ones_like(lcp_ori).cuda()
        # b,c,h,w=mask.shape
        # mask[:,:,int(h/5*4):]=0
        # lcp_mix_c=lcp_mix.clone()
        # lcp_mix_c[:]=0.515
        lcp_pred_flatten = create_dist_grid((lcp_mix).flatten(), self.n_e)
        # lcp_ori_pred_flatten = create_dist_grid((lcp_ori).flatten(), self.n_e)
        fq3_after, codebook_loss3, distance_map, [_, _], idexs = self.vq3(fq3, lcp_pred_flatten, [0, 0], tau=1,
                                                                          test=False)

        fq3_after = fq3_after + ((self.adj_m(fq3_after))) * (1 - lcp_pred)

        # fq3_after = fq3_after+((self.adj_m(fq3_after.permute(0, 2, 3, 1))).permute(0, 3, 1, 2))*mask+((self.adj_m_inv(fq3_after.permute(0, 2, 3, 1))).permute(0, 3, 1, 2))*(1-mask)
        # fq3_after=(fq3_after*(1+v_shift)+m_shift)*mask+(fq3_after*(1+v_shift_inv)+m_shift_inv)*(1-mask)
        r3_rec = self.conv_r3(fq3_after)

        r3_rec=img_d2*(1-mask*degree)+r3_rec*mask*degree

        #
        gra1 = self.gradient.construct_gt(x)
        gra2 = self.gradient.construct_gt(F.interpolate(x, scale_factor=0.5, mode='bicubic'))
        #
        r3_rec_up2 = F.interpolate(img_d2, scale_factor=2, mode='bicubic')
        r2_rec = self.conv_l2(torch.cat([r3_rec_up2, r2, gra2], dim=1))

        r3_rec_up4 = F.interpolate(img_d2, scale_factor=4, mode='bicubic')
        r1_rec = self.conv_l1(torch.cat([r3_rec_up4, r1, gra1], dim=1))

        result = self.up_lplas(r1_rec, r2_rec, r3_rec)


        return result, torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda(), [r3_rec]
