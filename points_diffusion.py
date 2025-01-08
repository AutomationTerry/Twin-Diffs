import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from twin_diffusions_main.twin_diffusions import ConvBnReLu
from inplace_abn import InPlaceABN
from diffusers import DDPMScheduler
from einops import repeat
from pytorch3d.ops import sample_farthest_points
from diffusers.models.unet_3d_blocks import CrossAttnUpBlock3D
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.attention import Attention

class PointsDiffusion(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.root = cfg.model.root
        self.revision = cfg.model.revision
        self.device = cfg.device
        self.num_views = cfg.data.num_views
        self.batch_size = cfg.batch_size
        self.ch_in = cfg.model_3d.get("ch_in", 3)
        self.feature_compress = cfg.model_3d.get("feature_compress", 16)
        self.num_points = cfg.data.get("num_points", 32768)
        self.num_coords = cfg.data.get("num_coords", 32) #16*16*16=4096 32*32*32=32768 96*96*96=884736
        self.points_dim =torch.tensor([self.num_coords,self.num_coords,self.num_coords])
        self.timebedding_channels = cfg.model_3d.get("timebedding_channels", 320)
        
        self.compress_layer = CompressNet(inc=self.ch_in,outc=self.feature_compress)
        
        self.feature_3d_net = Feature3dNet()
             
        sparse_ch_in = self.feature_compress *2
        sparse_ch_out = cfg.model_3d.get("sparse_ch_out", 6)
        self.denoiser3d = Denoiser3dNet(d_in=sparse_ch_in,d_out=sparse_ch_out,temb=self.timebedding_channels)
        
    def get_3d(self,features,mean,color,embedding,count,timestep):
        b,nv,_,size,_ = features.shape #[1,8,56,64,64]
        xyz = mean[count] #[32768,3]
        rgb = color[count].detach() #[32768,3] 
        x=torch.cat([xyz,rgb],dim=-1) #[32768,6]
        x =repeat(x,"L C -> B L C",B =b) 
        _,indices =farthest_point_sampling(x[...,:3],self.num_points,True) 
        x_=[]
        for x_i ,indices_i in zip(x,indices):
            x_.append(x_i[indices_i])
        x =torch.stack(x_,dim=0)
        x=x.moveaxis(-1,1)
        noise = torch.randn_like(x)
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000,beta_schedule="exp",coefficient=-20.0)
        noisy_x = noise_scheduler.add_noise(x,noise,timestep) 
        feature = self.compress_layer(features[0]) #[8,16,64,64]
        feature = feature.reshape[-1,16] #[32*32*32,16]
        embedding = embedding[count].float 
        feature_3d = self.feature_3d_net(noisy_x,embedding) #[32*32*32,16] #TODO:检查feature_3d_net
        feature_3d = torch.cat([feature_3d,feature],dim=-1) #[32*32*32,32]
        points = self.denoiser3d(feature_3d,embedding) #[32*32*32,6]
        loss_3d = 0.5*F.mse_loss(x,points,reduction="sum")/b
        
        assert points.shape[0] == self.num_coords**3
        out_mean = points[:,:3] #[32*32*32,3]
        out_color = points[:,3:] #[32*32*32,3]
        
        out=[]
        out["points"] = points
        out["loss_3d"] = loss_3d
        out["mean"] = out_mean
        out["color"] = out_color
        return out
            
class CompressNet(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x,inc,outc):
        compress_layer = ConvBnReLu(inc,outc,3,1,1,norm_act = InPlaceABN)
        return compress_layer(x)

class BasicFeature3dBlock(nn.Module):
    def __init__(self,inc,outc,embedding,dropout=0.1):
        super().__init__()
        self.resnet = ResnetBlock2D(inc,outc,emb_dim=embedding.shape[0],dropout=dropout,use_text_condition=True)
        self.cross_attn = CrossAttnUpBlock3D(outc,outc,embedding.shape[0])
        self.attn= nn.Sequential(
            nn.GroupNorm(min(outc//2,8),outc,eps=1e-6,affine=True),
            nn.SiLU(inplace=True),
            Attention(outc,heads=4))
    
    def forward(self,x):
        return self.net(x)

class Feature3dNet(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x,in_out,embedding,dropout): #x:[32768,6] ->[32768,8]
        feature=[]
        for inc,outc in in_out: #in_out:[(6,8),(8,16),(16,32)]
            net= BasicFeature3dBlock(inc,outc,embedding,dropout=dropout)
            x=net(x) 
            feature.append(x)
        feature = torch.cat((feature[0],feature[1],feature[2]),dim=1) #[32768,56]
        feature = CompressNet(feature,feature.shape[1],16) #[32768,16]
        return feature
        
class BasicDenoiser3dBlock(nn.Module): #X:[num_points_value,48],embedding:[1,320]
    def __init__(self,inc,outc,kernel_size=3,stride=1,dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(inc,outc,kernel_size,stride=stride,dilation=dilation), #[num_points_value,48]->[num_points_value,8]
            nn.BatchNorm3d(outc),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,x):
        return self.net(x)
    
        
class Denoiser3dNet(nn.Module):       
    def __init__(self,d_in,d_out=6,timebedding_channels=320): 
        super().__init__()
        self.d_in=d_in #48
        self.d_out=d_out #8
        self.nonlinearity = nn.GELU() #X:[num_points_value,48],embedding:[1,320
        self.conv0 = BasicDenoiser3dBlock(d_in,d_out) #[num_points_value,8]
        
        self.conv1 = BasicDenoiser3dBlock(d_out,16,stride=2) #
        self.conv2 = BasicDenoiser3dBlock(16,16)
        self.conv2_ = BasicDenoiser3dBlock(16,16)
        self.timebedding_projection2 = nn.Linear(timebedding_channels,2*16)
        self.norm2 = nn.GroupNorm(num_groups=8,num_channels=16,eps=1e-6,affine=True)
        
        self.conv3 = BasicDenoiser3dBlock(16,32,stride=2)
        self.conv4 = BasicDenoiser3dBlock(32,32)
        self.conv4_ = BasicDenoiser3dBlock(32,32)
        self.timebedding_projection4 = nn.Linear(timebedding_channels,2*32)
        self.norm4 = nn.GroupNorm(num_groups=16,num_channels=32,eps=1e-6,affine=True)
        
        self.conv5 = BasicDenoiser3dBlock(32,64,stride=2)
        self.conv6 = BasicDenoiser3dBlock(64,64)
        self.conv6_ = BasicDenoiser3dBlock(64,64)
        self.timebedding_projection6 = nn.Linear(timebedding_channels,2*64)
        self.norm6 = nn.GroupNorm(num_groups=16,num_channels=64,eps=1e-6,affine=True)
        
        self.conv7 = BasicDenoiser3dBlock(64,32,kernel_size=3,stride=2)
        
        self.conv9 = BasicDenoiser3dBlock(32,16,kernel_size=3,stride=2)
        
        self.conv11 = BasicDenoiser3dBlock(16,d_out,kernel_size=3,stride=2)
    
    def forward_resblock(self,x,conv,conv_,timebedding_projection,norm,embedding):
        input_tensor=x.F
        hidden_states = input_tensor
        hidden_states = norm(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        x.F = hidden_states
        x = conv(x)
        embedding=self.nonlinearity(embedding)
        timebedding = timebedding_projection(embedding)
        scale,shift = torch.chunk(timebedding,2,dim=1)
        x.F = x.F*(1+scale)+shift
        x.F = self.nonlinearity(x.F)
        x = conv_(x)
        x.F = x.F + input_tensor
        return x
        
    def forward(self,x,embedding): #x:[num_points_value,48],embedding:[1,320]
        b = embedding[0]
        assert b == 1
        conv0 = self.conv0(x) #[num_points_value,8]
        
        conv1 = self.conv1(conv0) 
        conv2=self.forward_resblock(conv1,self.conv2,self.conv2_,self.timebedding_projection2,self.norm2,embedding) 
        
        conv3 = self.conv3(conv2) #[12*12*12,32]
        conv4=self.forward_resblock(conv3,self.conv4,self.conv4_,self.timebedding_projection4,self.norm4,embedding) 
        
        conv5 = self.conv5(conv4) 
        conv6=self.forward_resblock(conv5,self.conv6,self.conv6_,self.timebedding_projection6,self.norm6,embedding)
        
        x=conv4+self.conv7(conv6)
        del conv4,conv6
        x=conv2+self.conv9(x)
        del conv2
        x=conv0+self.conv11(x)
        del conv0
        return x.F
    
@torch.no_grad()       
def farthest_point_sampling(mean:torch.Tensor,K,random_start_point=False):
    if mean.ndim==2:
        L= torch.tensor(mean.shape[0],dtype=torch.long).to(mean.device)
        pts,indices =sample_farthest_points(mean[None,...],L[None,...],K,random_start_point=random_start_point)
        return pts[0],indices[0]
    elif mean.ndim==3:
        B= mean.shape[0]
        L= torch.tensor(mean.shape[1],dtype=torch.long).to(mean.device)
        pts,indices =sample_farthest_points(mean,L[None,...].repeat(B),K,random_start_point=random_start_point)
        return pts,indices
    


    
