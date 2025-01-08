import torch
import numpy as np
import torch.nn as nn
import torchsparse as spnn
import torch.nn.functional as F

from twin_diffusions_main.twin_diffusions import ConvBnReLu
from inplace_abn import InPlaceABN
from torch.nn.functional import grid_sample
from diffusers import DDPMScheduler
from Diffusion3d.unet.unet import UNetModel
from torchsparse.tensor import SparseTensor
from einops import repeat


class SparseDiffusion(nn.Module):
    def __init__(self,batch, cfg,):
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
        self.num_points = cfg.data.get("num_points", 1024)
        self.num_coords = cfg.data.get("num_coords", 32) #16*16*16=4096 32*32*32=32768 96*96*96=884736
        self.voxel_dim =torch.tensor([self.num_coords,self.num_coords,self.num_coords])
        #self.origin = cfg.data.get("origin", [0, 0, 0])
        #self.voxel_size = cfg.data.get("voxel_size", 0.7/95)
        self.timebedding_channels = cfg.model_3d.get("timebedding_channels", 320)
        
        self.compress_layer = ConvBnReLu(self.ch_in,self.feature_compress,3,1,1,norm_act = InPlaceABN)
        
        attention_ds=[]
        for i in [4,8]:
            attention_ds.append(self.num_coords //int(i))
        self.color_net =UNetModel(image_size=self.num_coords,base_channels=32,dim_mults=(1,2,4,8,8),dropout=0.0,use_sketch_condition=False,use_text_condition=True,kernel_size=2.0,world_dims=3,
                                  num_heads=4,vit_global=False,vit_local=True,attention_resolutions=tuple(attention_ds),with_attention=True,verbose=False,text_condition_dim=self.timebedding_channels,
                                  image_condition_dim=56)
        
        if batch is not None:
           self.batch=self.initialize_3d(batch)
           
        sparse_ch_in = (self.feature_compress *2) +16
        sparse_ch_out = cfg.model_3d.get("sparse_ch_out", 8)
        self.sparsenet = SparseNet(d_in=sparse_ch_in,d_out=sparse_ch_out,temb=self.timebedding_channels)
        
    
    def get_3d(self,features,count,timestep):
        b,nv,_,size,_ = features.shape #[1,8,56,64,64]
        t=timestep[0].view([b,nv]) #[1,8]
        min_nv = np.min([1,nv-1])
        feature = self.compress_layer(features[0]) #[8,16,64,64]
        KRcam = self.affine_matrix[count].float #[8,1,4,4]
        coords =self.generate_grid(self.num_coords,1)[0]
        coords = coords.view(3,-1).to(self.device) #[3,96*96*96] 
        up_coords = []
        for i in range(b):
            up_coords.append(torch.cat([torch.ones(1,coords.shape[-1]).to(coords.device)*b,coords]))
        up_coords = torch.stack(up_coords,dim=1).permute(1,0).contiguous() #[96*96*96,4]
        
        mask= self.backproject(up_coords,feature,KRcam,only_mask=True)
        mask = mask.sun(mask,dim=-1) > min_nv
        up_coords = up_coords[mask]
        
        mv_features,mv_mask =self.backproject(up_coords,feature,KRcam,size,only_mask=False) #[num_pts_valid,4]
        volume = self.aggregate_mv_feature(mv_features,mv_mask,up_coords)
        feature_3d = volume #[num_pts_valid,32]
        del mv_features,mv_mask,volume,mask,coords
        
        embedding = self.embbeding[count].float
        noise = torch.randn_like(self.color)
        color_noise_scheduler = DDPMScheduler(num_train_timesteps=1000,beta_schedule="exp",coefficient=-20.0)
        raw_t = t.view([b,nv])[:,0] #[b,] #!t:timestep:[4,8] -> [1,8] raw_t:[1,8]->[1,]
        noisy_color = color_noise_scheduler.add_noise(self.color,noise,raw_t) #[]
        noisy_color = noisy_color.view(b,self.num_coords,self.num_coords,self.num_coords,3).permute(0,4,1,2,3).contiguous() #[1,3,96,96,96]
        noise_level = t /1000
        color_feature = self.color_net(noisy_color,noise_level,None,embedding,None,None,kernel_size=2) #[1,16,96,96,96]
        color_feature = color_feature.permute(0,2,3,4,1).reshape(-1,16) #[96*96*96*16/16,16]
        feature_3d = torch.cat([feature_3d,color_feature],dim=-1) #[96*96*96,32+16]
        r_coords = up_coords[:,[1,2,3,0]] #[96*96*96,4]
        sparse_feature = SparseTensor(feature_3d,r_coords.to(torch.int32)) #[96*96*96,48]
        feature_3d = self.sparsenet(sparse_feature,embedding)
        dense_volume,mask_volume = self.sparse_to_dense(up_coords[:,1:],feature_3d,self.voxel_dim,interval=1,device=None)
        
        out=[]
        out["dense_volume"] = dense_volume
        out["mask_volume"] = mask_volume
        out["visible_volume"] = mask_volume
        out["coords"] = self.generate_grid(self.voxel_dim,1).to(self.device)
        
        xyz = self.mean[count]
        rgb = self.color[count].detach()
        x=torch.cat([xyz,rgb],dim=-1)
        x =repeat(x,"L C -> B L C",B =b) #! 检查要对dense_volume做什么处理
        
        
        loss_3d = 0.5*F.mse_loss(x,dense_volume,reduction="sum")/b
        out["loss_3d"] = loss_3d
        
        return out
                
    def initialize_3d(self,batch):
        data_mv = batch["data_mv"]
        assert data_mv is not None , "data_mv is None"
        data_3d = batch["data_3d"]
        assert data_3d is not None , "data_3d is None"
        data_pm = batch["data_pm"]
        assert data_pm is not None , "data_pm is None"
        
        self.mean =batch.get("data_3d",{}).get("gs_attribute",{}).get("mean",None) #self.mean是数据
        assert self.mean is not None , "mean is None"
        self.color = batch.get("data_3d",{}).get("gs_attribute",{}).get("color",None)
        assert self.color is not None , "color is None"
        self.affine_matrix = batch.get("data_mv",{}).get("camera_info",{}).get("camera_info",{}).get("affine_matrix",None)
        assert self.affine_matrix is not None , "affine_matrix is None"
        self.color=batch.get("data_3d",{}).get("gs_attribute",{}).get("color",None)
        assert self.color is not None , "color is None"
        self.embbeding = batch.get("data_pm",{}).get("prompts_attribute",{}).get("embedding",None)
        assert self.embbeding is not None , "embbeding is None"
        
    def generate_grid(self,interval=1):
        with torch.no_grad():
            grid_range = [torch.arange(0,self.num_coords[axis],interval) for axis in range(3)]
            grid = torch.stack(torch.meshgrid(grid_range[0],grid_range[1],grid_range[2])) #[3,96,96,96]
            grid = grid.unsqueeze(0).type(torch.float32)
        return grid 

    def backproject(self,coords,feature,KRcam,size,only_mask=False):
        nv,b,c,size,_ = feature.shape #[8,1,16,64,64]
        device = feature.device
        feature_3d = torch.zeros((coords.shape[0],nv,c),device=device) #[96*96*96,8,16]
        mask_all = torch.zeros((coords.shape[0],nv),device=device) #[96*96*96,8]
        
        for i in range(b):
            num_points = torch.nonzero(coords[:,0]==i).squeeze(1) #[96*96*96]
            coords_batch = coords[num_points][:,1:] #[96*96*96,3]
            coords_batch = coords.view(-1,3) #[96*96*96,3]
            #origin = torch.tensor(self.origin).unsqueeeze(0).to(device) #[1,3]
            feature_batch = feature[:,i] #[8,16,64,64]
            project_batch = KRcam[:,i] #[8,4,4]
            #grid_batch = coords_batch*self.voxel_size +origin.float #[96*96*96,3]
            rs_grid = coords_batch.unsqueeze(0).expend(nv,-1,-1) #[8,96*96*96,3]
            rs_grid = rs_grid.permute(0,2,1).contiguous() #[8,3,96*96*96]
            NV = rs_grid.shape[-1]
            rs_grid =torch.cat([rs_grid,torch.ones([nv,1,NV]).to(device)],dim=1) #[8,4,96*96*96]
            
            im_p = project_batch @ rs_grid #[8,4,96*96*96]
            im_x,im_y,im_z = im_p[:,0],im_p[:,1],im_p[:,2] #[8,96*96*96]
            im_x = im_x/im_z
            im_y = im_y/im_z
            im_grid = torch.stack([2*im_x/(size-1)-1,2*im_y/(size-1)-1],dim=-1) #[8,96*96*96,2]
            mask = im_grid.abs() <=1
            mask=(mask.sum(dim=-1)==2) & (im_z<0) #[8,96*96*96]
            mask = mask.view(nv,-1)
            mask = mask.permute(1,0).contiguous() #[96*96*96,8]
            mask_all[num_points] = mask.to(torch.int32)
            
            if only_mask:
                return mask_all
            
            feature_batch = feature_batch.view(nv,c,size,size) #[8,16,64,64]
            im_grid = im_grid.view(nv,1,-1,2) #[8,1,96*96*96,2]
            features = grid_sample(feature_batch,im_grid,padding_mode="zeros",align_corners=True) #[num_pts_valid,8,16]
            features = features.view(nv,c,-1) #[8,16,num_pts_valid]
            features =  features.permute(2,0,1).contiguous()
            feature_3d[num_points] = features
        return feature_3d,mask_all    
    
    def aggregate_mv_feature(self,features,mask):
        num_points,nv,c = features.shape #[num_pts_valid,8,16]
        nv_valid = torch.sum(mask,dim=1,keepdim=False) #[num_pts_valid]
        assert torch.all(nv_valid) > 0 , "nv_valid is less than 0"
        volume_sum = torch.sum(features,dim=1,keepdim=False) #[num_pts_valid,16]
        volume_sq_sum = torch.sum(features**2,dim=1,keepdim=False)
        del features
        
        nv_valid = 1./(nv_valid+1e-6)
        costvar=volume_sq_sum * nv_valid[:,None] - (volume_sum*nv_valid[:,None])**2 #[num_pts_valid,16]
        costvar_mean = torch.cat([costvar,volume_sum*nv_valid[:,None]],dim=1) #[num_pts_valid,32]
        del volume_sum,volume_sq_sum
        
        return costvar_mean
    
    def sparse_to_dense(self,coords,feature,interval=1):
        device=feature.device
        coords_int = (coords/interval).to(torch.int64)
        voxel_dim = (self.voxel_dim/interval).to(torch.int64)
        
        dense_volume = sparse_to_dense_channel(coords_int.to(device),feature.to(device),voxel_dim.to(device),feature.shape[1],0,device)
        mask_volume = sparse_to_dense_channel(coords_int.to(device),torch.ones([feature.shape[0],1]).to(feature.device),voxel_dim.to(device),1,0,device)
        
        dense_volume = dense_volume.permute(3,0,1,2).contiguous().unsqueeze(0) 
        mask_volume = mask_volume.permute(3,0,1,2).contiguous().unsqueeze(0) #[1,1,96,96,96]
        
        return dense_volume,mask_volume
        
def sparse_to_dense_channel(coords,features,voxel_dim,channel,default,device):
    coords = coords.to(device)
    dense = torch.full([voxel_dim[0],voxel_dim[1],voxel_dim[2],channel],float(default),device=device)
    if coords.shape[0]>0:
        dense[coords[:,0],coords[:,1],coords[:,2]] = features
    return dense 

class BasicSparseBlock(nn.Module): #X:[num_points_value,48],embedding:[1,320]
    def __init__(self,inc,outc,kernel_size=3,stride=1,dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,outc,kernel_size,stride=stride,dilation=dilation), #[num_points_value,48]->[num_points_value,8]
            spnn.BatchNorm(outc), 
            spnn.ReLU(inplace=True),
        )
    
    def forward(self,x):
        return self.net(x)
    
        
class SparseNet(nn.Module):       
    def __init__(self,d_in,d_out=8,timebedding_channels=320): 
        super().__init__()
        self.d_in=d_in #48
        self.d_out=d_out #8
        self.nonlinearity = nn.GELU() #X:[num_points_value,48],embedding:[1,320
        self.conv0 = BasicSparseBlock(d_in,d_out) #[num_points_value,8]
        
        self.conv1 = BasicSparseBlock(d_out,16,stride=2) #
        self.conv2 = BasicSparseBlock(16,16)
        self.conv2_ = BasicSparseBlock(16,16)
        self.timebedding_projection2 = nn.Linear(timebedding_channels,2*16)
        self.norm2 = nn.GroupNorm(num_groups=8,num_channels=16,eps=1e-6,affine=True)
        
        self.conv3 = BasicSparseBlock(16,32,stride=2)
        self.conv4 = BasicSparseBlock(32,32)
        self.conv4_ = BasicSparseBlock(32,32)
        self.timebedding_projection4 = nn.Linear(timebedding_channels,2*32)
        self.norm4 = nn.GroupNorm(num_groups=16,num_channels=32,eps=1e-6,affine=True)
        
        self.conv5 = BasicSparseBlock(32,64,stride=2)
        self.conv6 = BasicSparseBlock(64,64)
        self.conv6_ = BasicSparseBlock(64,64)
        self.timebedding_projection6 = nn.Linear(timebedding_channels,2*64)
        self.norm6 = nn.GroupNorm(num_groups=16,num_channels=64,eps=1e-6,affine=True)
        
        self.conv7 = BasicSparseBlock(64,32,kernel_size=3,stride=2)
        
        self.conv9 = BasicSparseBlock(32,16,kernel_size=3,stride=2)
        
        self.conv11 = BasicSparseBlock(16,d_out,kernel_size=3,stride=2)
    
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
            
        
        
    