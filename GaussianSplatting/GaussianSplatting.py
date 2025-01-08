        
import torch
import torch.nn as nn
import numpy as np
import random

from kornia.geometry.conversions import quaternion_to_rotation_matrix,QuaternionCoeffOrder
from einops import repeat

try:
    import _gs as _backend
except:
    raise ImportError("Could not import Gaussian Splatting Renderer backend. Please compile the C++ extension first.")

_timing_ = False

class GaussianSplattingRenderer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device=cfg.device
        self.svec_activation = cfg.model.svec_activation
        self.alpha_activation = cfg.model.alpha_activation
        self.color_activation = cfg.model.color_activation
        self.multi_views=cfg.data.multi_views
        self.depth_detach=cfg.data.depth_detach
        self.mode=cfg.mode
        self.densify_enabled=cfg.model.densify_enabled
        self.masks=[]
        self.mean_2ds=[]
        self.image_size = cfg.data.image_size
        self.tile_size = cfg.tile_size
        self.T_threshold = cfg.T_threshold
        self.rgb_only = cfg.rgb_only

    def forward(self,mean,color,qvec,svec,alpha,camera_info):
        self.camera_info = camera_info
        gs_outputs = []
        for i in range(self.multi_views):
            c2ws = self.camera_info[str(i)]["c2w"].to(self.device)
            gs_outputs.append(self.render_one(c2ws[i],mean,qvec,color,svec,alpha))
        gs_outputs = stack_dicts(gs_outputs)
        
    #得到rgb图像，深度图depth，不透明度opacity，深度方差z_variance    
    def render_one(self, c2w,mean,qvec,color,svec,alpha,return_T=False):  
        N = mean.shape[0]     
        mask = torch.zeros(N,dtype=torch.bool,device=self.device)
        mean = mean[mask].contiguous()
        qvec = qvec[mask].contiguous()
        color = color[mask].contiguous()
        svec = svec[mask].contiguous()
        alpha = alpha[mask].contiguous()
        mean, cov,JW,depth =project_guussians(mean, qvec, svec, alpha,self.depth_detach)
        if self.mode == "train":
            with torch.no_grad():
                m = (cov[...,0,0]+cov[...,1,1])/2.0
                p =torch.det(cov)
                radii2d = m + torch.sqrt(m**2 - p).clamp(min=0)
                self.max_radii2d = torch.max(self.max_radii2d[mask], radii2d)
        if self.mode == "train" and self.densify_enabled:
            mean_2d = mean
            mean_2d.retain_grad()
            self.mean_2ds.append(mean_2d)
            self.masks.append(mask)
        
        tic()
        N_with_dub,aabb_topleft,aabb_bottomright = self.tile_culling_aabb_count(mean,cov,self.image_size,self.tile_size,self.tile_culling_radius)
        toc("count N with dub")
        
        self.total_dub_gaussians = N_with_dub
        
        n_tiles_h = self.image_size // self.tile_size +(self.image_size % self.tile_size > 0)
        n_tiles_w = self.image_size // self.tile_size +(self.image_size % self.tile_size > 0)
        n_tiles = n_tiles_h*n_tiles_w
        img_topleft = torch.FloatTensor([self.camera_info["K"][0,2]/self.camera_info["K"][0,0],
                                         self.camera_info["K"][1,2]/self.camera_info["K"][1,1]]).to(self.device)
        start = -torch.ones([n_tiles],dtype=torch.int32,device=self.device)
        end= -torch.ones([n_tiles],dtype=torch.int32,device=self.device)
        pixel_size_x =1.0/self.camera_info["K"][0,0]
        pixel_size_y =1.0/self.camera_info["K"][1,1]
        gaussian_ids = torch.zeros([N_with_dub],dtype=torch.int32,device=self.device)
        
        tic()
        #TODO:C++编写的后端模块_backend
        _backend.title_culling_aabb_start_end(aabb_topleft,aabb_bottomright,gaussian_ids,start,end,depth,n_tiles_h,n_tiles_w)
        toc("tile culling aabb")
        
        tic()
        T=torch.ones([self.image_size,self.image_size,1],dtype=torch.int32,device=self.device)
        rays_d=self.get_rays_d(c2w)
        render_with_T=_render_with_T.apply
        out=render_with_T(mean,cov,color,svec,alpha,start,end,gaussian_ids,img_topleft,self.tile_size,n_tiles_h,n_tiles_w,pixel_size_x,pixel_size_y,self.image_size,self.T_threshold,T,self.bg(rays_d)).view(self.image_size,self.image_size,3) 
        toc("render with T")
        
        output={}
        output["rgb"]=out
        if not self.rgb_only:
            tic()
            render_scalar = _render_scalar.apply
            rendered_depth = render_scalar(mean,cov,depth,alpha,start,end,gaussian_ids,img_topleft,self.tile_size,n_tiles_h,n_tiles_w,pixel_size_x,pixel_size_y,self.image_size,self.T_threshold,T).reshape(self.image_size,self.image_size,1)
            toc("render depth")
            
            tic()
            scalar = torch.ones_like(mean.data[...,0])
            opacity = render_scalar(mean,cov,scalar,alpha,start,end,gaussian_ids,img_topleft,self.tile_size,n_tiles_h,n_tiles_w,pixel_size_x,pixel_size_y,self.image_size,self.T_threshold,T).reshape(self.image_size,self.image_size,1)
            toc("render opacity")
            
            tic()
            z2 = depth*depth
            z2 = render_scalar(mean,cov,z2,alpha,start,end,gaussian_ids,img_topleft,self.tile_size,n_tiles_h,n_tiles_w,pixel_size_x,pixel_size_y,self.image_size,self.T_threshold,T).reshape(self.image_size,self.image_size,1)
            z_variance = z2 - rendered_depth**2
            toc("render z_variance")
            
            if not (self.mode == "train"):
                out = out.clamp(0,1)
            
            output.update({"depth":rendered_depth,"opacity":opacity,"z_variance":z_variance})
            if return_T:
                output["T"]=T
        return output
         
    def get_rays_d(self,c2w):
        xp =(torch.arange(0,self.image_size,dtype=torch.float32)-self.camera_info["K"][0,2])/self.camera_info["K"][0,0]
        yp =(torch.arange(0,self.image_size,dtype=torch.float32)-self.camera_info["K"][1,2])/self.camera_info["K"][1,1]
        xp,yp = torch.meshgrid(xp,yp,indexing='ij')
        xp = xp.reshape(-1)
        yp = yp.reshape(-1)
        padding = torch.ones_like(xp)
        xyz_cam = torch.stack([xp,yp,padding],dim=-1).to(c2w.device)
        rot=c2w[:3,:3]
        return (torch.einsum("ij,bj->bi",rot,xyz_cam).reshape(self.image_size,self.image_size,3).transpose(0,1)).to(c2w.device)

    @torch.no_grad()          
    def tile_culling_aabb_count(self,mean:torch.TensorType["N",2],cov:torch.TensorType["N",2,2],image_size:512,tile_size:int,D:float):
        centers = mean
        aabb_x = torch.sqrt(D*cov[:,0,0])
        aabb_y = torch.sqrt(D*cov[:,1,1])
        aabb_sidelength = torch.stack([aabb_x,aabb_y],dim=-1)
        aabb_topleft = centers - aabb_sidelength
        aabb_bottomright = centers + aabb_sidelength
        topleft_pixels = self.camera_space_to_pixel_space(aabb_topleft)
        bottomright_pixels = self.camera_space_to_pixel_space(aabb_bottomright)
        topleft_pixels[...,0].clamp_(min=0,max=image_size-1)
        topleft_pixels[...,1].clamp_(min=0,max=image_size-1)
        bottomright_pixels[...,0].clamp_(min=0,max=image_size-1)
        bottomright_pixels[...,1].clamp_(min=0,max=image_size-1)
        topleft_pixels = torch.div(topleft_pixels,tile_size,rounding_mode='floor')
        bottomright_pixels = torch.div(bottomright_pixels,tile_size,rounding_mode='floor')
        N_with_dub = (torch.prod(bottomright_pixels-topleft_pixels+1,dim=-1)).sum().items()
        return N_with_dub,aabb_topleft,aabb_bottomright

    def camera_space_to_pixel_space(self,pts):
        if pts.shape[1]==3:
            pts = pts[:,:2]/pts[:,2:]
        assert pts.shape[1]==2
        pts[:,0]=pts[:,0]*self.camera_info["K"][0,0]+self.camera_info["K"][0,2]
        pts[:,1]=pts[:,1]*self.camera_info["K"][1,1]+self.camera_info["K"][1,2]
        if isinstance(pts,np.ndarray):
            pts = pts.astype(np.int32)
        elif isinstance(pts,torch.Tensor):
            pts = pts.to(torch.int32)
            
    def setup_bg(self,cfg):
        bg_type = cfg.background.type
        if bg_type == "random":
            self.bg=RandomBackground(cfg.background)
        else:
            raise NotImplementedError(f"Background type {bg_type} not implemented")
         

def tic(): 
    import time
    if "_timing_" not in globals() or not _timing_:
        return
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()
    
def toc(msg=""):
    import time
    if "_timing_" not in globals() or not _timing_:
        return
    global startTime_for_tictoc
    if "startTime_for_tictoc" in globals():
        print(f"{msg}: {time.time() - startTime_for_tictoc}")
    else:
        print(f"{msg}: toc() called without tic().")

def project_guussians(
    mean:torch.TensorType["N",3],
    qvec:torch.TensorType["N",4],
    svec:torch.TensorType["N",3],
    c2w:torch.TensorType[3,4],
    detach_depth:bool=False,
    ) :
    projected_mean = project_pts(mean, c2w)
    rotmat=qsvec2rotmat_batched(qvec, svec)
    sigma = rotmat @torch.transpose(rotmat,-1,-2)
    W = torch.transpose(c2w[:3,:3],-1,-2)
    J = jacobian(projected_mean)
    JW = torch.einsum("bij,jk->bik",J,W)
    projected_cov =torch.bmm(torch.bmm(JW,sigma),torch.transpose(JW,-1,-2))[...,:2,:2].contiguous()
    depth=projected_mean[...,2:].clone().contiguous()
    projected_mean = (projected_mean[...,:2] / (projected_mean[...,2:])).contiguous()
    return projected_mean, projected_cov, JW, depth
    
    
def project_pts(pts: torch.Tensor, c2w: torch.TensorType["N", 3, 4]):
    d = -c2w[..., :3, 3]
    W = torch.transpose(c2w[..., :3, :3], -1, -2)
    return torch.einsum("ij,bj->bi", W, pts + d)

def qsvec2rotmat_batched(qvec: torch.TensorType["N", 4], svec: torch.TensorType["N", 3]):
    unscaled_rotmat = quaternion_to_rotation_matrix(qvec,QuaternionCoeffOrder.WXYZ)
    rotmat = svec.unsqueeze(-2)* unscaled_rotmat
    return rotmat

def jacobian(self,u):
    l = torch.norm(u,dim=-1)
    J =torch.zeros(u.shape(0),3,3).to(u)
    J[...,0,0]=1.0/u[...,2]
    J[...,2,0]=u[...,0]/1
    J[..., 1, 1] = 1.0 / u[..., 2]
    J[..., 2, 1] = u[..., 1] / l
    J[..., 0, 2] = -u[..., 0] / u[..., 2] / u[..., 2]
    J[..., 1, 2] = -u[..., 1] / u[..., 2] / u[..., 2]
    J[..., 2, 2] = u[..., 2] / l
    print(torch.det(J), "det(J)")

class _render_scalar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean, cov,scalar, alpha, start, end, gaussian_ids, topleft, tile_size, n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, image_size, threshold, T):
        out = torch.zeros([image_size*image_size], dtype=torch.float32, device=mean.device)
        #TODO:C++编写的后端模块_backend
        _backend.tile_based_vol_rendering_scalar(mean, cov, scalar, alpha, start, end, gaussian_ids, out, topleft, tile_size, n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, image_size, threshold, T)
        ctx.save_for_backward(mean, cov, scalar, alpha, start, end, gaussian_ids, out, topleft)
        ctx.const=[tile_size, n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, image_size, threshold]
        return out
    
    @staticmethod
    def backward(ctx, grad):
        (mean, cov, color, alpha, start, end, gaussian_ids, out, topleft) = ctx.saved_tensors
        grad_mean = torch.zeros_like(mean)
        grad_cov = torch.zeros_like(cov)
        grad_color = torch.zeros_like(color)
        grad_alpha = torch.zeros_like(alpha)
        (tile_size, n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, image_size, threshold) = ctx.const
        
        tic()
        #TODO:C++编写的后端模块_backend
        _backend.tile_based_vol_rendering_scalar_backward(mean, cov, color, alpha, start, end, gaussian_ids, out, grad_mean, grad_cov, grad_color, grad_alpha, grad, topleft, tile_size, n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, image_size, threshold)
        toc("tile_based_vol_rendering_scalar_backward")
        
        return (grad_mean, grad_cov, grad_color, grad_alpha, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
        
class _render_with_T(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean, cov, scalar, alpha, start, end, gaussian_ids, topleft, tile_size, n_tiles_h, n_tiles_w, pixel_size_x, pixel_size_y, image_size, threshold, bg):
        out=torch.zeros([image_size,image_size,3],dtype=torch.float32,device=mean.device)
        T=torch.ones_like(out[...,:1])
        #TODO:C++编写的后端模块_backend
        _backend.tile_based_vol_rendering_start_end_with_T(mean,cov,scalar,alpha,start,end,gaussian_ids,out,topleft,tile_size,n_tiles_h,n_tiles_w,pixel_size_x,pixel_size_y,image_size,threshold,T)
        
        if torch.isnan(out).any():
            print("nan in out")
        out = out+T*bg
        if torch.isnan(bg).any():
            print("nan in bg")
        if torch.isnan(T).any():
            print("nan in T")
        ctx.save_for_backward(mean,cov,scalar,alpha,start,end,gaussian_ids,out,topleft,T)
        ctx.const=[tile_size,n_tiles_h,n_tiles_w,pixel_size_x,pixel_size_y,image_size,threshold]
        if torch.isnan(out).any():
            print("nan in out")
        return out
    
    @staticmethod
    def backward(ctx, grad):
        (mean,cov,color,alpha,start,end,gaussian_ids,out,topleft,T)=ctx.saved_tensors
        grad = grad.contiguous()
        grad_mean = torch.zeros_like(mean)
        grad_cov = torch.zeros_like(cov)
        grad_color = torch.zeros_like(color)
        grad_alpha = torch.zeros_like(alpha)
        (tile_size,n_tiles_h,n_tiles_w,pixel_size_x,pixel_size_y,image_size,threshold)=ctx.const
        
        tic()
        #TODO:C++编写的后端模块_backend
        _backend.tile_based_vol_rendering_backward_start_end(mean,cov,color,alpha,start,end,gaussian_ids,out,grad_mean,grad_cov,grad_color,grad_alpha,grad,topleft,tile_size,n_tiles_h,n_tiles_w,pixel_size_x,pixel_size_y,image_size,threshold)
        toc("tile_based_vol_rendering_backward")
        return (grad_mean,grad_cov,grad_color,grad_alpha,None,None,None,None,None,None,None,None,None,None,None,None,torch.nan_to_num(grad*T))

class RandomBackground(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.device=cfg.device
        self.cfg=cfg
        self.random_augmentation = cfg.random_augmentation
        self.random_augmentation_prob = cfg.random_augmentation_prob
        self.range = cfg.range
    
    def get_bg(self, dirs):
        if self.cfg.mode == "train":
            color = torch.rand(3)
        else:
            color = torch.zeros(3)
        bg=repeat(color.to(dirs)*(self.range[1]-self.range[0])+self.range[0],"c->h w c",h=dirs.shape[0],w=dirs.shape[1])
        
        return bg
    
    def forward(self,dirs):
        if not self.random_augmentation or (not self.training):
            return self.get_bg(dirs)
        else:
            if random.random() < self.random_augmentation_prob:
                return self.get_bg(dirs)
            else:
                return repeat(torch.zeros(3).to(dirs),"c->h w c",h=dirs.shape[0],w=dirs.shape[1])
        
def stack_dicts(dicts):
    stack_dicts = {}
    for key in dicts[0].keys():
        stack_dicts[key] = torch.stack([d[key] for d in dicts],dim=0)
    return stack_dicts