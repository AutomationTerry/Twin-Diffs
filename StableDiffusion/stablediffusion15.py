
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from diffusers import UNet2DConditionModel
from diffusers.models.embeddings import Timesteps,TimestepEmbedding
from diffusers.models.unet_2d_blocks import get_down_block , UNetMidBlock2DCrossAttn
from typing import Tuple, Union
from diffusers import UNet3DConditionModel

class StableDiffusion15(nn.Module):
    _supports_gradient_checkpointing = True
    def __init__(self, cfg,dataset):
        super().__init__()
        self.cfg = cfg
        self.root=cfg.model.root
        self.revision=cfg.model.revision
        self.device=cfg.device
        self.dataset=dataset
        self.multi_views=cfg.data.multi_views
        self.guidance_scale =cfg.model.guidance_scale
        
        self.alphas = self.dataset.noise_schedule.alphas_cumprod.to(self.device)
        
        #TODO: Denoiser3d
               
        self.unet= UNet2DConditionModel.from_pretrained(self.root,subfolder='unet',
                                                   revision =self.revision,
                                                   local_files_only=True)
        self.unet.requires_grad_(False)
    
    def forward(self,rgb,mean,text_embedding,noise,timesteps,noisy_latents):
        assert rgb.shape == self.mv_images.shape, f"rgb shape {rgb.shape} does not match mv_images shape {self.mv_images.shape}"
        assert rgb
        b=rgb.shape[0] #batch size
        self.N = mean.shape[0]
        mv_images_fused = (rgb + self.mv_images) / 2 #TODO:rgb 是不是都是归一化的值
        noisy_latents_fused = self.dataset.noise_schedule.add_noise(mv_images_fused, timesteps, noise)
        with torch.no_grad():
            noise_pred = self.unet(noisy_latents_fused.view(b*self.multi_views,-1,64,64),timesteps,text_embedding)
        noise_pred_text,noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.guidance_scale*(noise_pred_text - noise_pred_uncond)
        w = (1-self.alphas[timesteps]).view(-1,1,1,1)
        grad = w*(noise_pred-noise)
        grad = torch.nan_to_num(grad)
        target = (noisy_latents-grad).detach()
        loss_2d_sds = 0.5*F.mse_loss(noisy_latents,target,reduction="sum")/b
        loss_2d_sds_each = 0.5*F.mse_loss(noisy_latents,target,reduction="none").sum(dim=(1,2,3))
        sd_out = {"loss_2d":loss_2d_sds,"loss_2d_each":loss_2d_sds_each,"grad_norm":grad.norm(),"features":noisy_latents}
        return sd_out,timesteps
        
def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


'''
#2d unet
        conv_in_kernel = 3
        conv_in_padding =(conv_in_kernel-1)//2
        in_channels =4
        block_out_channels= Tuple[int]=(320,640,1280,2560) #TODO:check
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], conv_in_kernel, conv_in_padding)
        
        down_bloack_types = Tuple[str] = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")
        midblock_type = 'UNetMidBlock2DCrossAttn'
        
        flip_sin_to_cos= True
        downscale_freq_shift =int(0)
        if cfg.time_embedding_type == "positional":
            time_embedding_dim = block_out_channels[0]*4 #[b ,4*m]
            self.time_embedding = Timesteps(block_out_channels[0], flip_sin_to_cos,downscale_freq_shift)
            time_input_dim = block_out_channels[0]
        else:
            raise NotImplementedError(f"Time embedding type {cfg.time_embedding_type} not implemented")
        act_fn = "silu"
        post_act_fn = None
        cond_proj_dim = None
        self.time_embedding=TimestepEmbedding(time_input_dim, time_embedding_dim, act_fn, post_act_fn,cond_proj_dim)
        
        #TODO:view_embedding, class_embedding, addition_embedding,controlnet_embedding
        
        if cfg.use_view_embedding:
            raise NotImplementedError("ViewEmbedding not implemented")
        if cfg.use_class_embedding:
            raise NotImplementedError("ClassEmbedding not implemented")
        if cfg.use_addition_embedding:
            raise NotImplementedError("AdditionEmbedding not implemented")
        if cfg.use_controlnet_embedding:
            raise NotImplementedError("ControlNetConditioningEmbedding not implemented")
        
        attention_head_dim = 8 
        attention_head_dim = (attention_head_dim,) * len(down_bloack_types)
        
        cross_attention_dim = 1280 #[b ,4*m]
        cross_attention_dim = (cross_attention_dim,) * len(down_bloack_types)
        
        layer_per_block = Union[int, Tuple[int]] = 2
        layer_per_block = (layer_per_block,) * len(down_bloack_types)
        
        #downsample
        output_channel = block_out_channels[0]
        controlnet_block = nn.Conv2d(output_channel, output_channel, 1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_downblocks = nn.ModuleList([])
        self.controlnet_downblocks.append(controlnet_block)
        self.downblocks = nn.ModuleList([])
        for i ,down_bloack_type in enumerate(down_bloack_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(down_bloack_types) - 1
            down_block = get_down_block(down_bloack_type, 
                                        num_layers=layer_per_block[i],
                                        in_channel=input_channel,
                                        out_channel=output_channel,
                                        temb_channels=time_embedding_dim,
                                        add_downsample= not is_final_block,
                                        resnet_eps=float(1e-5),
                                        resnet_act_fn=act_fn,
                                        resnet_groups=int(32),
                                        cross_attention_dim=cross_attention_dim[i],
                                        attention_head_dim=attention_head_dim[i],
                                        downsample_padding=int(1),
                                        )
            self.downblocks.append(down_block)
            
        for _ in range(layer_per_block[i]):
            controlnet_block = nn.Conv2d(output_channel, output_channel, 1)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_downblocks.append(controlnet_block)
            
        if not is_final_block:
            controlnet_block = nn.Conv2d(output_channel, output_channel, 1)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_downblocks.append(controlnet_block)
            
        #middle
        midblock_channels = block_out_channels[-1]
        controlnet_block = nn.Conv2d(midblock_channels, midblock_channels, 1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_midblock = controlnet_block
        
        if midblock_type == "UNetMidBlock2DCrossAttn":
            midblock = UNetMidBlock2DCrossAttn(in_channels=midblock_channels,
                                               temb_channels=time_embedding_dim,
                                               cross_attention_dim=cross_attention_dim[-1],
                                               attention_head_dim=attention_head_dim[-1],
                                               resnet_eps=float(1e-5),
                                               resnet_act_fn=act_fn,
                                               resnet_groups=int(32),
                                               )
        else:
            raise NotImplementedError(f"Midblock type {midblock_type} not implemented")
        
        device = torch.device(self.device)
        
def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
'''
        
    