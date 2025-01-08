import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.unet_2d_blocks import get_down_block, UNetMidBlock2DCrossAttn
from typing import Tuple, Union
from diffusers.models.controlnet import ControlNetConditioningEmbedding
from diffusers.models.controlnet import ControlNetModel


class ControlStableDiffusion15(ControlNetModel):
    _supports_gradient_checkpointing = True

    def __init__(self, cfg, dataset):
        super().__init__()
        self.cfg = cfg
        self.root = cfg.model.root
        self.revision = cfg.model.revision
        self.device = cfg.device
        self.controlnet_scale = cfg.model.get("controlnet_scale", 1.0)
        self.dataset = dataset
        self.guidance_scale = cfg.model.guidance_scale

        self.alphas = self.dataset.noise_schedule.alphas_cumprod.to(self.device)

        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        in_channels = 4
        block_out_channels = Tuple[int] = (320, 640, 1280, 2560)  # TODO:check
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], conv_in_kernel, conv_in_padding
        )

        # TODO:view_embedding, class_embedding, addition_embedding,controlnet_embedding

        if cfg.use_view_embedding:
            raise NotImplementedError("ViewEmbedding not implemented")
        if cfg.use_class_embedding:
            raise NotImplementedError("ClassEmbedding not implemented")
        if cfg.use_addition_embedding:
            raise NotImplementedError("AdditionEmbedding not implemented")
        if cfg.use_controlnet_embedding:
            raise NotImplementedError("ControlNetConditioningEmbedding not implemented")

        down_block_types = Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        )
        midblock_type = "UNetMidBlock2DCrossAttn"

        attention_head_dim = 8
        attention_head_dim = (attention_head_dim,) * len(down_block_types)

        cross_attention_dim = 1280  # [b ,4*m]
        cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        layer_per_block = Union[int, Tuple[int]] = 2
        layer_per_block = (layer_per_block,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        controlnet_block = nn.Conv2d(output_channel, output_channel, 1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_downblocks = nn.ModuleList([])
        self.controlnet_downblocks.append(controlnet_block)
        self.downblocks = nn.ModuleList([])
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(down_block_types) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=layer_per_block[i],
                in_channel=input_channel,
                out_channel=output_channel,
                temb_channels=None,
                add_downsample=not is_final_block,
                resnet_eps=float(1e-5),
                resnet_act_fn="silu",
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

        # middle
        midblock_channels = block_out_channels[-1]
        controlnet_block = nn.Conv2d(midblock_channels, midblock_channels, 1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_midblock = controlnet_block

        if midblock_type == "UNetMidBlock2DCrossAttn":
            self.midblock = UNetMidBlock2DCrossAttn(
                in_channels=midblock_channels,
                temb_channels=None,
                cross_attention_dim=cross_attention_dim[-1],
                attention_head_dim=attention_head_dim[-1],
                resnet_eps=float(1e-5),
                resnet_act_fn="silu",
                resnet_groups=int(32),
            )
        else:
            raise NotImplementedError(f"Midblock type {midblock_type} not implemented")

        # TODO: Denoiser3d
        self.controlnet_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            conditioning_channels=in_channels,
        )

    def forward(
        self,
        noisy_latents,
        timesteps,
        time_embedding,
        text_embbeding,
        noise,
        unet,
        mv_images_3d=None,
    ):
        b, nv, _, size, _ = noisy_latents.shape  # (4, 8, 3, 64, 64)
        text_embbeding = text_embbeding.repeat(1, nv, 1, 1)  # (4, 8,77, 4096)
        text_embbeding = text_embbeding.reshape(
            b * nv, *text_embbeding.shape[2:]
        )  # (32,77, 4096)
        latents = noisy_latents.reshape(b * nv, -1, size, size)
        latents = self.conv_in(latents)  # (b*nv,320,64,64)
        _, _, _, mv_size, _ = mv_images_3d.shape  # (4, 8, 3, 512, 512)
        if mv_images_3d is not None:
            mv_images_3d = mv_images_3d.reshape(
                b * nv, -1, mv_size, mv_size
            )  #! mv_images_3d:(512,512)
            controlnet_embedding = self.controlnet_embedding(
                mv_images_3d
            )  # (b*nv,320,64,64)
            latents = latents + controlnet_embedding

        # down
        downblock_samples = (latents,)
        for i, downblock in enumerate(self.downblocks):
            if (
                hasattr(downblock, "has_cross_attention")
                and downblock.has_cross_attention
            ):
                sample, res_sample = downblock(
                    hidden_states=latents,
                    temb=time_embedding,
                    encoder_hidden_states=text_embbeding,
                )
            else:
                sample, res_sample = downblock(
                    hidden_states=latents, temb=time_embedding
                )

            downblock_samples = downblock_samples + res_sample

        # mid
        if self.midblock is not None:
            sample = self.midblock(
                sample,
                temb=time_embedding,
                encoder_hidden_states=text_embbeding,
            )

        # control net
        controlnet_downblock_samples = ()
        for downblock_sample, controlnet_downblock in zip(
            downblock_samples, self.controlnet_downblocks
        ):
            downblock_sample = controlnet_downblock(downblock_sample)
            controlnet_downblock_samples = controlnet_downblock_samples + (
                downblock_sample,
            )
        downblock_samples = controlnet_downblock_samples
        midblock_samples = self.controlnet_midblock(sample)

        # scaling
        downblock_samples = [
            sample * self.controlnet_scale for sample in downblock_samples
        ]
        midblock_samples = midblock_samples * self.controlnet_scale

        # fuse
        noise_pred = unet(
            noisy_latents.reshape(b * nv, -1, size, size),
            timesteps,
            encoder_hidden_states=text_embbeding,
            down_block_additional_residuals=[sample for sample in downblock_samples],
            mid_block_additional_residuals=midblock_samples,
        )
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        w = (1 - self.alphas[timesteps]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        target = (noisy_latents - grad).detach()
        loss_2d_sds = 0.5 * F.mse_loss(noisy_latents, target, reduction="sum") / b
        loss_2d_sds_each = 0.5 * F.mse_loss(
            noisy_latents, target, reduction="none"
        ).sum(dim=(1, 2, 3))
        out = {
            "loss_2d": loss_2d_sds,
            "loss_2d_each": loss_2d_sds_each,
            "grad_norm": grad.norm(),
            "features": noisy_latents,
        }
        return out


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
