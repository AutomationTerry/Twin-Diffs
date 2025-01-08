import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os

from GaussianSplatting.GaussianSplatting import GaussianSplattingRenderer
from inplace_abn import InPlaceABN
from Diffusion3d.diffusion3d import Diffusion3d
from Diffusion3d.points_diffusion import PointsDiffusion
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from StableDiffusion.stablediffusion15_finetune import ControlStableDiffusion15
from scipy.special import logit
from diffusers import UNet2DConditionModel
from diffusers.models.modeling_utils import load_state_dict
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin

# if 3d:
# TODO:twin_diffusions(3dgs renderer -> 2d mvimages, sdxl mvimages,
# TODO:3d feautures(featurenet-> 3d feauture, positional encoding, latent point network),
# TODO:3d sparse diffusions)

# if 2d:
# TODO:twin_diffusions(sdxl mvimages, featurenet -> 3d feauture ,
# TODO:positional encoding ,latent point network)
# TODO:3d sparse diffusions,3dgs renderer -> 2d mvimages)


class TwinDiffusions(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    def __init__(self, cfg, max_steps, dataset):
        super().__init__()
        self.cfg = cfg
        self.root = cfg.model.root
        self.revision = cfg.model.revision
        self.device = cfg.device
        self.max_steps = max_steps
        self.dataset = dataset
        self.num_views = cfg.data.num_views
        self.batch_size = cfg.batch_size
        self.time_embedding_dim_pre = cfg.model.get("time_embedding_dim_pre", 320)
        self.mean_std = cfg.data.get("mean_std", 0.8)
        self.unet_2d_path = cfg.model.get("unet_2d_path", None)
        self.controlnet_path = cfg.model.get("controlnet_path", None)
        self.weight_dtype = cfg.model.get("weight_dtype", "float32")

        self.renderer = GaussianSplattingRenderer(self.cfg).to(self.device)

        self.featurenet = FeatureNet(self.cfg)
        self.diffusion3d = Diffusion3d(self.cfg)
        self.points_diffusion = PointsDiffusion(self.cfg)

        if self.unet_2d_path is not None:
            self.unet_2d = UNet2DConditionModel.from_pretrained(
                self.unet_2d_path, local_files_only=True
            )
            print(f"Loaded 2D UNet from {self.unet_2d_path}")
        else:
            self.unet_2d = UNet2DConditionModel.from_pretrained(
                self.root,
                subfolder="unet",
                revision=self.revision,
                local_files_only=True,
            )

        self.unet_2d.to(self.device, self.weight_dtype)
        self.unet_2d.requires_grad_(False)

        if self.controlnet_path is not None:
            controlnet_path = os.path.join(self.controlnet_path, "controlnet.bin")
            state_dict = load_state_dict(controlnet_path)
            self.control_sd_unet = (
                ControlStableDiffusion15._convert_deprecated_attention_blocks(
                    state_dict
                )
            )
            self.control_sd_unet = ControlStableDiffusion15._load_pretrained_model(
                self.control_sd_unet, state_dict, controlnet_path, self.controlnet_path
            )
            print(f"Loaded ControlNet from {self.controlnet_path}")
        else:
            self.control_sd_unet = ControlStableDiffusion15.from_unet(
                self.unet_2d, self.cfg, self.dataset
            )

        self.control_sd_unet.to(self.device, self.weight_dtype)

    def forward(self, batch, step):
        self.batch = self.initialize(batch)
        self.step = step
        mv_images_3d = []
        if self.timesteps.shape[0] <= 0:
            control_sd_out = self.control_sd_unet(
                self.noisy_latents,
                self.timesteps,
                self.time_embedding,
                self.text_embedding,
                self.noise,
                self.unet_2d,
            )
        elif self.timesteps.shape[0] >= 1 and mv_images_3d is not None:
            control_sd_out = self.control_sd_unet(
                self.noisy_latents,
                self.timesteps,
                self.time_embedding,
                self.text_embedding,
                self.noise,
                self.unet_2d,
                mv_images_3d,
            )
        features = control_sd_out["features"].views(
            self.batch * self.num_views, self.max_steps, self.step, -1
        )
        pyramid_features = self.featurenet.obtain_pyramid_feature_maps(features)
        pyramid_features = pyramid_features.view(
            self.batch, self.num_views, *pyramid_features.shape[1:]
        )  # (B,num_views,32+16+8=56   ,H,W)
        for i in range(self.batch[0]):
            with torch.autocast("cuda", enabled=False):
                feature = pyramid_features[i : i + 1].float
                pt_out = self.points_diffusion.get_3d(
                    feature,
                    self.mean,
                    self.color,
                    self.text_embedding,
                    i,
                    self.timesteps,
                )
                guidance_out = self.diffusion3d(
                    self.mean, self.color, self.timesteps, self.mvimages, self.prompt_3d
                )
                with torch.no_grad():
                    for k, v in pt_out.items():
                        pt_out[k] = v.to(pyramid_features.dtype)
                    for k, v in guidance_out.items():
                        guidance_out[k] = v.to(pyramid_features.dtype)
        gs_output = self.renderer(
            pt_out["mean"],
            pt_out["color"],
            self.qvec,
            self.svec,
            self.alpha,
            self.camera_info,
        )
        with torch.no_grad():
            assert np.all((gs_output["rgb"] < 1.0) & (gs_output["rgb"] > -1.0))
            for i in range(gs_output["rgb"].shape[1]):
                mv_image = self.dataset.get_data_mv.encode_images(
                    batch[i] for batch in gs_output["rgb"]
                )
                mv_images_3d.append(mv_image)  # (512,512)
        loss_2d = control_sd_out["loss_2d_each"]  # [4]
        loss_3d = pt_out["loss_3d"]  # [4]
        loss_3d_guidance = guidance_out["loss_3d"]  # [4]
        for i in range(self.batch[0]):
            losses = loss_2d[i] + loss_3d[i] + loss_3d_guidance[i]

        return control_sd_out, pt_out, guidance_out, losses

    def initialize(self, batch):
        data_mv = batch["data_mv"]
        assert data_mv is not None, "data_mv is None"
        data_3d = batch["data_3d"]
        assert data_3d is not None, "data_3d is None"
        data_pm = batch["data_pm"]
        assert data_pm is not None, "data_pm is None"

        self.affine_matrix = (
            batch.get("data_mv", {})
            .get("camera_info", {})
            .get("camera_info", {})
            .get("affine_matrix", None)
        )
        assert self.affine_matrix is not None, "affine_matrix is None"
        self.mean = (
            batch.get("data_3d", {}).get("gs_attribute", {}).get("mean", None)
        )  # self.mean是数据
        assert self.mean is not None, "mean is None"
        self.qvec = batch.get("data_3d", {}).get("gs_attribute", {}).get("qvec", None)
        assert self.qvec is not None, "qvec is None"

        self.color = batch.get("data_3d", {}).get("gs_attribute", {}).get("color", None)
        assert self.color is not None, "color is None"
        self.color = inv_activations[self.color_activation](self.color)
        self.svec = batch.get("data_3d", {}).get("gs_attribute", {}).get("svec", None)
        assert self.svec is not None, "svec is None"
        self.svec = inv_activations[self.svec_activation](self.svec)
        self.alpha = batch.get("data_3d", {}).get("gs_attribute", {}).get("alpha", None)
        assert self.alpha is not None, "alpha is None"
        self.alpha = inv_activations[self.alpha_activation](self.alpha)

        self.mvimages = (
            batch.get("data_mv", {}).get("images_attribute", {}).get("mvimages", None)
        )
        assert self.mvimages is not None, "mvimages is None"
        main_view_id = (
            batch.get("data_mv", {})
            .get("images_attribute", {})
            .get("main_view_id", None)
        )
        assert main_view_id is not None, "main_view_id is None"

        self.text_embedding = (
            batch.get("data_pm", {})
            .get("prompts_attribute", {})
            .get("embeddings", None)
        )
        assert self.text_embedding is not None, "text_embedding is None"
        self.camera_info = batch.get("data_mv", {}).get("camera_info", None)
        assert self.camera_info is not None, "camero_info is None"

        b, v, c0, h0, w0 = (
            self.mvimages.shape
        )  # (4, 8, 3,64,64) vae:size 512,512 -> 64,64 pixel sapce -> latent space
        self.model_input = self.mvimages
        if h0 > 64:
            self.model_input = F.interpolate(
                self.model_input.view(b * v, c0, h0, w0),
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
            )
            c, h, w = (c0, 64, 64)  # (32，3，64，64)
        else:
            c, h, w = (c0, h0, w0)

        self.clean_images = self.model_input.reshape(b, v, c, h, w)  # (4, 8, 3, 64, 64)

        self.model_input = self.model_input.view(b, v * c, h, w)  # (4, 24, 64, 64)
        self.noise = torch.randn_like(self.model_input)  # (4, 24, 64, 64)
        model_input = self.model_input.view(b, v * c, h, w)  # (4, 24, 64, 64)
        self.noise = torch.randn_like(model_input)  # (4, 24, 64, 64)

        if self.max_steps < self.dataset.noise_schedule.config.num_train_timesteps:
            self.timesteps = torch.randit(
                0, self.max_steps, (b,), device=model_input.device
            )
        else:
            self.timesteps = torch.randint(
                0,
                self.dataset.noise_schedule.config.num_train_timesteps,
                (b,),
                device=model_input.device,
            )
        self.timesteps = self.timesteps.long
        self.timesteps = (
            self.timesteps.unsqueeze(1).repeat(-1, self.multi_views).view(-1)
        )  # (4, 8)

        self.noisy_latents = self.dataset.noise_schedule.add_noise(
            model_input, self.timesteps, self.noise
        )
        self.noisy_latents = self.noisy_latents.view(b, v, c, h, w)  # (4, 8, 3, 64, 64)

        self.prompts = (
            batch.get("data_pm", {}).get("prompts_attribute", {}).get("prompts", None)
        )
        assert self.prompts is not None, "prompts is None"
        self.prompt_3d = [p[0].split(".")[2] for p in self.prompts]
        self.prompt_3d = dict(text=self.prompt_3d * self.batch_size)

        self.time_embedding_dim = self.time_embedding_dim_pre * 4
        self.time_projection = Timesteps(
            self.time_embedding_dim_pre, flip_sin_to_cos=True, freq_shift=0
        )
        self.time_embedding = TimestepEmbedding(
            self.time_embedding_dim_pre,
            self.time_embedding_dim,
            act_fn="silu",
            post_act_fn=None,
            cond_proj_dim=None,
        )


def inv_activations(activations_name):
    min_scale = 1e-3
    softplus_inv_numeric = lambda x: np.log(np.expm1(x))
    if activations_name == "abs":
        return wrapper(np.abs, torch.abs)
    elif activations_name == "nothing":
        return lambda x: x
    elif activations_name == "sigmoid":
        return wrapper(logit, torch.logit)
    elif activations_name == "relu":
        return lambda x: x
    elif activations_name == "exp":
        return wrapper(np.log, torch.log)
    elif activations_name == "biased_relu":
        return lambda x: x - min_scale
    elif activations_name == "biased_abs":
        return lambda x: x - min_scale
    elif activations_name == "softplus_inv":
        return wrapper(softplus_inv_numeric, softplus_inv)
    else:
        raise NotImplementedError(
            f"Activation function {activations_name} not implemented"
        )


def softplus_inv(x):
    return x + torch.log(-torch.expm1(-x))


def wrapper(number_fn, tensor_fn):
    def wrapped(x):
        if isinstance(x, torch.Tensor):
            return tensor_fn(x)
        else:
            return number_fn(x)

    return wrapped


class FeatureNet(nn.Module):
    def __init__(self, cfg, img_ch=3, norm_act=InPlaceABN):
        super().__init__()
        self.img_ch = img_ch * 2
        self.num_views = cfg.data_views

        self.conv0 = nn.Sequential(
            ConvBnReLu(self.img_ch, 8, 3, 1, 1, norm_act=norm_act),
            ConvBnReLu(8, 8, 3, 1, 1, norm_act=norm_act),
        )
        self.conv1 = nn.Sequential(
            ConvBnReLu(8, 16, 5, 2, 2, norm_act=norm_act),
            ConvBnReLu(16, 16, 3, 1, 1, norm_act=norm_act),
            ConvBnReLu(16, 16, 3, 1, 1, norm_act=norm_act),
        )
        self.conv2 = nn.Sequential(
            ConvBnReLu(16, 32, 5, 2, 2, norm_act=norm_act),
            ConvBnReLu(32, 32, 3, 1, 1, norm_act=norm_act),
            ConvBnReLu(32, 32, 3, 1, 1, norm_act=norm_act),
        )
        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)
        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True) + y

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        feat2 = self.toplayer(conv2)
        feat1 = self._upsample_add(self.lat1(conv1), feat2)
        feat0 = self._upsample_add(self.lat0(conv0), feat1)

        feat1 = self.smooth1(feat1)
        feat0 = self.smooth0(feat0)

        return [feat2, feat1, feat0]

    def obtain_pyramid_feature_maps(self, imgs):
        pyramid_features_maps = self(imgs)
        fused_feature_map = torch.cat(
            [
                F.interpolate(
                    pyramid_features_maps[0],
                    scale_factor=4,
                    mode="bilinear",
                    align_corners=True,
                ),
                F.interpolate(
                    pyramid_features_maps[1],
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=True,
                ),
                pyramid_features_maps[2],
            ],
            dim=1,
        )
        return fused_feature_map


class ConvBnReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnReLu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.GroupNorm(
            min(out_channels // 2, 16), out_channels, eps=1e-6, affine=True
        )
        self.act = nn.LeakyReLu(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
