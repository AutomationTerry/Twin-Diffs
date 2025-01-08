import torch
import glob
import os
import numpy as np
import json
import vector

from tqdm import tqdm
from termcolor import colored
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from diffusers import StableDiffusionPipeline


# TODO: 渲染8视图  固定仰角为 30°，方位角从 -180° 到 180° 均匀分布。
# image to rgb
# ? FIXME: mv2latants vae, normal map , depth map
class TwinDiffusionMVDataset(Dataset):
    def __init__(self, cfg, weight_dtype) -> dict:
        self.cfg = cfg
        self.multi_views = cfg.data.get("multi_views", int(8))
        self.model = cfg.model
        self.instance_file = cfg.data.instance_file
        self.multi_view_file = cfg.data.multi_view_file
        self.depth_file = cfg.data.get("depth_file", None)
        self.normal_file = cfg.data.get("normal_file", None)
        self.images_file = cfg.data.images_file
        self.elevation = cfg.data.get("elevation", 30)
        self.multi_view_azimuths = np.linspace(
            0, 360, self.multi_views, endpoint=False
        )  # np数组
        with open(self.instance_file, "r") as f:
            self.instance_list = [fn.strip() for fn in f.readlines()]

        self.rebuild = cfg.data.rebuild
        self.invalid_file_2d = cfg.data.invalid_file_2d
        self.image_size = cfg.data.get("image_size", int(512))
        self.weight_dtype = weight_dtype
        self.camera_info = cfg.data.get("camera_info", None)
        self.camera_distance = cfg.data.get("camera_distance", 2.0)
        self.light_direction = cfg.data.get(
            "light_direction", [0.09387503, -0.63953443, -0.7630093]
        )

        self.multi_view_list = glob.glob(
            os.path.join()(cfg.data.root + "/*.glb")
        )  # 获取所有的ply文件的路径
        self.multi_view_list = [
            p.split(".")[-2].split("/")[-1] for p in self.multi_view_list
        ]  # 获取所有的ply文件的文件名作为id
        print("Loading multi-view images...")

        invalid_list = []
        images = {}
        assert isinstance(self.multi_views, int), "multi_views must be an integer"
        if self.rebuild or not os.path.exists(self.images_file):
            for id in tqdm(self.instance_list, total=len(self.instance_list)):
                if id not in self.multi_view_list:
                    print(f"{id} not found in the multi_views dataset")
                    invalid_list.append(id)
                    continue
                image_path = os.path.join(self.images_file, f"{id}.npz")
                if not os.path.exists(image_path):
                    images_attribute = self.load_multi_view_images(id)
                    camera_info = self.load_camera_info(id)
                    images = {
                        "id": id,
                        "images_attribute": images_attribute,
                        "camera_info": camera_info,
                    }
                    np.savez(image_path, **images)

            with open(self.invalid_file_2d, "w") as f:
                for id in invalid_list:
                    f.write(f"{id}\n")
                    invalid_len = len(invalid_list)
            print(
                colored(
                    f"{invalid_len} invalid multi-views datas are saved in {self.invalid_file_2d}.",
                    "red",
                )
            )
            if invalid_len > 0:
                answer = input(
                    "please check the multi-view images, do you want to terminate the training? (y/n): "
                )
                if answer.lower() == "y":
                    print("Terminating the training...")
                    raise Exception("Invalid multi-view images found.")
                else:
                    print("Continuing the training...")
        print("Multi-view images are ready.")

    def load_multi_view_images(self, id):
        mvimages = []
        depth_images = []
        normal_images = []
        for i in range(self.multi_views):
            mvimage_path = os.path.join(
                self.multi_view_file, f"{id}", "point_clouds", f"{i}.png"
            )  # data/multi_view/xxxx_45.png
            mvimage = Image.open(mvimage_path).convert("RGB")
            # mvimage = mvimage.permute(0,1,4,2,3)  #TODO:check if (b, v, c0, h0, w0)
            mvimage = Image.resize((self.image_size, self.image_size))
            mvimages.append(mvimage)

            # TODO: check encoder(depth and normal)
            """
            if self.depth_file is not None:
                depth_path = os.path.join(self.depth_file, f"{i}.png")
                depth_image = Image.open(depth_path).convert("RGB")
                depth_image = depth_image.resize((self.image_size, self.image_size))
                depth_images.append(depth_image)
            if self.normal_file is not None:
                normal_path = os.path.join(self.normal_file, f"{i}.png")
                normal_image = Image.open(normal_path).convert("RGB")
                normal_image = normal_image.resize((self.image_size, self.image_size))
                normal_images.append(normal_image)
            # vae:"""

        mvimages = torch.stack(mvimages)
        if self.depth_file is not None:
            depth_images = torch.stack(depth_images)
        if self.normal_file is not None:
            normal_images = torch.stack(normal_images)

        transform = transforms.Compose(
            [
                transforms.Resize(self.image_size, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  #!change
            ]
        )
        mvimages = [transform(image) for image in mvimages]

        mvlatents = self.encoder_images(mvimages)
        with open(os.path.join(self.root, "latent.npy"), "wb") as f:
            torch.save(mvlatents, f)

        if self.depth_file is not None:
            depth_tansform = transforms.Compose(
                [
                    transforms.Resize(self.image_size, antialias=True),
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda x: torch.where(x < 0.99, x, torch.zeros_like(x))[0:1]
                    ),
                    transforms.Resize(self.image_size, antialias=True),
                    transforms.Lambda(lambda x: x[0]),
                ]
            )
            depth_images = [depth_tansform(depth_image) for depth_image in depth_images]
        # normal map
        if self.normal_file is not None:
            raise NotImplementedError("Normal map is not implemented yet")
            normal_transform = transforms.Compose(
                [
                    transforms.Resize(self.image_size, antialias=True),
                    # TODO: normal map
                ]
            )
            normal_images = [
                normal_transform(normal_image) for normal_image in normal_images
            ]

        main_view_id = torch.randint(0, self.multi_views, ())  # 随机主视图id
        images_attribute = {
            "mvimages": mvlatents,
            "depth_images": depth_images,
            # "normal_images": normal_images,
            "main_view_id": main_view_id,
        }
        return images_attribute

    def load_camera_info(self, id):
        camera_info = {}
        with open(
            os.path.join(self.camera_info, f"{id}", "camera_info.json"), "r"
        ) as f:
            camera_info_list = json.load(f)
        for i in range(self.multi_views):
            camera_info = camera_info_list[str(i)]
            K = np.array(camera_info["intrinsic_matrix"])
            rotation_matrix = np.array(camera_info["rotation_matrix"]).astype(
                np.float32
            )
            translation_matrix = np.array(camera_info["translation_matrix"]).astype(
                np.float32
            )
            c2w = np.eye(4)
            c2w[:3, :3] = rotation_matrix
            c2w[:3, 3] = translation_matrix
            w2c = np.linalg.inv(c2w).astype(np.float32)
            affine_matrix = np.eye(4)
            affine_matrix[:3, :3] = K @ w2c[:3, :4]

            light_position = vector(self.light_direction)

            camera_info = {
                f"{i}": {
                    "K": K,
                    "c2w": c2w,
                    "w2c": w2c,
                    "affine_matrix": affine_matrix,
                    "light_position": light_position,
                    "light_color": torch.ones(3),
                    "elevation": self.elevation,
                    "azimuth": self.multi_view_azimuths[i],
                }
            }
        return camera_info

    #  ?FIXME：mv2latants vae
    def encoder_images(self, images):
        self.pipe = self.get_pipeline()
        self.vae = self.pipe.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        input_dtype = images.dtype
        latents = self.vae.encode(images.to(self.weight_dtype)).latent_dist
        latents = latents.rsample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    def get_pipeline(self):
        if self.model.name == "stable diffusion v1.5":
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model.root, torch_dtype=self.weight_dtype
            )
        elif self.model.name == "stable diffusion xl":
            raise NotImplementedError("Stable diffusion xl is notimplemented yet")
        return self.pipe

    def update(self, step):
        self.step = step

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, index):
        data = {}
        ids = self.instance_list[index]
        for id in ids:
            images_attribute_path = os.path.join(self.images_file, f"{id}.npz")
            images_attribute = np.load(images_attribute_path)
            data["data_mv"] = images_attribute
        return data
