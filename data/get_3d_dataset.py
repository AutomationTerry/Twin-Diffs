import torch
import torch.nn as nn
import glob
import os
import open3d as o3d
import numpy as np

from tqdm import tqdm
from termcolor import colored
from torch.utils.data import Dataset

# TODO:get 3d pcd(xyz,rgb)(3:3) to gs attribute(mean color svec qvec alpha)


class TwinDiffusion3dDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = cfg.model
        self.instance_file = cfg.data.instance_file
        with open(self.instance_file, "r") as f:
            self.instance_list = [fn.strip() for fn in f.readlines()]

        self.pcd_root = cfg.data.pcd_root
        self.rebuild = cfg.data.rebuild
        self.invalid_file = cfg.data.invalid_file
        self.number_of_points = cfg.data.get("number_of_points", 4096)
        self.mean_std = cfg.data.get("mean_std", 0.8)
        self.svec_value = cfg.data.get("svec_value", 0.02)
        self.alpha_value = cfg.data.get("alpha_value", 0.8)
        self.facex = cfg.data.get("facex", False)
        self.z_scale = cfg.data.get("z_scale", 1.0)
        self.gs_attribute_root = cfg.data.gs_attribute_root

        self.pcd_path_list = glob.glob(
            os.path.join()(cfg.data.root + "/*.glb")
        )  # 获取所有的ply文件的路径
        self.pcd_id_list = [
            p.split(".")[-2].split("/")[-1] for p in self.pcd_path_list
        ]  # 获取所有的ply文件的文件名作为id
        print("Loading 3D information...")
        self.pcd_root = cfg.data.pcd_root
        invalid_list = []
        if self.rebuild:
            for id in tqdm(self.instance_list, total=len(self.instance_list)):
                if id not in self.pcd_id_list:
                    print(f"{id} not found in the 3D dataset")
                    invalid_list.append(id)
                    continue
                pcd_path = os.path.join(
                    self.pcd_root, f"{id}", "point_clouds", f"{id}.ply"
                )
                gs_attribute_path = os.path.join(self.gs_attribute_root, f"{id}.npz")
                if not os.path.exists(gs_attribute_path):
                    gs_attribute = self.get_gs_attribute(pcd_path)
                    gs_attribute = {"id": id, "initial_value": gs_attribute}
                    np.savez(gs_attribute_path, **gs_attribute)

            with open(self.invalid_file, "w") as f:
                for id in invalid_list:
                    f.write(f"{id}\n")
                    invalid_len = len(invalid_list)
            print(
                colored(
                    f"{invalid_len} invalid 3d datas are saved in {self.invalid_file}.",
                    "red",
                )
            )
            if invalid_len > 0:
                answer = input(
                    "please check the 3d data, do you want to terminate the training? (y/n): "
                )
                if answer.lower() == "y":
                    print("Terminating the training...")
                    raise Exception("Invalid 3d data found.")
                else:
                    print("Continuing the training...")
        print("3D data is ready.")

    def get_gs_attribute(self, pcd_path) -> dict:
        #! 检查float
        pcd_data = o3d.io.read_point_cloud(pcd_path)
        pcd_data = np.asarray(pcd_data.points)
        xyz = torch.from_numpy(pcd_data[:, :3]).to(torch.float32)
        rgb = torch.from_numpy(pcd_data[:, 3:]).to(torch.float32)

        # TODO:点云数量>4096   <4096

        assert xyz.shape == (self.number_of_points, 3), "XYZ data shape is not correct."
        assert rgb.shape == (self.number_of_points, 3), "RGB data shape is not correct."

        xyz -= xyz.mean(dim=0, keepdim=True)  # zero-centering
        xyz = xyz / (xyz.norm(dim=1, keepdim=True).max() + 1e-5)  # normalize [0,1]
        xyz = xyz * self.mean_std

        if self.facex:
            print(
                colored(
                    "Will align the points cloud to the x axis.",
                    "yellow",
                )
            )
            # rotate along the x-axis
            x, y, z = xyz.chunk(3, dim=-1)
            xyz = torch.cat([-y, x, z], dim=-1)

        z_scale = self.z_scale
        xyz[..., 2] *= z_scale

        gs_attribute = {}
        gs_attribute["mean"] = xyz
        gs_attribute["color"] = rgb
        gs_attribute["svec"] = self.get_svec()
        gs_attribute["qvec"] = self.get_qvec()
        gs_attribute["alpha"] = self.get_alpha()

        return gs_attribute

    def get_svec(self):
        svec = (
            torch.ones(self.number_of_points, 3, dtype=torch.float32) * self.svec_value
        )
        return svec

    def get_qvec(self):
        qvec = torch.zeros(self.number_of_points, 4, dtype=torch.float32)
        qvec[:, 0] = 1.0
        return qvec

    def get_alpha(self):
        alpha = (
            torch.ones(self.number_of_points, 3, dtype=torch.float32) * self.alpha_value
        )
        return alpha

    def update(self, step):
        self.step = step

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, index):
        data = {}
        ids = self.instance_list[index]
        for id in ids:
            gs_attribute_path = os.path.join(self.gs_attribute_root, f"{id}.npz")
            gs_attribute = np.load(gs_attribute_path)
            data["data_mv"] = gs_attribute
        return data
