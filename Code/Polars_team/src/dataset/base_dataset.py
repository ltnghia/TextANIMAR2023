import os
import numpy as np
import random
import torch

random.seed = 42

from torch.utils.data import Dataset
from torchvision.io import read_image

from src.dataset.utils import read_pc


class SiameseNetworkDataset(Dataset):
    def __init__(self, root_dir, folder, pc_transform=None, img_transform=None):
        self.root_dir = root_dir

        self.pc_transforms = pc_transform
        self.img_transforms = img_transform

        self.point_cloud_files = []
        self.image_sketch_files = []

        point_cloud_path = os.path.join(root_dir, "pointcloud")
        img_sketch_path = os.path.join(root_dir, "sketch")

        for files in sorted(os.listdir(point_cloud_path)):
            self.point_cloud_files.append(os.path.join(point_cloud_path, files))

        for files in sorted(os.listdir(img_sketch_path)):
            self.image_sketch_files.append(os.path.join(img_sketch_path, files))

    def _preprocess_pc(self, file):
        verts = read_pc(file)
        pointcloud = np.array(verts)
        if self.pc_transforms:
            pointcloud = self.pc_transforms(pointcloud)
        return pointcloud

    def _preprocess_image(self, file):
        image = read_image(file)
        if self.img_transforms:
            image = self.img_transforms(image)
        return image

    def __len__(self):
        return len(self.point_cloud_files)

    def __getitem__(self, idx):
        pc_path = self.point_cloud_files[idx]
        img_path = self.image_sketch_files[idx]
        with open(pc_path, "r") as f:
            point_cloud = self._preprocess_pc(f)

        with open(img_path, "r") as f:
            sketch_image = self._preprocess_image(f)

        return {"point_cloud": point_cloud, "sketch_image": sketch_image}

    def collate_fn(self, batch):
        batch_as_dict = {
            "point_clouds": torch.stack([x["point_cloud"] for x in batch]),
            "sketch_images": torch.stack([x["sketch_image"] for x in batch]),
        }

        return batch_as_dict
