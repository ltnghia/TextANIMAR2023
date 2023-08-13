import os
import numpy as np
import random
import torch
import pandas as pd

from src.utils.loading import load_point_cloud

from torch.utils.data import Dataset

from transformers import BertTokenizer


class TextPointCloudDataset(Dataset):
    def __init__(
        self,
        text_queries_path: str,
        pc_ids_path: str,
        pc_dir: str,
        ground_truth_path: str,
        pc_transform=None,
    ):
        self.text_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

        self.pc_transforms = pc_transform

        self.ground_truth = pd.read_csv(ground_truth_path, sep=";")

        self.text_queries = pd.read_csv(text_queries_path, sep=";")

        self.point_cloud_ids = pd.read_csv(pc_ids_path)["ID"].to_list()

        self.pc_dir = pc_dir
        self.text_queries_mapping = dict(self.text_queries.values)

        self.text_queries_list = self.ground_truth["Text Query ID"].to_list()
        self.point_cloud_list = self.ground_truth["Model ID"].to_list()

    def _preprocess_pc(self, filename):
        pointcloud = load_point_cloud(filename)
        if self.pc_transforms:
            pointcloud = self.pc_transforms(pointcloud)
        return pointcloud

    def _preprocess_batch_text(self, batch):
        return self.text_tokenizer.batch_encode_plus(
            batch, padding="longest", return_tensors="pt"
        )

    def __len__(self):
        return len(self.text_queries_list)

    def __getitem__(self, idx):
        query_id = self.text_queries_list[idx]
        pc_id = self.point_cloud_list[idx]

        pc_path = os.path.join(self.pc_dir, f"{pc_id}.obj")

        pc = self._preprocess_pc(pc_path)

        text_sample = self.text_queries_mapping[query_id]

        return {
            "point_cloud": pc,
            "query": text_sample,
            "point_cloud_id": pc_id,
            "query_id": query_id,
        }

    def collate_fn(self, batch):
        batch_as_dict = {
            "point_clouds": torch.stack([x["point_cloud"] for x in batch])
            .float()
            .transpose(1, 2),
            "queries": self._preprocess_batch_text([x["query"] for x in batch]),
            "point_cloud_ids": [x["point_cloud_id"] for x in batch],
            "query_ids": [x["query_id"] for x in batch],
        }

        return batch_as_dict


class TextOnlyDataSet(Dataset):
    def __init__(
        self,
        text_queries_path: str,
    ):
        self.text_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

        self.text_queries = pd.read_csv(text_queries_path, sep=";")

        self.text_queries_list = list(self.text_queries.values)

    def _preprocess_batch_text(self, batch):
        return self.text_tokenizer.batch_encode_plus(
            batch, padding="longest", return_tensors="pt"
        )

    def __len__(self):
        return len(self.text_queries_list)

    def __getitem__(self, idx):
        query_id = self.text_queries_list[idx][0]

        text_sample = self.text_queries_list[idx][1]

        return {
            "query": text_sample,
            "query_id": query_id,
        }

    def collate_fn(self, batch):
        batch_as_dict = {
            "queries": self._preprocess_batch_text([x["query"] for x in batch]),
            "query_ids": [x["query_id"] for x in batch],
        }

        return batch_as_dict


class PointCloudOnlyDataset(Dataset):
    def __init__(
        self,
        pc_ids_path: str,
        pc_dir: str,
        pc_transform=None,
    ):
        self.text_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

        self.pc_transforms = pc_transform

        self.point_cloud_ids = pd.read_csv(pc_ids_path)["ID"].to_list()

        self.pc_dir = pc_dir

    def _preprocess_pc(self, filename):
        pointcloud = load_point_cloud(filename)
        if self.pc_transforms:
            pointcloud = self.pc_transforms(pointcloud)
        return pointcloud

    def __len__(self):
        return len(self.point_cloud_ids)

    def __getitem__(self, idx):
        pc_id = self.point_cloud_ids[idx]

        pc_path = os.path.join(self.pc_dir, f"{pc_id}.obj")

        pc = self._preprocess_pc(pc_path)

        return {
            "point_cloud": pc,
            "point_cloud_id": pc_id,
        }

    def collate_fn(self, batch):
        batch_as_dict = {
            "point_clouds": torch.stack([x["point_cloud"] for x in batch])
            .float()
            .transpose(1, 2),
            "point_cloud_ids": [x["point_cloud_id"] for x in batch],
        }

        return batch_as_dict
