
import bisect
import os
import random
import warnings
from glob import glob
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _is_image_file(name: str) -> bool:
    return name.lower().endswith(IMG_EXTENSIONS)


def _list_images(root: str) -> List[str]:
    files = [
        os.path.join(root, name)
        for name in sorted(os.listdir(root))
        if _is_image_file(name)
    ]
    if not files:
        raise FileNotFoundError(f"No image files found in {root}")
    return files


def _load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _pad_to_crop(img: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)
    if pad_h == 0 and pad_w == 0:
        return img
    mode = "reflect" if h > 1 and w > 1 else "edge"
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)


def _augment_img(img: np.ndarray, mode: int = 0) -> np.ndarray:
    if mode == 0:
        out = img
    elif mode == 1:
        out = np.flipud(np.rot90(img))
    elif mode == 2:
        out = np.flipud(img)
    elif mode == 3:
        out = np.rot90(img, k=3)
    elif mode == 4:
        out = np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        out = np.rot90(img)
    elif mode == 6:
        out = np.rot90(img, k=2)
    else:
        out = np.flipud(np.rot90(img, k=3))
    return out.copy()


def _random_crop_pair(inp: np.ndarray, gt: np.ndarray, crop_size: int) -> Tuple[np.ndarray, np.ndarray]:
    inp = _pad_to_crop(inp, crop_size)
    gt = _pad_to_crop(gt, crop_size)
    h, w = inp.shape[:2]
    top = random.randint(0, max(0, h - crop_size))
    left = random.randint(0, max(0, w - crop_size))
    return (
        inp[top:top + crop_size, left:left + crop_size],
        gt[top:top + crop_size, left:left + crop_size],
    )


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    return transforms.ToTensor()(img)


def _resolve(root: str, rel_or_abs: str) -> str:
    rel_or_abs = rel_or_abs.strip()
    return rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(root, rel_or_abs)


class my_dataset(Dataset):

    def __init__(self, rootA_in, rootA_label, crop_size=256, fix_sample_A=500, regular_aug=True):
        super().__init__()
        self.regular_aug = regular_aug
        self.crop_size = crop_size

        input_paths = _list_images(rootA_in)
        if fix_sample_A and fix_sample_A > 0:
            sample_count = min(fix_sample_A, len(input_paths))
            input_paths = random.sample(input_paths, sample_count)

        self.imgs_in_A = sorted(input_paths)
        self.imgs_gt_A = []
        for in_path in self.imgs_in_A:
            gt_path = os.path.join(rootA_label, os.path.basename(in_path))
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"Missing GT pair for {in_path}: expected {gt_path}")
            self.imgs_gt_A.append(gt_path)

    def __getitem__(self, index):
        return self.read_imgs_pair(
            self.imgs_in_A[index],
            self.imgs_gt_A[index],
            self.train_transform,
            self.crop_size,
        )

    def read_imgs_pair(self, in_path, gt_path, transform, crop_size):
        img_name = os.path.basename(in_path)
        inp = _load_rgb(in_path)
        gt = _load_rgb(gt_path)
        data_in, data_gt = transform(inp, gt, crop_size)
        return data_in, data_gt, img_name

    def augment_img(self, img, mode=0):
        return _augment_img(img, mode)

    def train_transform(self, img, label, patch_size=256):
        img, label = _random_crop_pair(img, label, patch_size)
        if self.regular_aug:
            mode = random.randint(0, 7)
            img = self.augment_img(img, mode=mode)
            label = self.augment_img(label, mode=mode)
        return _to_tensor(img), _to_tensor(label)

    def __len__(self):
        return len(self.imgs_in_A)


def read_txt(txt_name="RealHaze.txt", sample_num=5000):
    pairs = []
    with open(txt_name, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Expected `input gt` pair in {txt_name}: {line}")
            pairs.append((parts[0], parts[1]))

    if sample_num and sample_num > 0:
        pairs = random.sample(pairs, min(sample_num, len(pairs)))
    return [p[0] for p in pairs], [p[1] for p in pairs]


class my_dataset_wTxt(Dataset):

    def __init__(self, rootA, rootA_txt, crop_size=256, fix_sample_A=500, regular_aug=True):
        super().__init__()
        self.regular_aug = regular_aug
        self.crop_size = crop_size

        in_files, gt_files = read_txt(rootA_txt, sample_num=fix_sample_A)
        self.imgs_in_A = [_resolve(rootA, path) for path in in_files]
        self.imgs_gt_A = [_resolve(rootA, path) for path in gt_files]

    def __getitem__(self, index):
        return self.read_imgs_pair(
            self.imgs_in_A[index],
            self.imgs_gt_A[index],
            self.train_transform,
            self.crop_size,
        )

    def read_imgs_pair(self, in_path, gt_path, transform, crop_size):
        img_name = os.path.basename(in_path)
        inp = _load_rgb(in_path)
        gt = _load_rgb(gt_path)
        data_in, data_gt = transform(inp, gt, crop_size)
        return data_in, data_gt, img_name

    def augment_img(self, img, mode=0):
        return _augment_img(img, mode)

    def train_transform(self, img, label, patch_size=256):
        img, label = _random_crop_pair(img, label, patch_size)
        if self.regular_aug:
            mode = random.randint(0, 7)
            img = self.augment_img(img, mode=mode)
            label = self.augment_img(label, mode=mode)
        return _to_tensor(img), _to_tensor(label)

    def __len__(self):
        return len(self.imgs_in_A)


class my_dataset_eval(Dataset):

    def __init__(self, root_in, root_label, transform=None, fix_sample=100):
        super().__init__()
        self.transform = transform or transforms.ToTensor()

        input_paths = _list_images(root_in)
        gt_paths = _list_images(root_label)
        if fix_sample and fix_sample > 0:
            input_paths = input_paths[:min(fix_sample, len(input_paths))]

        gt_by_name = {os.path.basename(path): path for path in gt_paths}
        if all(os.path.basename(path) in gt_by_name for path in input_paths):
            paired = [(path, gt_by_name[os.path.basename(path)]) for path in input_paths]
        else:
            paired = list(zip(input_paths, gt_paths[:len(input_paths)]))

        self.imgs_in = [p[0] for p in paired]
        self.imgs_gt = [p[1] for p in paired]

    def __getitem__(self, index):
        in_path = self.imgs_in[index]
        gt_path = self.imgs_gt[index]
        img_name = os.path.basename(in_path)

        inp = Image.open(in_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        data_in = self.transform(inp)
        data_gt = self.transform(gt)

        _, h, w = data_gt.shape
        if (h % 16 != 0) or (w % 16 != 0):
            new_size = ((h // 16) * 16, (w // 16) * 16)
            data_gt = transforms.Resize(new_size)(data_gt)
            data_in = transforms.Resize(new_size)(data_in)

        return data_in, data_gt, img_name

    def __len__(self):
        return len(self.imgs_in)


class DatasetForInference(Dataset):

    def __init__(self, dir_path):
        self.image_paths = sorted(path for path in glob(os.path.join(dir_path, "*")) if _is_image_file(path))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        input_path = self.image_paths[index]
        input_image = Image.open(input_path).convert("RGB")
        input_tensor = self.transform(input_image)
        _, h, w = input_tensor.shape
        if (h % 16 != 0) or (w % 16 != 0):
            input_tensor = transforms.Resize(((h // 16) * 16, (w // 16) * 16))(input_tensor)
        return input_tensor, os.path.basename(input_path)


class ConcatDataset(Dataset):

    @staticmethod
    def cumsum(sequence: Sequence[Dataset]) -> List[int]:
        result, total = [], 0
        for dataset in sequence:
            total += len(dataset)
            result.append(total)
        return result

    def __init__(self, datasets: Iterable[Dataset]):
        super().__init__()
        self.datasets = list(datasets)
        assert self.datasets, "datasets should not be empty"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn(
            "cummulative_sizes is deprecated; use cumulative_sizes instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cumulative_sizes


class BaseDataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

    def reset(self):
        return None


class FusionDataset(BaseDataset):

    def __init__(self, datasets, fusion_ratios=None):
        self.datasets = list(datasets)
        self.size = sum(len(dataset) for dataset in self.datasets)
        self.fusion_ratios = fusion_ratios or [1.0 / len(self.datasets)] * len(self.datasets)
        print(
            "[i] fusion dataset: %d images, branches=%s, ratios=%s"
            % (self.size, [len(dataset) for dataset in self.datasets], self.fusion_ratios)
        )

    def reset(self):
        for dataset in self.datasets:
            if hasattr(dataset, "reset"):
                dataset.reset()

    def __getitem__(self, index):
        residual = 1.0
        for i, ratio in enumerate(self.fusion_ratios):
            if random.random() < ratio / residual or i == len(self.fusion_ratios) - 1:
                dataset = self.datasets[i]
                return dataset[index % len(dataset)]
            residual -= ratio

    def __len__(self):
        return self.size


class MultiTaskDataset(Dataset):

    def __init__(self, datasets, tmats):
        assert len(datasets) == len(tmats)
        self.datasets = list(datasets)
        self.tmats = list(tmats)
        self.cum_sizes = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cum_sizes.append(total)

    def __len__(self):
        return self.cum_sizes[-1] if self.cum_sizes else 0

    def __getitem__(self, idx):
        task_idx = bisect.bisect_right(self.cum_sizes, idx)
        sample_idx = idx if task_idx == 0 else idx - self.cum_sizes[task_idx - 1]
        data_in, data_gt, name = self.datasets[task_idx][sample_idx]
        return data_in, data_gt, name, self.tmats[task_idx]


if __name__ == "__main__":
    print("DDL dataset helpers loaded. Use train_pretrain.py, train_finetune.py, or train_unified.py.")

