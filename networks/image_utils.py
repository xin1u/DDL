
import math
import pickle

import cv2
import numpy as np
import torch


def is_numpy_file(filename):
    return filename.endswith(".npy")


def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))


def is_png_file(filename):
    return filename.lower().endswith(".png")


def is_pkl_file(filename):
    return filename.endswith(".pkl")


def load_pkl(filename_):
    with open(filename_, "rb") as f:
        return pickle.load(f)


def save_dict(dict_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(dict_, f)


def load_npy(filepath):
    return np.load(filepath)


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff ** 2).mean().sqrt()
    return 20 * torch.log10(1 / rmse)


def batch_PSNR(img1, img2, average=True):
    values = [myPSNR(im1, im2) for im1, im2 in zip(img1, img2)]
    return sum(values) / len(values) if average else sum(values)


def _starts(length, crop_size, overlap_size):
    if crop_size <= overlap_size:
        raise ValueError("crop_size must be larger than overlap_size")
    if length <= crop_size:
        return [0]

    stride = crop_size - overlap_size
    starts = list(range(0, length - crop_size + 1, stride))
    last = length - crop_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def splitimage(imgtensor, crop_size=128, overlap_size=64):
    _, _, H, W = imgtensor.shape
    hstarts = _starts(H, crop_size, overlap_size)
    wstarts = _starts(W, crop_size, overlap_size)

    split_data, starts = [], []
    for hs in hstarts:
        for ws in wstarts:
            split_data.append(imgtensor[:, :, hs:hs + crop_size, ws:ws + crop_size])
            starts.append((hs, ws))
    return split_data, starts


def splitimage_fix(imgtensor, crop_size=128, overlap_size=64):
    return splitimage(imgtensor, crop_size=crop_size, overlap_size=overlap_size)


def get_scoremap(H, W, C, B=1, is_mean=True):
    score = torch.ones((B, C, H, W))
    if is_mean:
        return score

    center_h = H / 2
    center_w = W / 2
    for h in range(H):
        for w in range(W):
            dist = math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-6)
            score[:, :, h, w] = 1.0 / dist
    return score


def mergeimage(split_data, starts, crop_size=128, resolution=(1, 3, 128, 128), is_mean=True):
    B, C, H, W = resolution
    device = split_data[0].device
    dtype = split_data[0].dtype

    tot_score = torch.zeros((B, C, H, W), device=device, dtype=dtype)
    merge_img = torch.zeros((B, C, H, W), device=device, dtype=dtype)
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=is_mean).to(device=device, dtype=dtype)

    for simg, (hs, ws) in zip(split_data, starts):
        h_end = hs + simg.shape[-2]
        w_end = ws + simg.shape[-1]
        local_score = scoremap[:, :, :simg.shape[-2], :simg.shape[-1]]
        merge_img[:, :, hs:h_end, ws:w_end] += local_score * simg
        tot_score[:, :, hs:h_end, ws:w_end] += local_score

    return merge_img / torch.clamp(tot_score, min=1e-8)

