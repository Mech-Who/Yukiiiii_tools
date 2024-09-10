import numpy as np
import torch
import cv2
from PIL import Image


def np2tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr)


def tensor2np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.numpy()


def pil2np(img: Image) -> np.ndarray:
    return np.array(img)


def np2pil(arr: np.ndarray) -> Image:
    return Image.fromarray(arr)


def tensor2pil(tensor: torch.Tensor) -> Image:
    return Image.fromarray(tensor.cpu().numpy())


def pil2tensor(img: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(img))
