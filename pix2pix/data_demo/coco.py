import os
from PIL import Image
from skimage.color import rgb2lab

import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import color


def reverse_transform(L, AB):
    L = np.clip(L.cpu().numpy(), -1, 1)
    AB = np.clip(AB.cpu().numpy(), -1, 1)

    L = (L + 1) * 50.0  # Scale to [0, 100] from [-1, 1]
    AB = AB * 128  # scale to [-128 127] from [-1 1]
    LAB = np.concatenate([L, AB], axis=0).transpose(1, 2, 0)

    return L.squeeze(), color.lab2rgb(LAB)


class COCO(Dataset):
    def __init__(self, root_dir, train_fold, num_dat=-1, resize_to=(256, 256)):
        assert num_dat > 0 or num_dat == -1
        self.root_dir = root_dir
        self.train_path = os.path.join(root_dir, train_fold)
        self.resize_to = resize_to
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.RandomHorizontalFlip()]
        )

        fnames = [
            f
            for f in os.listdir(self.train_path)
            if os.path.isfile(os.path.join(self.train_path, f))
        ]
        max_imgs = len(fnames)
        lim = max_imgs if (num_dat == -1 or num_dat > max_imgs) else num_dat
        self.fnames = fnames[:lim]

        # remove any files that are already black and white in the dataset
        # This is a simple bw detectio but it removes most of the bws
        self.exc_path = os.path.join(root_dir, "bw_indices.txt")
        if not os.path.exists(self.exc_path):
            self._filter_bw()

        with open(self.exc_path, "r") as file:
            bw_indices = {int(line.strip()) for line in file}

        self.fnames = [f for i, f in enumerate(self.fnames) if i not in bw_indices]
        print(f"removed {len(bw_indices)} black and white images")

    def _filter_bw(self, grayscale_thresh=1e-1):
        print("Filtering to remove black and white images")
        bad_indices = []
        for i, filename in tqdm(enumerate(self.fnames)):
            img_path = os.path.join(self.train_path, filename)
            image = Image.open(img_path)

            if image.mode != "RGB":
                bad_indices.append(i)
                continue

            img = np.array(image.convert("RGB"))
            b, g, r = cv2.split(img)
            r_g = np.count_nonzero(abs(r - g))
            r_b = np.count_nonzero(abs(r - b))
            g_b = np.count_nonzero(abs(g - b))
            diff_sum = float(r_g + r_b + g_b)
            ratio = diff_sum / img.size

            if ratio < grayscale_thresh:
                bad_indices.append(i)

        print(f"removed {len(bad_indices)} images.")
        with open(self.exc_path, "w") as file:
            for index in bad_indices:
                file.write(f"{index}\n")

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.train_path, self.fnames[idx])
        image = Image.open(img_path)
        image = image.resize(self.resize_to, Image.BILINEAR)

        rgb = np.array(image.convert("RGB"))
        lab = rgb2lab(rgb)

        # Convert to a normalized range
        L = (lab[:, :, 0:1] / 50) - 1
        AB = lab[:, :, 1:] / 128
        lab = np.concatenate((L, AB), axis=-1)

        lab = self.transforms(lab)

        return {"L": lab[0:1], "AB": lab[1:]}
