# MultiMNIST dataset loaders.
# Merged from PMTL (pickle-based) and CPMTL (raw download + process).
# Use load_pickle() for PMTL-style .pickle files.
# Use MultiMNIST class for CPMTL-style download/process pipeline.

import codecs
import gzip
import pickle
import random
import urllib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data


# ---------------------------------------------------------------------------
# PMTL-style: load from pre-built pickle files
# ---------------------------------------------------------------------------

DATASET_FILES = {
    "mnist": "multi_mnist.pickle",
    "fashion": "multi_fashion.pickle",
    "fashion_and_mnist": "multi_fashion_and_mnist.pickle",
}


def load_pickle(data_dir: str, name: str):
    """Load MultiMNIST variant from a .pickle file.

    Returns:
        train_set, test_set: TensorDataset
    """
    path = Path(data_dir) / DATASET_FILES[name]
    with open(path, "rb") as f:
        trainX, trainLabel, testX, testLabel = pickle.load(f)

    trainX = torch.from_numpy(trainX.reshape(-1, 1, 36, 36)).float()
    testX = torch.from_numpy(testX.reshape(-1, 1, 36, 36)).float()
    trainLabel = torch.from_numpy(trainLabel).long()
    testLabel = torch.from_numpy(testLabel).long()

    return (
        data.TensorDataset(trainX, trainLabel),
        data.TensorDataset(testX, testLabel),
    )


def build_dataloaders(dataset_cfg: dict):
    """Build train/test DataLoaders from config.

    dataset_cfg keys:
        name (str): 'mnist' | 'fashion' | 'fashion_and_mnist'
        data_dir (str): path to directory containing .pickle files
        batch_size (int, optional): default 256
    """
    train_set, test_set = load_pickle(dataset_cfg["data_dir"], dataset_cfg["name"])
    bs = dataset_cfg.get("batch_size", 256)
    train_loader = data.DataLoader(train_set, batch_size=bs, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=bs, shuffle=False)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# CPMTL-style: download and process raw MNIST files
# ---------------------------------------------------------------------------

class MultiMNIST(data.Dataset):
    """MultiMNIST dataset that downloads and processes raw MNIST files.

    Used by CPMTL (WeightedSum / CPMTL methods).
    """

    urls = [
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    ]
    raw_folder = "raw"
    processed_folder = "processed"
    training_file = "training.pth"
    test_file = "test.pth"

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Use download=True to download it.")

        file = self.training_file if train else self.test_file
        self.data, self.labels_l, self.labels_r = torch.load(
            self.root / self.processed_folder / file
        )

        if transform is not None:
            self.data = [
                self.transform(Image.fromarray(img.numpy().astype(np.uint8), mode="L"))
                for img in self.data
            ]

    def __getitem__(self, index):
        img = self.data[index]
        target = torch.stack([self.labels_l[index], self.labels_r[index]])
        return img, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return (
            (self.root / self.processed_folder / self.training_file).is_file()
            and (self.root / self.processed_folder / self.test_file).is_file()
        )

    def download(self):
        if self._check_exists():
            return

        (self.root / self.raw_folder).mkdir(parents=True, exist_ok=True)
        (self.root / self.processed_folder).mkdir(parents=True, exist_ok=True)

        for url in self.urls:
            print(f"Downloading {url}")
            data_bytes = urllib.request.urlopen(url).read()
            filename = url.rpartition("/")[2]
            file_path = self.root / self.raw_folder / filename
            file_path.write_bytes(data_bytes)
            unzipped_path = self.root / self.raw_folder / ".".join(filename.split(".")[:-1])
            with open(unzipped_path, "wb") as out_f, gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            file_path.unlink()

        print("Processing...")
        ims, ext = self._read_image_file(self.root / self.raw_folder / "train-images-idx3-ubyte")
        labels_l, labels_r = self._read_label_file(self.root / self.raw_folder / "train-labels-idx1-ubyte", ext)
        tims, text = self._read_image_file(self.root / self.raw_folder / "t10k-images-idx3-ubyte")
        tlabels_l, tlabels_r = self._read_label_file(self.root / self.raw_folder / "t10k-labels-idx1-ubyte", text)

        torch.save((ims, labels_l, labels_r), self.root / self.processed_folder / self.training_file)
        torch.save((tims, tlabels_l, tlabels_r), self.root / self.processed_folder / self.test_file)
        print("Done!")

    @staticmethod
    def _get_int(b):
        return int(codecs.encode(b, "hex"), 16)

    @staticmethod
    def _read_label_file(path, extension):
        with open(path, "rb") as f:
            data1 = f.read()
        with open(path, "rb") as f:
            data2 = f.read()
        length = MultiMNIST._get_int(data1[4:8])
        parsed1 = np.frombuffer(data1, dtype=np.uint8, offset=8)
        parsed2 = np.frombuffer(data2, dtype=np.uint8, offset=8)
        labels_l = np.zeros(length, dtype=np.int64)
        labels_r = np.zeros(length, dtype=np.int64)
        for i in range(length):
            labels_l[i] = parsed1[i]
            labels_r[i] = parsed2[extension[i]]
        return (
            torch.from_numpy(labels_l).long(),
            torch.from_numpy(labels_r).long(),
        )

    @staticmethod
    def _read_image_file(path, shift_pix=4, rand_shift=True):
        with open(path, "rb") as f:
            data1 = f.read()
        with open(path, "rb") as f:
            data2 = f.read()
        length = MultiMNIST._get_int(data1[4:8])
        num_rows = MultiMNIST._get_int(data1[8:12])
        num_cols = MultiMNIST._get_int(data1[12:16])
        pv1 = np.frombuffer(data1, dtype=np.uint8, offset=16).reshape(length, num_rows, num_cols)
        pv2 = np.frombuffer(data2, dtype=np.uint8, offset=16).reshape(length, num_rows, num_cols)
        multi_data = np.zeros((length, num_rows, num_cols))
        extension = np.zeros(length, dtype=np.int32)
        rights = np.random.permutation(length)
        for left in range(length):
            extension[left] = rights[left]
            lim = pv1[left]
            rim = pv2[rights[left]]
            s1 = s2 = 0
            if rand_shift:
                if random.choice([True, False]):
                    s1 = random.randint(0, shift_pix - 1)
                    s2 = random.randint(0, shift_pix)
                else:
                    s1 = random.randint(0, shift_pix)
                    s2 = random.randint(1, shift_pix)
            new_im = np.zeros((36, 36))
            new_im[s1:s1 + 28, s1:s1 + 28] += lim
            new_im[s2 + 4:s2 + 4 + 28, s2 + 4:s2 + 4 + 28] += rim
            new_im = np.clip(new_im, 0, 255)
            multi_data[left] = np.array(Image.fromarray(new_im).resize((28, 28), resample=Image.NEAREST))
        return torch.from_numpy(multi_data).view(length, num_rows, num_cols), extension
