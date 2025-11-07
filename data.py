#data.py
from sympy import false
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
import math
class MnistDataModule:
    def __init__(self, data_dir="./data", batch_size=128, num_workers = 2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307),(0.3081,))
            ])
        self._train = None
        self._val = None
        self._test = None

    def perpare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, val_ratio = 0.1):
        full_train = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
        n_val = int(len(full_train) * val_ratio)
        n_train = len(full_train) - n_val
        self._train, self._val = random_split(full_train, [n_train, n_val])
        self._test = datasets.MNIST(self.data_dir, train= False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self._val, batch_size=self.batch_size, shuffle=false,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self._test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def save_sample_grid(self, out_dir="outputs", n=16):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        raw = datasets.MNIST(self.data_dir, train=True, download=True, transform=transforms.ToTensor())
        cols = int(math.sqrt(n))
        rows = math.ceil(n/cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        axes = axes.flatten()
        for i in range(n):
            img, label = raw[i]
            axes[i].imshow(img.squeeze(0),cmap = "gray")
            axes[i].set_title(f"{label}")
            axes[i].axis("off")
        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.subtitle("MNIST Samples (label shown)")
        fig.tight_layout()
        fig.savefig(Path(out_dir) / "mnist_samples.png", dpi=150)
        plt.close(fig)