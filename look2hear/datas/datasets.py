import librosa
import torch
import random
import multiprocessing
import numpy as np
from omegaconf import OmegaConf
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple
from pytorch_lightning import LightningDataModule
from .preprocess import get_filelist


class TrainDataset(Dataset):
    def __init__(
        self,
        filelists: list[dict],
        sr: int = 44100,
        segments: int = 10,
        num_steps: int = 1000,
    ) -> None:
        self.filelists = filelists
        self.segments = int(segments * sr)
        self.sr = sr
        self.num_steps = num_steps

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        random_file = random.choice(self.filelists)

        while True:
            original, _ = librosa.load(random_file["original"], sr=self.sr, mono=False)
            codec, _ = librosa.load(random_file["codec"], sr=self.sr, mono=False)
            min_length = min(original.shape[-1], codec.shape[-1])
            if min_length > self.segments:
                break

        start = random.randint(0, min_length - self.segments)
        ori_wav = original[..., start:start + self.segments]
        codec_wav = codec[..., start:start + self.segments]

        if len(ori_wav.shape) == 1:
            ori_wav = np.stack([ori_wav, ori_wav], axis=0)
        if len(codec_wav.shape) == 1:
            codec_wav = np.stack([codec_wav, codec_wav], axis=0)

        ori_wav = torch.tensor(ori_wav)
        codec_wav = torch.tensor(codec_wav)

        max_scale = max(ori_wav.abs().max(), codec_wav.abs().max())
        if max_scale > 0:
            ori_wav = ori_wav / max_scale
            codec_wav = codec_wav / max_scale
        return ori_wav, codec_wav


class ValidDataset(Dataset):
    def __init__(
        self,
        filelists: list[dict],
        sr: int = 44100
    ) -> None:
        self.sr = sr
        self.filelists = filelists

    def __len__(self) -> int:
        return len(self.filelists)

    def __getitem__(self, idx: int):
        original, _ = librosa.load(self.filelists[idx]["original"], sr=self.sr, mono=False)
        codec, _ = librosa.load(self.filelists[idx]["codec"], sr=self.sr, mono=False)

        min_length = min(original.shape[-1], codec.shape[-1])
        ori_wav = original[..., :min_length]
        codec_wav = codec[..., :min_length]

        if len(ori_wav.shape) == 1:
            ori_wav = np.stack([ori_wav, ori_wav], axis=0)
        if len(codec_wav.shape) == 1:
            codec_wav = np.stack([codec_wav, codec_wav], axis=0)

        ori_wav = torch.tensor(ori_wav)
        codec_wav = torch.tensor(codec_wav)
        return ori_wav, codec_wav


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_type: int,
        stems: dict,
        train: dict,
        valid: dict,
        sr: int = 44100,
        segments: int = 10,
        num_steps: int = 1000,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        expdir: str = None
    ) -> None:
        super().__init__()
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.sr = sr
        self.segments = segments
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.expdir = expdir
        self.config = OmegaConf.create({
            "dataset_type": dataset_type,
            "stems": stems,
            "train": train,
            "valid": valid
        })

    def setup(self, stage: Optional[str] = None) -> None:
        if self.num_workers == 0:
            threads = multiprocessing.cpu_count()
        else:
            threads = self.num_workers
        filelist = get_filelist(self.config, self.expdir, threads)

        self.data_train = TrainDataset(
            filelists=filelist["train"],
            sr=self.sr,
            segments=self.segments,
            num_steps=self.num_steps
        )

        self.data_val = ValidDataset(
            filelists=filelist["valid"],
            sr=self.sr
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
