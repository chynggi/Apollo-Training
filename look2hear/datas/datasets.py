import os
import librosa
import torch
import random
import soundfile as sf
import pickle
from typing import Tuple
from torchaudio.functional import apply_codec
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple
from pytorch_lightning import LightningDataModule
from tqdm import tqdm


def match2(x, d):
    minlen = min(x.shape[-1], d.shape[-1])
    x, d = x[...,:minlen], d[...,:minlen]
    Fx = torch.fft.rfft(x, dim=-1)
    Fd = torch.fft.rfft(d, dim=-1)
    Phi = Fd * Fx.conj()
    Phi = Phi / (Phi.abs() + 1e-3)
    Phi[:, 0] = 0
    tmp = torch.fft.irfft(Phi, dim=-1)
    tau = torch.argmax(tmp.abs(), dim=-1).tolist()
    return tau


def codec_simu(wav, sr=16000, options={'bitrate':'random','compression':'random'}):
    if options['bitrate'] == 'random':
        options['bitrate'] = random.choice([24000, 32000, 48000, 64000, 96000, 128000])
    compression = int(options['bitrate']//1000)
    param = {'format': "mp3", "compression": compression}
    wav_encdec = apply_codec(wav, sr, **param)
    if wav_encdec.shape[-1] >= wav.shape[-1]:
        wav_encdec = wav_encdec[...,:wav.shape[-1]]
    else:
        wav_encdec = torch.cat([wav_encdec, wav[..., wav_encdec.shape[-1]:]], -1)
    tau = match2(wav, wav_encdec)
    wav_encdec = torch.roll(wav_encdec, -tau[0], -1)
    return wav_encdec


def is_valid_audio(file_path):
    try:
        with sf.SoundFile(file_path) as _:
            return True
    except:
        return False


class TrainDataset(Dataset):
    def __init__(
        self,
        filelists: list,
        original_dir: str,
        codec_dir: str,
        codec_format: str,
        codec: dict,
        sr: int = 44100,
        segments: int = 10,
        num_steps: int = 1000,
    ) -> None:
        self.filelists = filelists
        self.original_dir = original_dir
        self.codec_dir = codec_dir
        self.codec_format = codec_format
        self.auto_codec = codec.get("enable", True)
        self.codec_options = codec.get("options", {'bitrate':'random','compression':'random'})
        self.segments = int(segments * sr)
        self.sr = sr
        self.num_steps = num_steps

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        all_files = list(self.filelists)
        random_file = random.choice(all_files)
        wav, _ = librosa.load(os.path.join(self.original_dir, random_file), sr=self.sr, mono=False)
        start = random.randint(0, wav.shape[-1] - self.segments)

        ori_wav = wav[..., start:start + self.segments]
        if not isinstance(ori_wav, torch.Tensor):
            ori_wav = torch.tensor(ori_wav)

        if self.auto_codec:
            codec_wav = codec_simu(ori_wav, sr=self.sr, options=self.codec_options)
        else:
            base_name = os.path.splitext(random_file)[0]
            load_wav, _ = librosa.load(os.path.join(self.codec_dir, f"{base_name}.{self.codec_format}"), sr=self.sr, mono=False)
            codec_wav = load_wav[..., start:start + self.segments]

        if not isinstance(codec_wav, torch.Tensor):
            codec_wav = torch.tensor(codec_wav)

        max_scale = max(ori_wav.abs().max(), codec_wav.abs().max())
        if max_scale > 0:
            ori_wav = ori_wav / max_scale
            codec_wav = codec_wav / max_scale

        return ori_wav, codec_wav


class ValidDataset(Dataset):
    def __init__(
        self,
        sr,
        valid_dir: str,
        valid_original: str,
        valid_codec: str,
        filelists: list
    ) -> None:
        self.sr = sr
        self.valid_original = valid_original
        self.valid_codec = valid_codec
        self.data_path = [os.path.join(valid_dir, i) for i in filelists]

    def __len__(self) -> int:
        return len(self.data_path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ori_wav, _ = librosa.load(os.path.join(self.data_path[idx], self.valid_original), sr=self.sr, mono=False)
        codec_wav, _ = librosa.load(os.path.join(self.data_path[idx], self.valid_codec), sr=self.sr, mono=False)
        minlen = min(ori_wav.shape[-1], codec_wav.shape[-1])
        ori_wav, codec_wav = ori_wav[..., :minlen], codec_wav[..., :minlen]

        if not isinstance(ori_wav, torch.Tensor):
            ori_wav = torch.tensor(ori_wav)
        if not isinstance(codec_wav, torch.Tensor):
            codec_wav = torch.tensor(codec_wav)

        return ori_wav, codec_wav


class DataModule(LightningDataModule):
    def __init__(
        self,
        original_dir: str,
        codec_dir: str,
        codec_format: str,
        valid_dir: str,
        valid_original: str,
        valid_codec: str,
        codec: dict,
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

        self.original_dir = original_dir
        self.codec_dir = codec_dir
        self.codec_format = codec_format
        self.valid_dir = valid_dir
        self.valid_original = valid_original
        self.valid_codec = valid_codec
        self.codec = codec
        self.sr = sr
        self.segments = segments
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.expdir = expdir

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.data_train and not self.data_val:
            if not os.path.isfile(os.path.join(self.expdir, "filelists.pkl")):
                self.collect_files()
            else:
                print("Loading train filelists from cache...")

            with open(os.path.join(self.expdir, "filelists.pkl"), "rb") as f:
                filelists = pickle.load(f)

            self.data_train = TrainDataset(
                filelists=filelists["train"],
                original_dir=self.original_dir,
                codec_dir=self.codec_dir,
                codec_format=self.codec_format,
                codec=self.codec,
                sr=self.sr,
                segments=self.segments,
                num_steps=self.num_steps
            )
            self.data_val = ValidDataset(
                sr=self.sr,
                valid_dir=self.valid_dir,
                valid_original=self.valid_original,
                valid_codec=self.valid_codec,
                filelists=filelists["valid"]
            )

    def collect_files(self) -> None:
        train_original_files = []
        print(f"Collecting train files, total files: {len(os.listdir(self.original_dir))}")
        for file in tqdm(os.listdir(self.original_dir), desc="Collecting train files"):
            if is_valid_audio(os.path.join(self.original_dir, file)) is False:
                print(f"Error loading {file} in original dir, skipping...")
                continue
            if not self.codec.get("enable", True):
                if is_valid_audio(os.path.join(self.codec_dir, f"{os.path.splitext(file)[0]}.{self.codec_format}")) is False:
                    print(f"Error loading {file} in codec dir, skipping...")
                    continue
            train_original_files.append(os.path.join(file))
        print(f"Total valid train files: {len(train_original_files)}")

        valid_original_files = []
        print(f"Collecting valid files, total dirs: {len(os.listdir(self.valid_dir))}")
        for dir in tqdm(os.listdir(self.valid_dir)):
            orig = is_valid_audio(os.path.join(self.valid_dir, dir, self.valid_original))
            codec = is_valid_audio(os.path.join(self.valid_dir, dir, self.valid_codec))
            if not orig or not codec:
                print(f"Error loading {dir} in valid dir, skipping...")
                continue
            valid_original_files.append(dir)
        print(f"Total valid valid files: {len(valid_original_files)}")

        filelists = {"train": train_original_files, "valid": valid_original_files}
        with open(os.path.join(self.expdir, "filelists.pkl"), "wb") as f:
            pickle.dump(filelists, f)

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
