import os
import librosa
import multiprocessing
import json
from tqdm import tqdm


def valid_audio(audio_path):
    audio, _ = librosa.load(audio_path, mono=False)
    if len(audio.shape) != 1 and audio.shape[0] > 2:
        print(f"Warning: {audio_path} has more than 2 channels. Skipping.")
        return False
    return True

def process_audio(path: dict):
    original_path, codec_path = path["original"], path["codec"]
    try:
        valid_ori = valid_audio(original_path)
        valid_cod = valid_audio(codec_path)
        if not valid_ori or not valid_cod:
            return False, path
        return True, path
    except Exception as e:
        print(f"Could not process {original_path} and {codec_path}. Error: {str(e)}")
        return False, path

def process(datas, threads, type="train"):
    files = []
    if type == "train":
        dataset_type = datas.dataset_type
        print(f"Processing training sets, dataset type: {dataset_type}")

        if dataset_type == 1:
            for path in datas.train.dir:
                path = os.path.abspath(path)
                for folder in os.listdir(path):
                    if not os.path.isdir(os.path.join(path, folder)):
                        continue
                    original = os.path.join(path, folder, f"{datas.stems.original}.{datas.train.original_format}")
                    codec = os.path.join(path, folder, f"{datas.stems.codec}.{datas.train.codec_format}")
                    if os.path.exists(original) and os.path.exists(codec):
                        files.append({"original": original, "codec": codec})
                    else:
                        print(f"Warning: Folder {folder} is invalid, please check!")

        elif dataset_type == 2:
            for path in datas.train.dir:
                path = os.path.abspath(path)
                original_folder = os.path.join(path, datas.stems.original)
                codec_folder = os.path.join(path, datas.stems.codec)
                for file in os.listdir(original_folder):
                    original = os.path.join(original_folder, file)
                    codec = os.path.join(codec_folder, file.replace(f".{datas.train.original_format}", f".{datas.train.codec_format}"))
                    if os.path.exists(codec):
                        files.append({"original": original, "codec": codec})
                    else:
                        print(f"Warning: Could not find file {codec}, please check!")
        print(f"Total files found: {len(files)} for training")

    elif type == "valid":
        print("Processing validation sets")
        for path in datas.valid.dir:
            path = os.path.abspath(path)
            for folder in os.listdir(path):
                if not os.path.isdir(os.path.join(path, folder)):
                    continue
                original = os.path.join(path, folder, f"{datas.stems.original}.{datas.valid.original_format}")
                codec = os.path.join(path, folder, f"{datas.stems.codec}.{datas.valid.codec_format}")
                if os.path.exists(original) and os.path.exists(codec):
                    files.append({"original": original, "codec": codec})
        print(f"Total files found: {len(files)} for validation")

    datas_process = files.copy()
    if threads <= 1:
        for path in tqdm(datas_process):
            out = process_audio(path)
            if not out[0]:
                files.remove(out[1])
    else:
        p = multiprocessing.Pool(threads)
        for out in tqdm(p.imap(process_audio, datas_process), total=len(datas_process)):
            if not out[0]:
                files.remove(out[1])
    return files

def get_filelist(cfg, expdir, threads=multiprocessing.cpu_count(), reprocess=False):
    os.makedirs(expdir, exist_ok=True)
    if not os.path.exists(os.path.join(expdir, "filelist.json")) or reprocess:
        print("Strat processing filelist")
        train_datas = process(cfg, threads, type="train")
        valid_datas = process(cfg, threads, type="valid")
        datas = {"train": train_datas, "valid": valid_datas}
        with open(os.path.join(expdir, "filelist.json"), "w") as f:
            json.dump(datas, f, indent=4)
    else:
        print(f"Loading filelist from {os.path.join(expdir, 'filelist.json')}")
        with open(os.path.join(expdir, "filelist.json"), "r") as f:
            datas = json.load(f)
    return datas