import librosa
import os
import random
import numpy as np
import soundfile as sf
import traceback
import subprocess
import multiprocessing
import argparse
from tqdm import tqdm

FFMPEG = "ffmpeg"  # specify the path to ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_folder", type=str, default="input", help="Folder containing the input audio files.")
parser.add_argument("-o", "--output_folder", type=str, default="output", help="Folder to store the generated dataset.")
parser.add_argument("-t", "--dataset_type", type=int, choices=[1, 2], default=1, help="Type of dataset to generate. See README for more details.")
parser.add_argument("-gt", "--generate_train", action="store_true", help="Generate the training dataset.")
parser.add_argument("-gv", "--generate_valid", action="store_true", help="Generate the validation dataset.")
parser.add_argument("-th", "--threads", type=int, default=multiprocessing.cpu_count(), help="Number of threads to use for processing.")
parser.add_argument("--save_logs", action="store_true", help="Save the logs of the dataset generation.")
parser.add_argument("--bitrates", nargs="+", default=["64k", "96k", "128k", "192k", "256k", "320k"], help="List of bitrates to use for the mp3 conversion.")
parser.add_argument("--enable_quality", action="store_true", help="Enable quality settings for the mp3 conversion.")
parser.add_argument("--quality_possibility", type=float, default=1.0, help="Possibility of using quality settings for the mp3 conversion.")
parser.add_argument("--quality_min", type=int, default=0, help="Minimum quality to use for the mp3 conversion.")
parser.add_argument("--quality_max", type=int, default=9, help="Maximum quality to use for the mp3 conversion.")
parser.add_argument("--enable_lowpass", action="store_true", help="Enable lowpass filter for the mp3 conversion.")
parser.add_argument("--lowpass_possibility", type=float, default=1.0, help="Possibility of using lowpass filter for the mp3 conversion.")
parser.add_argument("--lowpass_min_freq", type=int, default=12000, help="Minimum frequency for the lowpass filter.")
parser.add_argument("--lowpass_max_freq", type=int, default=20000, help="Maximum frequency for the lowpass filter.")
args = parser.parse_args()


def check_isvalid():
    infos = []
    try:
        ffmpeg_version_output = subprocess.check_output(["ffmpeg", "-version"], text=True)
        first_line = ffmpeg_version_output.splitlines()[0]
        infos.append(f"FFmpeg installed: {first_line}")
    except FileNotFoundError:
        print("FFmpeg is not installed. Please install FFmpeg to use this package.")
        os._exit(-1)

    if args.generate_train and not args.generate_valid:
        infos.append(f"Generating training dataset, dataset type: {args.dataset_type}")
    elif args.generate_valid and not args.generate_train:
        infos.append("Generating validation dataset")
    else:
        print("ERROR: You need to specify either --generate_train or --generate_valid.")
        os._exit(-1)

    infos.append(f"Bitrates: {args.bitrates}")

    if args.enable_quality:
        if args.quality_min >= 0 and args.quality_max <= 9 and args.quality_min <= args.quality_max:
            infos.append(f"Quality settings enabled. Quality range: {args.quality_min} to {args.quality_max}, possibility: {args.quality_possibility}")
        else:
            print("ERROR: Invalid quality settings.")
            os._exit(-1)

    if args.enable_lowpass:
        if args.lowpass_min_freq >= 0 and args.lowpass_max_freq <= 22050 and args.lowpass_min_freq <= args.lowpass_max_freq:
            infos.append(f"Lowpass settings enabled. Frequency range: {args.lowpass_min_freq} to {args.lowpass_max_freq}, possibility: {args.lowpass_possibility}")
        else:
            print("ERROR: Invalid lowpass settings.")
            os._exit(-1)

    print("\n".join(infos))
    return "\n".join(infos)


def to_mp3(args, input_file, output_path, sr=44100):
    infos = ["\n", f"Processing file: {input_file}"]
    bitrate = random.choice(args.bitrates)
    infos.append(f"Use random bitrate {bitrate}")

    param = ""
    if args.enable_quality and random.random() < args.quality_possibility:
        quality = random.randint(args.quality_min, args.quality_max)
        param += f" -q:a {quality}"
        infos.append(f"Use random quality: {quality}")

    if args.enable_lowpass and random.random() < args.lowpass_possibility:
        value = random.randint(args.lowpass_min_freq, args.lowpass_max_freq)
        param += f' -af "lowpass=f={value}"'
        infos.append(f"Use random lowpass: {value}")

    to_mp3 = f"{FFMPEG} -i \"{input_file}\" -ar {sr} -ac 2 -b:a {bitrate} {param} -vn \"{output_path}\" -y"
    infos.append(f"FFmpeg command: {to_mp3}")
    os.system(f"{to_mp3} > {os.devnull} 2>&1")
    return "\n".join(infos)


def process_audio(audio_name):
    base_name = os.path.splitext(audio_name)[0]
    try:
        waveform, _ = librosa.load(os.path.join(args.input_folder, audio_name), sr=44100, mono=False)
        if len(waveform.shape) == 1:
            waveform = np.stack([waveform, waveform], axis=0)
            print(f"Warning: {audio_name} is mono. Converting to stereo.")
        if len(waveform.shape) != 1 and waveform.shape[0] > 2:
            print(f"Warning: {audio_name} has more than 2 channels. Skipping.")
            return

        if args.generate_train:
            if args.dataset_type == 1:
                store_dir = os.path.join(args.output_folder, base_name)
                os.makedirs(store_dir, exist_ok=True)
                infos = to_mp3(
                    args=args,
                    input_file=os.path.join(args.input_folder, audio_name),
                    output_path=os.path.join(store_dir, "codec.mp3")
                )
                sf.write(os.path.join(store_dir, "original.wav"), waveform.T, 44100)
            elif args.dataset_type == 2:
                os.makedirs(os.path.join(args.output_folder, "original"), exist_ok=True)
                os.makedirs(os.path.join(args.output_folder, "codec"), exist_ok=True)
                infos = to_mp3(
                    args=args,
                    input_file=os.path.join(args.input_folder, audio_name),
                    output_path=os.path.join(args.output_folder, "codec", f"{base_name}.mp3")
                )
                sf.write(os.path.join(args.output_folder, "original", f"{base_name}.wav"), waveform.T, 44100)
            else:
                print("ERROR: Invalid dataset type.")
                os._exit(1)

        if args.generate_valid:
            store_dir = os.path.join(args.output_folder, base_name)
            os.makedirs(store_dir, exist_ok=True)
            infos = to_mp3(
                args=args,
                input_file=os.path.join(args.input_folder, audio_name),
                output_path=os.path.join(store_dir, "codec.mp3")
            )
            sf.write(os.path.join(store_dir, "original.wav"), waveform.T, 44100)

        return infos

    except Exception as e:
        print(f"Could not process {audio_name}. Error: {str(e)}\n{traceback.format_exc()}")
        traceback.print_exc()
        return None


def generate_dataset():
    infos = check_isvalid()
    infos += f"\nInput folder: {args.input_folder}, Output folder: {args.output_folder}"
    infos += f"\nTotal files found: {len(os.listdir(args.input_folder))}"

    print(f"Total files found: {len(os.listdir(args.input_folder))}")
    all_files = os.listdir(args.input_folder)

    if args.threads <= 1:
        all_files = tqdm(all_files)
        for audio in all_files:
            all_files.set_postfix_str(audio)
            info = process_audio(audio)
            if info is not None:
                infos += info
    else:
        p = multiprocessing.Pool(args.threads)
        for info in tqdm(p.imap(process_audio, all_files), total=len(all_files)):
            if info is not None:
                infos += info

    if args.save_logs:
        if args.generate_valid:
            with open(os.path.join(args.output_folder, "valid_datasets.txt"), "w") as f:
                f.write(infos)
        elif args.generate_train:
            with open(os.path.join(args.output_folder, "train_datasets.txt"), "w") as f:
                f.write(infos)


if __name__ == "__main__":
    generate_dataset()
    print("Dataset generation completed.")
