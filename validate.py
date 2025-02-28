import os
import numpy as np
import glob
import librosa
import torch
import soundfile as sf
from look2hear.utils import get_metrics
from inference import ApolloSeparator


def process(args) -> None:
    input_folder = args.input
    output_folder = args.output

    files = []
    path = os.path.abspath(input_folder)
    for folder in os.listdir(path):
        if not os.path.isdir(os.path.join(path, folder)):
            continue
        original = glob.glob(os.path.join(path, folder, "original.*"))
        codec = glob.glob(os.path.join(path, folder, "codec.*"))
        if len(original) != 1 or len(codec) != 1:
            print(f"Warning: Folder {folder} is invalid, please check!")
            continue
        files.append({"original": original[0], "codec": codec[0], "folder": folder})
    print(f"Total files found: {len(files)}")

    separator = ApolloSeparator(
        model_path=args.model,
        config_path=args.config,
        segments=args.segments,
        overlap=args.overlap,
        batch_size=args.batch_size
    )

    all_metrics = dict()
    for metric in args.metrics:
        all_metrics[metric] = []

    if args.save_results:
        os.makedirs(output_folder, exist_ok=True)
        out = open(os.path.join(output_folder, "results.txt"), 'w')
        out.write(str(args))
        out.write("\n")

    for file in files:
        print(f"\nProcessing {file['codec']}")
        reference = librosa.load(file["original"], sr=separator.config.model.sr, mono=False)[0]
        mixture = librosa.load(file["codec"], sr=separator.config.model.sr, mono=False)[0]
        reference = vaild_audio(reference)
        mixture = vaild_audio(mixture)
        estimate = separator.demix_track(torch.tensor(mixture, dtype=torch.float32))["restored"]
        metrics = get_metrics(args.metrics, reference, estimate, mixture)

        if args.save_results:
            out.write("\n" + file["codec"] + "\n")
            save_name = os.path.join(output_folder, file["folder"] + "_restored.wav")
            sf.write(save_name, estimate.T, separator.config.model.sr)

        for metric_name in metrics:
            metric_value = metrics[metric_name]
            print("Metric {:11s} value: {:.4f}".format(metric_name, metric_value))
            all_metrics[metric_name].append(metric_value)
            if args.save_results:
                out.write("Metric {:11s} value: {:.4f}\n".format(metric_name, metric_value))

    if args.save_results:
        out.write("\nSummary:\n")

    print("\nSummary:")
    for metric_name in all_metrics:
        metric_values = np.array(all_metrics[metric_name])
        mean_val = metric_values.mean()
        std_val = metric_values.std()
        print("{}: {:.4f} (Std: {:.4f})".format(metric_name, mean_val, std_val))
        if args.save_results:
            out.write("{}: {:.4f} (Std: {:.4f})\n".format(metric_name, mean_val, std_val))

    if args.save_results:
        out.close()

def vaild_audio(mix: np.ndarray) -> np.ndarray:
    if len(mix.shape) == 1: # if model is stereo, but track is mono, add a second channel
        mix = np.stack([mix, mix], axis=0)
        print(f"Track is mono, but model is stereo, adding a second channel.")
    elif len(mix.shape) != 1 and mix.shape[0] > 2: # fi model is stereo, but track has more than 2 channels, take mean
        mix = np.mean(mix, axis=0)
        mix = np.stack([mix, mix], axis=0)
        print(f"Track has more than 2 channels, taking mean of all channels and adding a second channel.")
    return mix


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/apollo.yaml", help="Path to the config file")
    parser.add_argument("-m", "--model", type=str, default=None, help="Path to the checkpoint model for validation")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input audio file")
    parser.add_argument("--metrics", nargs='+', type=str, default=['sdr', 'si_sdr', 'l1_freq', 'log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless', 'fullness'], choices=['sdr', 'si_sdr', 'l1_freq', 'log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless', 'fullness'], help='List of metrics to use.')
    parser.add_argument("--segments", type=float, default=10, help="Segment size in seconds")
    parser.add_argument("--overlap", type=int, default=4, help="Number of overlapping chunks")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--save_results", action="store_true", help="Save the result audio and metrics")
    parser.add_argument("--output", type=str, default="output", help="Path to save the results")
    args = parser.parse_args()

    process(args)
    print("Validation finished!")