import os
import torch
import numpy as np
import torch.nn as nn
import librosa
import soundfile as sf
import warnings
from tqdm import tqdm
from omegaconf import OmegaConf
from look2hear.models import Apollo

warnings.filterwarnings("ignore")


class ApolloSeparator:
    def __init__(
        self, 
        model_path: str,
        config_path: str,
        segments: float = 10.0,
        overlap: int = 4,
        batch_size: int = 1
    ) -> None:

        self.config = OmegaConf.load(config_path)
        self.chunk_size = int(self.config.model.sr * segments)
        self.overlap = overlap
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_config = {
            "sr": self.config.model.sr,
            "win": self.config.model.win,
            "feature_dim": self.config.model.feature_dim,
            "layer": self.config.model.layer
        }
        self.model = Apollo(**model_config)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)["state_dict"]
        state_dict = {k.replace("audio_model.", ""): v for k, v in state_dict.items() if "discriminator" not in k}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        print(f"Loaded model from {model_path}")

    def process_folder(self, input_folder: str, output_folder: str, save_addition: bool = False) -> None:
        if os.path.isfile(input_folder):
            print(f"Starting separation process for audio_file: {input_folder}")
            self.separate(input_folder, output_folder, save_addition)
            print(f"Successfully separated audio {input_folder}")
        else:
            print(f"Total audio files found: {len(os.listdir(input_folder))}")
            all_files = tqdm(os.listdir(input_folder), desc="Processing", leave=False)
            for audio_file in all_files:
                try:
                    all_files.set_postfix_str(audio_file)
                    self.separate(os.path.join(input_folder, audio_file), output_folder, save_addition)
                except Exception as e:
                    print(f"Error separating audio file {audio_file}: {e}, skipping...")
            print(f"Separate finished!")

    def separate(self, audio_path: str, output_path: str, save_addition: bool = False) -> None:
        os.makedirs(output_path, exist_ok=True)
        sample_rate = self.config.model.sr
        mix, sr = librosa.load(audio_path, sr=sample_rate, mono=False)

        if len(mix.shape) == 1: # if model is stereo, but track is mono, add a second channel
            mix = np.stack([mix, mix], axis=0)
            print(f"Track is mono, but model is stereo, adding a second channel.")
        elif len(mix.shape) != 1 and mix.shape[0] > 2: # fi model is stereo, but track has more than 2 channels, take mean
            mix = np.mean(mix, axis=0)
            mix = np.stack([mix, mix], axis=0)
            print(f"Track has more than 2 channels, taking mean of all channels and adding a second channel.")

        mix_orig = mix.copy()
        results = self.demix_track(torch.tensor(mix, dtype=torch.float32))

        path_restored = os.path.join(output_path, f"{os.path.basename(audio_path).split('.')[0]}_restored.wav")
        path_addition = os.path.join(output_path, f"{os.path.basename(audio_path).split('.')[0]}_addition.wav")
        sf.write(path_restored, results["restored"].T, sr, subtype='FLOAT')
        if save_addition:
            other = mix_orig - results["restored"]
            sf.write(path_addition, other.T, sr, subtype='FLOAT')

    def demix_track(self, mix) -> dict[str, np.ndarray]:
        C = self.chunk_size
        N = self.overlap
        fade_size = C // 10
        step = int(C // N)
        border = C - step
        batch_size = self.batch_size
        length_init = mix.shape[-1]

        # Do pad from the beginning and end to account floating window results better
        if length_init > 2 * border and (border > 0):
            mix = nn.functional.pad(mix, (border, border), mode='reflect')

        # Prepare windows arrays (do 1 time for speed up). This trick repairs click problems on the edges of segment
        window_size = C
        fadein = torch.linspace(0, 1, fade_size)
        fadeout = torch.linspace(1, 0, fade_size)
        window_start = torch.ones(window_size)
        window_middle = torch.ones(window_size)
        window_finish = torch.ones(window_size)
        window_start[-fade_size:] *= fadeout # First audio chunk, no fadein
        window_finish[:fade_size] *= fadein # Last audio chunk, no fadeout
        window_middle[-fade_size:] *= fadeout
        window_middle[:fade_size] *= fadein

        with torch.amp.autocast('cuda'):
            with torch.inference_mode():
                req_shape = (1, ) + tuple(mix.shape)
                result = torch.zeros(req_shape, dtype=torch.float32)
                counter = torch.zeros(req_shape, dtype=torch.float32)
                i = 0
                batch_data = []
                batch_locations = []
                progress_bar = tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False)

                while i < mix.shape[1]:
                    part = mix[:, i:i + C].to(self.device)
                    length = part.shape[-1]
                    if length < C:
                        if length > C // 2 + 1:
                            part = nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
                        else:
                            part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                    batch_data.append(part)
                    batch_locations.append((i, length))
                    i += step

                    if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                        arr = torch.stack(batch_data, dim=0)
                        x = self.model(arr)
                        window = window_middle
                        if i - step == 0:  # First audio chunk, no fadein
                            window = window_start
                        elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                            window = window_finish
                        for j in range(len(batch_locations)):
                            start, l = batch_locations[j]
                            result[..., start:start+l] += x[j][..., :l].cpu() * window[..., :l]
                            counter[..., start:start+l] += window[..., :l]
                        batch_data = []
                        batch_locations = []
                    progress_bar.update(step)
                progress_bar.close()

                estimated_sources = result / counter
                estimated_sources = estimated_sources.cpu().numpy()
                np.nan_to_num(estimated_sources, copy=False, nan=0.0)
                if length_init > 2 * border and (border > 0):
                    estimated_sources = estimated_sources[..., border:-border]
        return {k: v for k, v in zip(["restored"], estimated_sources)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the model config")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input audio file or folder")
    parser.add_argument("-o", "--output", type=str, default="output", help="Path to the output folder")
    parser.add_argument("--segments", type=float, default=10, help="Segment size in seconds")
    parser.add_argument("--overlap", type=int, default=4, help="Number of overlapping chunks")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--save_addition", action="store_true", help="Save the addition track")
    args = parser.parse_args()
    print(args)

    separator = ApolloSeparator(args.model, args.config, args.segments, args.overlap, args.batch_size)
    separator.process_folder(args.input, args.output, args.save_addition)
