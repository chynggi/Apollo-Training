<div align="center">
  
# Apollo Training

</div>

> [!WARNING]
> 目前已知问题：
> 1. 还未在多卡环境跑过此训练代码！目前仅在Linux和Windows下使用此代码单卡训练过。
> 2. 目前已知问题：`batchsize>1`会出现tensor不匹配的情况。还没研究为什么会这样。因此建议设置`batchsize=1`，如果要压榨显存的话，把切片长度`segments`调大一点，或者调大模型网络，就可以了。

## 1. 环境配置

经测试，python=3.10可以运行，其他版本未测试。此外，建议手动安装PyTorch。

```shell
conda create -n apollo python=3.10 -y
conda activate apollo
pip install -r requirements.txt
```

如果在后续训练过程中遇到报错：`RuntimeError: use_libuv was requested but PyTorch was build without libuv support`，有以下两种解决方法：
1. 降低pytorch的版本，经过测试，torch==2.0.1可以运行。
2. 在 `train.py` 的 `if __name__ == "__main__":` 中，将 `init_method="env://"` 修改为 `init_method="env://?use_libuv=False"`。

## 2. 数据集构建

### 2.1 手动构建压缩后的音频

按照以下结构构建训练集文件夹。codec代表的是压缩后的音频，original代表的是原始音频。你需要确保original文件夹中的音频文件和codec文件夹中的音频文件，除后缀名以外的其余名称是一一对应的。并且需要确保配置文件夹中 `datas.codec.enable` 设置成 `False` 以禁用自动构建压缩音频。

```
train
  ├─codec
  │    my_song.wav
  │    test_song.wav
  │    vocals.wav
  │    114514.wav
  │    ...
  └─original
       my_song.wav
       test_song.wav
       vocals.wav
       114514.wav
       ...
```

### 2.2 自动构建压缩后的音频

按照以下结构构建训练集文件夹，无需codec文件夹。并且需要确保配置文件夹中 `datas.codec.enable` 设置成 `True` 以启用自动构建压缩音频。

```
train
  └─original
       my_song.wav
       test_song.wav
       vocals.wav
       114514.wav
       ...
```

如果在自动构建的过程中遇到`RuntimeError: torchaudio.functional.functional.apply_codec requires sox extension, but TorchAudio is not compiled with it. Please build TorchAudio with libsox support.`，则表明自动构建不可用。请自行解决或使用上面的手动构建压缩音频的方法。

### 2.3 验证集构建

无论上面选择何种方式，都需要按照以下结构构建验证集文件夹。并且需要保证同一文件夹中的两段音频形状（`audio.shape`）保持一致。文件夹名字可以自定义，音频文件名字需要一致。

```
valid
  ├─folder_1
  │    codec.wav
  │    original.wav
  │    ...
  └─folder_2
       codec.wav
       original.wav
       ...
```

### 2.4 修改配置文件

配置文件位于`configs/apollo.yaml`，下面仅介绍一些关键参数

```yaml
exp: 
  dir: ./exps # 训练结果存放路径
  name: apollo # 实验名称
  # 上面两行加起来，即会在./exps/apollo中存放此次训练的结果和日志

datas:
  _target_: look2hear.datas.DataModule
  original_dir: train/original # 训练集，存放原始音频的文件夹
  codec_dir: train/codec # 训练集，存放压缩音频的文件夹
  codec_format: mp3 # 训练集，存放压缩音频的文件夹中的音频格式
  valid_dir: valid # 验证集路径
  valid_original: original.wav # 验证集中原始音频的文件名
  valid_codec: codec.mp3 # 验证集中压缩音频的文件名
  codec:
    enable: false # 自动生成压缩音频，如果启用，将自动生成压缩音频。上面的codec_dir和codec_format将被忽略
    options: # 压缩参数设置
      bitrate: random # 随机或固定，如果固定，则采用设定的值（整型），如果随机，则将从[24000、32000、48000、64000、96000、128000]中随机选择比特率
      compression: random # 随机或固定，如果固定，则采用设定的值（整型），如果随机，将按比特率计算
  sr: 44100 # 采样率
  segments: 3 # 训练时随机裁剪的音频长度（单位：秒）。该值应小于训练集中最短音频时长
  num_steps: 1000 # 一个epoch中的迭代次数，也可理解为一个epoch中随机抽取的音频数量
  batch_size: 1
  num_workers: 0
  pin_memory: true

model:
  _target_: look2hear.models.apollo.Apollo
  sr: 44100 # sample rate
  win: 20 # ms
  feature_dim: 256 # feature dimension
  layer: 6 # number of layers

trainer:
  _target_: pytorch_lightning.Trainer
  devices: [0] # GPU ID
  max_epochs: 1000 # 最大训练轮数
  sync_batchnorm: true
  default_root_dir: ${exp.dir}/${exp.name}/
  accelerator: cuda
  limit_train_batches: 1.0
  fast_dev_run: false
  precision: bf16 # 可选项：[16, bf16, 32, 64]，建议采用bf16
```

## 3. 训练

> [!WARNING]
> 目前已知问题：
> 1. 还未在多卡环境跑过此训练代码！目前仅在Linux和Windows下使用此代码单卡训练过。
> 2. 目前已知问题：`batchsize>1`会出现tensor不匹配的情况。还没研究为什么会这样。因此建议设置`batchsize=1`，如果要压榨显存的话，把切片长度`segments`调大一点，或者调大模型网络，就可以了。

使用下面的代码开始训练。若需要wandb在线可视化，需设置环境变量`WANDB_API_KEY`为你的api key。

```bash
python train.py -c [配置文件路径]
# 例如：python train.py -c ./configs/apollo.yaml
```

如果需要继续训练，添加 `-m [继续训练的模型路径]`。但还未经过充分测试。<br>
关于更详细的多卡分布式训练的环境变量设置，前往 `train.py` 的 `if __name__ == "__main__":`。

## 4. 推理/验证

> [!NOTE]
> 更推荐使用[ZFTurbo](https://github.com/ZFTurbo)的[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)进行模型推理和验证。

apollo官方也提供了简单的推理脚本 `inference.py`。使用方法:

```bash
python inference.py -m [模型路径] -i [输入音频路径] -o [输出音频路径]
# 例如：python inference.py -m ./exps/apollo/epoch=0001-step=0000000.ckpt -i ./test.wav -o ./test_out.wav
```

## 5. 导出[msst](https://github.com/ZFTurbo/Music-Source-Separation-Training)模型和配置文件

由此仓库训练出来的apollo模型无法直接在msst中使用，需要进行一些转换。使用 `generate_msst.py`。该脚本可以删除模型中的无用参数，并且转换成[msst](https://github.com/ZFTurbo/Music-Source-Separation-Training)支持的模型。运行下述命令后，会在输出文件夹输出配置文件和模型文件。

```bash
python generate_msst.py -c [apollo配置文件路径] -m [训练出来的apollo模型路径] -o [输出文件夹路径，默认为output]
# 例如：python generate_msst.py -c ./configs/apollo.yaml -m ./exps/apollo/epoch=0001-step=0000000.ckpt
```

----

<div align="center">

# Apollo: Band-sequence Modeling for High-Quality Audio Restoration

  <strong>Kai Li<sup>1,2</sup>, Yi Luo<sup>2</sup></strong><br>
    <strong><sup>1</sup>Tsinghua University, Beijing, China</strong><br>
    <strong><sup>2</sup>Tencent AI Lab, Shenzhen, China</strong><br>
  <a href="https://arxiv.org/abs/2409.08514">ArXiv</a> | <a href="https://cslikai.cn/Apollo/">Demo</a>
</div>

## 📖 Abstract

Audio restoration has become increasingly significant in modern society, not only due to the demand for high-quality auditory experiences enabled by advanced playback devices, but also because the growing capabilities of generative audio models necessitate high-fidelity audio. Typically, audio restoration is defined as a task of predicting undistorted audio from damaged input, often trained using a GAN framework to balance perception and distortion. Since audio degradation is primarily concentrated in mid- and high-frequency ranges, especially due to codecs, a key challenge lies in designing a generator capable of preserving low-frequency information while accurately reconstructing high-quality mid- and high-frequency content. Inspired by recent advancements in high-sample-rate music separation, speech enhancement, and audio codec models, we propose Apollo, a generative model designed for high-sample-rate audio restoration. Apollo employs an explicit **frequency band split module** to model the relationships between different frequency bands, allowing for **more coherent and higher-quality** restored audio. Evaluated on the MUSDB18-HQ and MoisesDB datasets, Apollo consistently outperforms existing SR-GAN models across various bit rates and music genres, particularly excelling in complex scenarios involving mixtures of multiple instruments and vocals. Apollo significantly improves music restoration quality while maintaining computational efficiency.

## 🔥 News

- [2024.09.10] Apollo is now available on [ArXiv](#) and [Demo](https://cslikai.cn/Apollo/).
- [2024.09.106] Apollo checkpoints and pre-trained models are available for download.

## ⚡️ Installation

clone the repository

```bash
git clone https://github.com/JusperLee/Apollo.git && cd Apollo
conda create --name look2hear --file look2hear.yml
conda activate look2hear
```

## 🖥️ Usage

### 🗂️ Datasets

Apollo is trained on the MUSDB18-HQ and MoisesDB datasets. To download the datasets, run the following commands:

```bash
wget https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1
wget https://ds-website-downloads.55c2710389d9da776875002a7d018e59.r2.cloudflarestorage.com/moisesdb.zip
```
During data preprocessing, we drew inspiration from music separation techniques and implemented the following steps:

1. **Source Activity Detection (SAD):**  
   We used a Source Activity Detector (SAD) to remove silent regions from the audio tracks, retaining only the significant portions for training.

2. **Data Augmentation:**  
   We performed real-time data augmentation by mixing tracks from different songs. For each mix, we randomly selected between 1 and 8 stems from the 11 available tracks, extracting 3-second clips from each selected stem. These clips were scaled in energy by a random factor within the range of [-10, 10] dB relative to their original levels. The selected clips were then summed together to create simulated mixed music.

3. **Simulating Dynamic Bitrate Compression:**  
   We simulated various bitrate scenarios by applying MP3 codecs with bitrates of [24000, 32000, 48000, 64000, 96000, 128000]. 

4. **Rescaling:**  
   To ensure consistency across all samples, we rescaled both the target and the encoded audio based on their maximum absolute values.

5. **Saving as HDF5:**  
   After preprocessing, all data (including the source stems, mixed tracks, and compressed audio) was saved in HDF5 format, making it easy to load for training and evaluation purposes.

### 🚀 Training
To train the Apollo model, run the following command:

```bash
python train.py --conf_dir=configs/apollo.yml
```

### 🎨 Evaluation
To evaluate the Apollo model, run the following command:

```bash
python inference.py --in_wav=assets/input.wav --out_wav=assets/output.wav
```

## 📊 Results

*Here, you can include a brief overview of the performance metrics or results that Apollo achieves using different bitrates*

![](./asserts/bitrates.png)


*Different methods' SDR/SI-SNR/VISQOL scores for various types of music, as well as the number of model parameters and GPU inference time. For the GPU inference time test, a music signal with a sampling rate of 44.1 kHz and a length of 1 second was used.*
![](./asserts/types.png)

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Third Party

[Apollo-Colab-Inference](https://github.com/jarredou/Apollo-Colab-Inference)

## Acknowledgements

Apollo is developed by the **Look2Hear** at Tsinghua University.

## Citation

If you use Apollo in your research or project, please cite the following paper:

```bibtex
@inproceedings{li2025apollo,
  title={Apollo: Band-sequence Modeling for High-Quality Music Restoration in Compressed Audio},
  author={Li, Kai and Luo, Yi},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2025},
  organization={IEEE}
}
```

## Contact

For any questions or feedback regarding Apollo, feel free to reach out to us via email: `tsinghua.kaili@gmail.com`
