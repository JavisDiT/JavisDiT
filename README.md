## <div align="center"> JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization</div>

<div align="center">

[[`HomePage`](https://javisdit.github.io/)] 
[[`ArXiv Paper`](https://arxiv.org/pdf/2503.23377)] 
[[`HF Paper`](https://huggingface.co/papers/2503.23377)]
[[`Models`](https://huggingface.co/collections/JavisDiT/javisdit-v01-67f2ac8a0def71591f7e2974)]
<!-- [[`Gradio Demo`](https://447c629bc8648ce599.gradio.live)] -->

</div>


We introduce **JavisDiT**, a novel & SoTA Joint Audio-Video Diffusion Transformer designed for synchronized audio-video generation (JAVG) from open-ended user prompts. 

https://github.com/user-attachments/assets/de5f0bcc-fb5d-4410-a795-2dd3ae3ac788

<!-- <video controls width="100%">
  <source src="assets/video/teaser-video-JavisDit3.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video> -->

## 📰 News

- **[2025.04.15]** 🔥 We released the data preparation and model training instructions. You can train JavisDiT with your own dataset!
- **[2025.04.07]** 🔥 We released the inference code and a preview model of **JavisDiT-v0.1** at [HuggingFace](https://huggingface.co/JavisDiT), which includes **JavisDiT-v0.1-audio**, **JavisDiT-v0.1-prior**, and **JavisDiT-v0.1-jav** (with a [low-resolution version](https://huggingface.co/JavisDiT/JavisDiT-v0.1-jav-240p4s) and a [full-resolution version](https://huggingface.co/JavisDiT/JavisDiT-v0.1-jav)).
- **[2025.04.03]** We release the repository of [JavisDiT](https://arxiv.org/pdf/2503.23377). Code, model, and data are coming soon.

### 👉 TODO 
- [ ] Release the data and evaluation code for JavisBench & JavisScore.
- [ ] Deriving a more efficient and powerful JAVG model.

## Brief Introduction

**JavisDiT** addresses the key bottleneck of JAVG with Hierarchical Spatio-Temporal Prior Synchronization.

<!-- <p align="center">
  <img src="./assets/image/JavisDiT-intro-resized.png" width="550"/>
</p> -->

![framework](./assets/image/JavisDiT-framework-resized.png)

- We introduce **JavisDiT**, a novel Joint Audio-Video Diffusion Transformer designed for synchronized audio-video generation (JAVG) from open-ended user prompts. 
- We propose **JavisBench**, a new benchmark consisting of 10,140 high-quality text-captioned sounding videos spanning diverse scenes and complex real-world scenarios. 
- We devise **JavisScore**, a robust metric for evaluating the synchronization between generated audio-video pairs in real-world complex content.
- We curate **JavisEval**, a dataset with 3,000 human-annotated samples to quantitatively evaluate the accuracy of synchronization estimate metrics. 

We hope to set a new standard for the JAVG community. For more technical details, kindly refer to the original [paper](https://arxiv.org/pdf/2503.23377.pdf). 


## Installation

### Install from Source

For CUDA 12.1, you can install the dependencies with the following commands.

```bash
# create a virtual env and activate (conda as an example)
conda create -n javisdit python=3.10
conda activate javisdit

# download the repo
git clone https://github.com/JavisDiT/JavisDiT
cd JavisDiT

# install torch, torchvision and xformers
pip install -r requirements/requirements-cu121.txt

# the default installation is for inference only
pip install -v .
# for development mode, `pip install -v -e .`
# to skip dependencies, `pip install -v -e . --no-deps`

# replace
cp assets/src/pytorchvideo_augmentations.py /path/to/python3.10/site-packages/pytorchvideo/transforms/augmentations.py
```

(Optional, recommended for fast speed, especially for training) To enable `layernorm_kernel` and `flash_attn`, you need to install `apex` and `flash-attn` with the following commands.

```bash
# install flash attention
# set enable_flash_attn=False in config to disable flash attention
pip install packaging ninja
pip install flash-attn --no-build-isolation

# install apex
# set enable_layernorm_kernel=False in config to disable apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git
```


### Pre-trained Weights


| Model     | Resolution | Model Size | Data | #iterations | Batch Size |
| --------- | ---------- | ---------- | ---- | ----------- | ---------- |
| [JavisDiT-v0.1-prior](https://huggingface.co/JavisDiT/JavisDiT-v0.1-prior)  | 144P-1080P | 29M  | 611K | 36k | Dynamic |
| [JavisDiT-v0.1](https://huggingface.co/JavisDiT/JavisDiT-v0.1-jav)        | 144P-1080P | 3.4B | 611K | 1k  | Dynamic |
| [JavisDiT-v0.1-240p4s](https://huggingface.co/JavisDiT/JavisDiT-v0.1-jav-240p4s) | 240P       | 3.4B | 611K | 16k | 4       |


:warning: **LIMITATION**: [JavisDiT-v0.1](https://huggingface.co/collections/JavisDiT/javisdit-v01-67f2ac8a0def71591f7e2974) is a preview version trained on a limited budget. We are working on improving the quality by optimizing both model architecture and training data.

Weight will be automatically downloaded when you run the inference script. Or you can also download these weights to local directory and change the path configuration in `configs/.../inference/sample.py`.

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download JavisDiT/JavisDiT-v0.1-jav --local-dir ./checkpoints/JavisDiT-v0.1-jav
```

> For users from mainland China, try `export HF_ENDPOINT=https://hf-mirror.com` to successfully download the weights.


## Inference

### Weight Prepare

Download [imagebind_huge.pth](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) and put it into `./checkpoints/imagebind_huge.pth`.

### Command Line Inference

The basic command line inference is as follows:

```bash
python scripts/inference.py \
  configs/javisdit-v0-1/inference/sample.py \
  --num-frames 2s --resolution 720p --aspect-ratio 9:16 \
  --prompt "a beautiful waterfall" --verbose 2
```

`--verbose 2` will display the progress of a single diffusion.
If your installation do not contain `apex` and `flash-attn`, you need to disable them in the config file, or via the folowing command.

```bash
python scripts/inference.py \
  configs/javisdit-v0-1/inference/sample_240p4s.py \
  --num-frames 2s --resolution 720p --aspect-ratio 9:16 \
  --layernorm-kernel False --flash-attn False \
  --prompt "a beautiful waterfall" --verbose 2
```

Try this configuration to generate low-resolution sounding-videos:

```bash
python scripts/inference.py \
  configs/javisdit-v0-1/inference/sample_240p4s.py \
  --num-frames 4s --resolution 240p --aspect-ratio 9:16 \
  --prompt "a beautiful waterfall" --verbose 2
```

If you want to generate on a given prompt list (organized with a `.txt` for `.csv` file):

```bash
python scripts/inference.py \
  configs/javisdit-v0-1/inference/sample_240p4s.py \
  --num-frames 4s --resolution 240p --aspect-ratio 9:16 \
  --prompt-path data/meta/JavisBench.csv --verbose 1
```

`--verbose 1` will display the progress of the whole generation list.

### Multi-Device Inference

To enable multi-device inference, you need to use `torchrun` to run the inference script. The following command will run the inference with 2 GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 scripts/inference.py \
  configs/javisdit-v0-1/inference/sample_240p4s.py \
  --num-frames 4s --resolution 240p --aspect-ratio 9:16 \
  --prompt-path data/meta/JavisBench.csv --verbose 1
```

### X-Conditional Generation

- [ ] Coming soon.

## Training 

### Data Preparation

In this project, we use a `.csv` file to manage all the training entries and their attributes for efficient training:

| path | id | relpath | num_frames | height | width | aspect_ratio | fps | resolution | audio_path | audio_fps | text|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
| /path/to/xxx.mp4 | xxx | xxx.mp4 | 240 | 480 | 640 | 0.75 | 24 | 307200 | /path/to/xxx.wav | 16000 | yyy |

The content of columns may vary in different training stages. The detailed instructions for each training stage can be found in [here](assets/docs/data.md).

### Stage1 - JavisDiT-audio

In this stage, we perform audio pretraining to intialize the text-to-audio generation capability:

```bash
ln -s /path/to/local/OpenSora-STDiT-v3 ./checkpoints/OpenSora-STDiT-v3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --standalone --nproc_per_node 8 \
    scripts/train.py \
    configs/javisdit-v0-1/train/stage1_audio.py \
    --data-path data/meta/audio/train_audio.csv
```

The resulting checkpoints will be saved at `runs/0aa-VASTDiT3-XL-2/epoch0bb-global_stepccc/model`.

### Stage2 - JavisDiT-prior

In this stage, we estimate the spatio-temporal synchronization prior under a contrastive learning framewrok:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --standalone --nproc_per_node 8 \
    scripts/train_prior.py \
    configs/javisdit-v0-1/train/stage2_prior.py \
    --data-path data/meta/prior/train_prior.csv
```

The resulting checkpoints will be saved at `runs/0xx-STIBPrior/epoch0yy-global_stepzzz/model`.

### Stage3 - JavisDiT-jav

In this stage, we freeze the previously learned modules, and train the audio-video synchronization modules:

```bash
# link to previous stages
mv runs/0aa-VASTDiT3-XL-2/epoch0bb-global_stepccc checkpoints/JavisDiT-v0.1-audio
mv runs/0xx-STIBPrior/epoch0yy-global_stepzzz checkpoints/JavisDiT-v0.1-prior

# start training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --standalone --nproc_per_node 8 \
    scripts/train.py \
    configs/javisdit-v0-1/train/stage3_jav.py \
    --data-path data/meta/TAVGBench/train_jav.csv
```

The resulting checkpoints will be saved at `runs/0aa-VASTDiT3-XL-2/epoch0bb-global_stepccc/model`.

```bash
mv runs/0aa-VASTDiT3-XL-2/epoch0bb-global_stepccc checkpoints/JavisDiT-v0.1-jav
```

## Evaluation

- [ ] Coming soon.


## Acknowledgement

Below we show our appreciation for the exceptional work and generous contribution to open source. Special thanks go to the authors of [Open-Sora](https://github.com/hpcaitech/Open-Sora) and [TAVGBench](https://github.com/OpenNLPLab/TAVGBench) for their valuable codebase and dataset. For other works and datasets, please refer to our paper.

- [Open-Sora](https://github.com/hpcaitech/Open-Sora): A wonderful project for democratizing efficient text-to-video production for all, with the model, tools and all details accessible.
- [TAVGBench](https://github.com/OpenNLPLab/TAVGBench): A large-scale dataset encompasses an impressive 1.7 million video-audio entries, each meticulously annotated with corresponding text.
- [ColossalAI](https://github.com/hpcaitech/ColossalAI): A powerful large model parallel acceleration and optimization system.
- [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers.
- [OpenDiT](https://github.com/NUS-HPC-AI-Lab/OpenDiT): An acceleration for DiT training. We adopt valuable acceleration strategies for training progress from OpenDiT.
- [PixArt](https://github.com/PixArt-alpha/PixArt-alpha): An open-source DiT-based text-to-image model.
- [Latte](https://github.com/Vchitect/Latte): An attempt to efficiently train DiT for video.
- [StabilityAI VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse-original): A powerful image VAE model.
- [CLIP](https://github.com/openai/CLIP): A powerful text-image embedding model.
- [T5](https://github.com/google-research/text-to-text-transfer-transformer): A powerful text encoder.

## Citation

If you find JavisDiT is useful and use it in your project, please kindly cite:

```
@inproceedings{liu2025javisdit,
      title={JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization}, 
      author={Kai Liu and Wei Li and Lai Chen and Shengqiong Wu and Yanhao Zheng and Jiayi Ji and Fan Zhou and Rongxin Jiang and Jiebo Luo and Hao Fei and Tat-Seng Chua},
      booktitle={arxiv},
      year={2025}, 
}
```

<!-- ---

# ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=JavisDiT/JavisDiT&type=Date)](https://star-history.com/#JavisDiT/JavisDiT&Date) -->

