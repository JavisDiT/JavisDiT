
## Data Preparation


### Stage1 - JavisDiT-audio


In this stage, we only need audio files to initialize the audio generation capability:

| path | id | relpath | num_frames | height | width | aspect_ratio | fps | resolution | audio_path | audio_fps | text| audio_text|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| placeholder.mp4 | xxx | xxx.mp4 | 240 | 480 | 640 | 0.75 | 24 | 307200 | /path/to/xxx.wav | 16000 | placeholder | yyy |

Download the audios (including [AudioCaps](https://drive.google.com/file/d/16J1CVu7EZPD_22FxitZ0TpOd__FwzOmx/view?usp=drive_link), [VGGSound](https://huggingface.co/datasets/Loie/VGGSound), [AudioSet](https://huggingface.co/datasets/agkphysics/AudioSet), [WavCaps](ttps://huggingface.co/datasets/cvssp/WavCaps), [Clotho](https://zenodo.org/records/3490684), [ESC50](https://github.com/karolpiczak/ESC-50?tab=readme-ov-file#download), [MACS](https://zenodo.org/records/2589280), [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html), [MusicInstrument](https://www.kaggle.com/datasets/soumendraprasad/musical-instruments-sound-dataset), [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), etc.), and put them into the same folder `/path/to/audios`. Follow the commands to automatically generate a `train_audio.csv` for configuration:

```bash
ROOT_AUDIO="/path/to/audios"
ROOT_META="./data/meta/audio"

# 1.1 Create a meta file from a unified audio folder. This should output ${ROOT_META}/meta.csv
python -m tools.datasets.convert audio ${ROOT_AUDIO} --output ${ROOT_META}/meta.csv

# 1.2 Get audio information. This should output ${ROOT_META}/meta_ainfo.csv
python -m tools.datasets.datautil ${ROOT_META}/meta.csv --audio-info

# 2.1 Trim audios within 30 seconds. This should overwrite the raw audios by default and output ${ROOT_META}/meta_ainfo_trim30s.csv
python -m tools.datasets.datautil ${ROOT_META}/audio_meta.csv --trim-audio 30

# 2.2 Unify the sample rate to 16k Hz for all audios. This should output ${ROOT_META}/audio_meta_trim30s_sr16000.csv
python -m tools.datasets.datautil ${ROOT_META}/meta_ainfo_trim30s.csv --resample-audio --audio-sr 16000

# 3.1 Set dummy videos. This should output ${ROOT_META}/audio_meta_trim30s_sr16000_dummy_videos.csv
python -m tools.datasets.datautil ${ROOT_META}/audio_meta_trim30s_sr16000.csv --dummy-video

# 3.2 Get training meta csv. This should output ${ROOT_META}/train_audio.csv
python -m tools.datasets.find_audio_ds all \
    --data_root ${ROOT_AUDIO} \
    --meta_file ${ROOT_META}/audio_meta_trim30s_sr16000_dummy_videos.csv \
    --save_file ${ROOT_META}/train_audio.csv
```

### Stage2 - JavisDiT-prior

As detailed in our [paper](https://arxiv.org/pdf/2503.23377), the prior estimator is trainning with the contrastive learning paradigm. 
We take the extracted spatio-temporal priors as **anchor**, view the paired audio-video in the training datasets as **positive samples**, and randomly augment the audio or video to construct asychronized audio-video pairs as **negative samples**.
In particular, saptial- and temporal-asynchronization are separately generated.

| path | id | relpath | num_frames | height | width | aspect_ratio | fps | resolution | audio_path | audio_fps | text | unpaired_audio_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| /path/to/xxx.mp4 | xxx | xxx.mp4 | 240 | 480 | 640 | 0.75 | 24 | 307200 | /path/to/xxx.wav | 16000 | yyy | /path/to/zzz.wav |

#### Ground-truth synchronized audio-video pairs

Follow the instructions in [Stage3](#stage3---javisdit-jav) to read the audio-video information from training dataset (eg, [TAVGBench](https://github.com/OpenNLPLab/TAVGBench)).
The obtained basic meta file can be `/path/to/train_jav.csv`.


#### Offline asynchronized audio generation

Given a synchronized audio-video pair, we efficiently construct asynchronized audio-video pairs by generating standalone audios from [AudioLDM2](https://github.com/haoheliu/AudioLDM2) without reference videos. 
The native text descrption, native video, generated audio jointly contribute to an asynchronized (negative) sample for contrastive learning.
Generated audio paths will be recorded in the `unpaired_audio_path` column.

```bash
ROOT_META="./data/meta/prior"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4\
    tools/st_prior/gen_unpaired_audios.py \
    --input_meta ${ROOT_META}/train_jav.csv \
    --output_dir ./data/st_prior/audio/unpaired \
    --output_meta ${ROOT_META}/train_prior.csv \
    --match_duration
```

#### Online asynchronized audio-video augmentation

This part is implemented in `javisdit/datasets/augment.py`, where we developed various spatial/temporal augmentations for video/audio samples independently to constructing spatially/temporally asynchronized audio-video pairs.
For implementation details please kindly refer to our [paper](https://arxiv.org/pdf/2503.23377) and [code](javisdit/datasets/augment.py), and here we introduce the data preparation to perform corresponding augmentations:

- Auxiliary Video Resource ([SA-V](https://ai.meta.com/datasets/segment-anything-video/))

For video spatial augmentation, one of the efficient approaches is to randomly adding a sounding-object's masklet into a video sequence, causing spatial asynchrony between video and audio pairs.
Here we take the training set of [SA-V](https://ai.meta.com/datasets/segment-anything-video/) to collect native object maskelets at 6fps:

```
data/st_prior/video/SA_V/
├── sav_train
│   ├── sav_000
│   ├── sav_001
│   └── sav_002
```

Then, we utilize [GroundedSAM](https://github.com/zhengyuhang123/GroundedSAM.git) to extend 6fps annotations to 24fps masklets:

```bash
mkdir third_party && cd third_party

git clone https://github.com/zhengyuhang123/GroundedSAM.git

cd GroundedSAM

export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True

python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO

wget -P EfficientSAM/ https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_sam.pt

cd ../../

ls data/st_prior/video/SA_V/sav_train/sav_*/*.mp4 > data/st_prior/video/SA_V/sa_v_list.txt

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    tools/st_prior/get_masklets.py \
    --data_path data/st_prior/video/SA_V/sa_v_list.txt \
    --output_dir data/st_prior/video/SA_V/crops

ls data/st_prior/video/SA_V/crops/*.json > data/st_prior/video/SA_V/crops/pool_list.txt
```

The exracted masklets will be stored as:
```
data/st_prior/video/SA_V/crops/
├── pool_list.txt
├── sav_000001_mask_000.mp4
├── sav_000001_masklet_000.mp4
├── sav_000001_meta_000.json
├── sav_000002_mask_000.mp4
├── sav_000002_mask_001.mp4
├── sav_000002_masklet_000.mp4
├── sav_000002_masklet_001.mp4
├── sav_000002_meta_000.json
├── sav_000002_meta_001.json
├── ...
```

- Auxiliary Audio Resource ([AudioSep](https://github.com/Audio-AGI/AudioSep))

After seperating audio sources from original audio files, we can apply arbitrary addition and deletion operations on audios to introduce spatial asynchrony between video and audio pairs:

```bash
cd third_party
git clone https://github.com/Audio-AGI/AudioSep.git

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    tools/st_prior/sep_audios.py \
    --audio_path /path/to/TAVGBench \
    --output_path ./data/st_prior/audio/TAVGBench

ls data/st_prior/audio/TAVGBench/*.wav > data/st_prior/audio/TAVGBench/pool_list.txt
```


### Stage3 - JavisDiT-jav

Here we provide an example with [TAVGBench](https://github.com/OpenNLPLab/TAVGBench) to prepare video-audio-text triplets for training. You can easily transfer to your own datasets.

| path | id | relpath | num_frames | height | width | aspect_ratio | fps | resolution | audio_path | audio_fps | text|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
| /path/to/xxx.mp4 | xxx | xxx.mp4 | 240 | 480 | 640 | 0.75 | 24 | 307200 | /path/to/xxx.wav | 16000 | yyy |

With our cleaned [`release_captions_clean.txt`](https://huggingface.co/datasets/JavisDiT/TAVGBench_clean/tree/main) file, the following script will automatically generate a `train_jav.csv` for configuration:

```bash
ROOT_VIDEO="/path/to/videos"
ROOT_META="./data/meta/TAVGBench"

fmin=10  # minial frames for each video

# 1.1 Create a meta file from a video folder. This should output ${ROOT_META}/meta.csv
python -m tools.datasets.convert video ${ROOT_VIDEO} --output ${ROOT_META}/meta.csv

# 1.2 Get video information and remove broken videos. This should output ${ROOT_META}/meta_info_fmin${fmin}.csv
python -m tools.datasets.datautil ${ROOT_META}/meta.csv --info --fmin ${fmin}

# 2.1 Unify FPS to 24 Hz for all videos. This will change the raw videos, and output ${ROOT_META}/meta_info_fmin${fmin}_fps24.csv
python -m tools.datasets.datautil ${ROOT_META}/meta_info_fmin${fmin}.csv --uni-fps 24 --overwrite

# 2.2 Extract audios from videos, and fix the sample rate to 16k Hz for all audios. This should output ${ROOT_META}/meta_info_fmin${fmin}_fps24_au_sr16000.csv
python -m tools.datasets.datautil ${ROOT_META}/meta_info_fmin${fmin}_fps24.csv --extract-audio --audio-sr 16000

# 3.1 Get training meta csv. This should output ${ROOT_META}/train_jav.csv
python -m tools.datasets.find_jav_ds tavgbench \
    --meta_src /path/to/TAVGBench_clean/release_captions_clean.txt \
    --meta_file ${ROOT_META}/meta_info_fmin${fmin}_fps24_au_sr16000.csv \
    --save_file ${ROOT_META}/train_jav.csv
```

If you get multiple data sources, just merge the csv files to a single one:
```bash
python -m tools.datasets.datautil ds1.csv ds2.csv ... --output /path/to/output.csv
```