## <div align="center"> JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization </div>

<div align="center">

[[`HomePage`](https://javisdit.github.io/)] 
[[`ArXiv Paper`](https://arxiv.org/pdf/2503.23377)] 
[[`HF Paper`](https://huggingface.co/papers/2503.23377)]

</div>


<div align="center">
<video  controls="" autoplay="" muted="" loop="" width="99%" style="display: block; margin:0 auto">
                        <source src="./assets/video/teaser-video-JavisDit2.mp4" type="video/mp4">
                        Your browser does not support the video tag. 
</video>
</div>

## Abstract

We introduce **JavisDiT**, a novel Joint Audio-Video Diffusion Transformer designed for synchronized audio-video generation (JAVG) from open-ended user prompts. To ensure optimal synchronization, we introduce a fine-grained spatio-temporal alignment mechanism through a Hierarchical Spatial-Temporal Synchronized Prior (HiST-Sypo) Estimator. This module extracts both global and fine-grained spatio-temporal priors, guiding the synchronization between the visual and auditory components. Furthermore, we propose a new benchmark, **JavisBench**, consisting of 10,140 high-quality text-captioned sounding videos spanning diverse scenes and complex real-world scenarios. Further, we specifically devise a robust metric termed **JavisScore** for evaluating the synchronization between generated audio-video pairs in real-world complex content, with 3,000 human-annotated samples to quantitatively evaluate synchronization metrics themselves. We hope to set a new standard for the JAVG community.

![framework](./assets/image/framework.png)

## Code & Weight & Data

- [ ] Please stay tuned.


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
