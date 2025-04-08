---
title: JavisDiT
emoji: 🎥
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 4.25.0
app_file: app.py
pinned: false
license: apache-2.0
preload_from_hub:
    - JavisDiT/JavisDiT-v0.1-jav
    - JavisDiT/JavisDiT-v0.1-prior
    - hpcai-tech/OpenSora-VAE-v1.2
    - cvssp/audioldm2
    - DeepFloyd/t5-v1_1-xxl
---


# 🕹 Gradio Demo

We have provided a Gradio demo app for you to generate sounding videos via a web interface. You can choose to run it locally or deploy it to Hugging Face by following the instructions given below.

## 🚀 Run Gradio Locally

We assume that you have already installed `javisdit` based on the instructions given in the [main README](../README.md). Follow the steps below to run this app on your local machine.

1. First of all, you need to install `gradio` and `spaces`.

```bash
pip install gradio spaces
```

2. Afterwards, you can use the following command to launch the application. Remember to launch the command in the project root directory instead of the `gradio` folder.

```bash
# start the gradio app
python gradio/app.py

# run with a different port
python gradio/app.py --port 8000

# run with acceleration such as flash attention and fused norm
python gradio/app.py --enable-optimization

# run with a sharable Gradio link
python gradio/app.py --share
```

3. You should then be able to access this demo via the link which appears in your terminal.
