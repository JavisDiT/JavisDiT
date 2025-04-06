import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import os.path as osp
import shutil
import math
from tqdm import tqdm
from matplotlib.colors import Normalize

# === 音频部分 ===
def plot_mel_spectrogram(
    audio_path, start_s, end_s,
    sr=22050, hop_length=512, n_mels=128, fmin=50, fmax=8000, ref_db=-80,
):
    # 加载音频
    y, sr = librosa.load(audio_path, sr=sr)  
    y = y[int(start_s*sr):int(end_s*sr)]
    # 计算 Mel 频谱
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    # 转换为对数幅度，并增强对比
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = np.clip(mel_spec_db, ref_db, 0)  # 限制动态范围

    # 绘制 Mel 频谱
    fig, ax = plt.subplots(figsize=(2*(end_s-start_s), 2))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    return fig

# === 视频部分 ===
def extract_video_frames(video_path, start_s=0, end_s=None, sample_rate=6):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    start, end = int(start_s * fps), int(end_s * fps) if end_s else total_frames
    frame_indices = np.arange(start, end, step=sample_rate)
    # num_frames = int((end_s - start_s) * fps / sample_rate)
    # frame_indices = np.linspace(int(start_s*fps), int(end_s*fps), num_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 转换为RGB格式
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames

def extract_av(
    video_path='/home/haofei/kailiu/datasets/SyncVA-Bench/youtube/clip/2IdNpHQOaJQ_scene-4.mp4', 
    frame_output_dir='debug/frames', 
    output_path='debug/output.pdf',
):
    audio_path = video_path.replace('.mp4', '.wav')
    shutil.rmtree(frame_output_dir) if os.path.exists(frame_output_dir) else None
    os.makedirs(frame_output_dir, exist_ok=True)

    start_s, end_s = 5, 12

    fig = plot_mel_spectrogram(audio_path, sr=16000, hop_length=160, n_mels=64, start_s=start_s, end_s=end_s)
    fig.savefig(output_path, dpi=400)

    frames = extract_video_frames(video_path, start_s=start_s, end_s=end_s)
    for i, frame in enumerate(frames):
        cv2.imwrite(f'{frame_output_dir}/{i:03d}.jpg', frame)


def add_video_attention(
    image, attention_map, output_path, 
    alpha=0.4, colormap=cv2.COLORMAP_JET
):
    H, W = image.shape[:2]
    attention_map_resized = cv2.resize(attention_map, (W, H), interpolation=cv2.INTER_CUBIC)

    # Normalize the attention map to [0, 1]
    attention_map_normalized = Normalize()(attention_map_resized)

    # Apply colormap to the attention map
    attention_colored = cv2.applyColorMap((attention_map_normalized * 255).astype(np.uint8), colormap)
    attention_colored = cv2.cvtColor(attention_colored, cv2.COLOR_BGR2RGB)

    # Blend the original image and attention map
    overlay = cv2.addWeighted(image[..., ::-1], 1 - alpha, attention_colored, alpha, 0)

    # Save and display the result
    plt.imsave(output_path, overlay)
    # plt.imshow(overlay)
    # plt.axis("off")
    # plt.show()

def add_audio_attention(
    mel_spectrogram, attention_map, output_path, 
    alpha=0.5, mel_colormap=cv2.COLORMAP_VIRIDIS, attn_colormap=cv2.COLORMAP_JET
):
    # cmap = cv2.COLORMAP_RAINBOW
    # Normalize Mel spectrogram to [0, 1] for colormap application
    mel_min, mel_max = -80, 0
    # mel_normalized = (mel_spectrogram - mel_min) / (mel_max - mel_min)
    mel_normalized = np.clip((mel_spectrogram - mel_min) / (mel_max - mel_min) * 255, 0, 255).astype(np.uint8)
    mel_colored = cv2.applyColorMap(mel_normalized, mel_colormap)[..., :3][..., ::-1]  # Apply colormap and discard alpha channel

    # Resize attention map to match Mel spectrogram size
    H, W = mel_spectrogram.shape
    att_norm = np.clip(attention_map / attention_map.max() * 255, 0, 255).astype(np.uint8)
    att_norm = cv2.resize(att_norm, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # Convert attention map to a heatmap (jet colormap)
    att_colormap = cv2.applyColorMap(att_norm, attn_colormap)
    
    # Blend the two images
    overlay = cv2.addWeighted(mel_colored, 1 - alpha, att_colormap, alpha, 0)
    
    # Save and display the result
    plt.imsave(output_path, overlay)
    # plt.imshow(overlay)
    # plt.axis("off")
    # plt.show()

def vis_attn(
    video_path='samples/visualize/sample_0000.mp4', 
    output_dir='debug/attn_vis', 
    sr=16000, hop_length=160, n_mels=64, fmin=0, fmax=8000, ref_db=-80,
):
    import torch

    shutil.rmtree(output_dir) if osp.exists(output_dir) else None
    os.makedirs(output_dir, exist_ok=True)

    audio_path = video_path.replace('.mp4', '.wav')
    y, sr = librosa.load(audio_path, sr=sr)
    # 计算 Mel 频谱
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    # 转换为对数幅度，并增强对比
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = np.clip(mel_spec_db, ref_db, 0)  # 限制动态范围
    # mel_spec_db = mel_spec_db.T
    # shape(t, n_mels), -80 ~ 0

    sample_rate = 4
    frames = extract_video_frames(video_path, sample_rate=1)
    # for i, frame in enumerate(frames):
    #     cv2.imwrite(f'{frame_output_dir}/{i:03d}.jpg', frame)
    ar = frames[0].shape[0] / frames[0].shape[1]  # h/w

    audio_t_stride = 4 * 4
    audio_m_stride = 4
    video_wh_stride = 8 * 2
    video_t_stride = 17 / 5  # 4

    ax_h = mel_spec_db.shape[0] // audio_m_stride
    ax_w = int(round(mel_spec_db.shape[1] / audio_t_stride))
    vx_h = math.ceil(frames[0].shape[0] / video_wh_stride)
    vx_w = math.ceil(frames[0].shape[1] / video_wh_stride)
    vx_t = math.ceil(len(frames) / video_t_stride)

    attn_dict = torch.load(video_path.replace('.mp4', '.pt'))
    for step, attn_map in attn_dict.items():
        if step != 'step29':
            continue
        for bi in tqdm(range(28), desc=step):
            v_s_attn_map = attn_map[f'block{bi}_video_spatial'].float()
            v_t_attn_map = attn_map[f'block{bi}_video_temporal'].float()
            v_s_attn_map = v_s_attn_map.view(vx_t, vx_h, vx_w)
            v_t_attn_map = v_t_attn_map.view(vx_h, vx_w, vx_t).permute(2, 0, 1)

            a_s_attn_map = attn_map[f'block{bi}_audio_spatial'].float()
            a_t_attn_map = attn_map[f'block{bi}_audio_temporal'].float()
            assert a_s_attn_map.shape == (ax_w, ax_h)
            a_s_attn_map = a_s_attn_map.transpose(0, 1)
            assert a_t_attn_map.shape == (ax_h, ax_w)
            
            if step == 'step29' and bi == 0:
                add_audio_attention(mel_spec_db, a_s_attn_map.cpu().numpy(), 
                                f'{output_dir}/audio_original.png', alpha=0.0)
            # add_audio_attention(mel_spec_db, a_s_attn_map.cpu().numpy(), 
            #                     f'{output_dir}/audio_{step}_block{bi}_audio_spatial.png')
            add_audio_attention(mel_spec_db, a_s_attn_map.cpu().numpy(),  # + a_s_attn_map.cpu().numpy()  a_t_attn_map
                                f'{output_dir}/audio_{step}_block{bi}_audio_temporal.png', alpha=1.0)

            for fi in range(0, len(frames), sample_rate):
                ai = int(fi / video_t_stride)
                if bi == 0:
                    add_video_attention(frames[fi], v_s_attn_map[ai].cpu().numpy(), 
                                    f'{output_dir}/frame{fi}_original.png', alpha=0.0)
                add_video_attention(frames[fi], v_s_attn_map[ai].cpu().numpy(), 
                                    f'{output_dir}/frame{fi}_{step}_block{bi}_video_spatial.png', alpha=1.0)
                # add_video_attention(frame, v_t_attn_map[ai].cpu().numpy(), 
                #                     f'{output_dir}/{step}_block{bi}_video_temporal.png')



def extract_t2av_demos():
    for path in [
        'samples/benchmark/JavisDiT-preliminary_240p4s/sample_0073.mp4',
        'samples/benchmark/JavisDiT-preliminary_240p4s/sample_0104.mp4',
        'samples/benchmark/JavisDiT-preliminary_240p4s/sample_3280.mp4',
        'samples/benchmark/JavisDiT-preliminary_240p4s/sample_3323.mp4',
        'samples/benchmark/JavisDiT-preliminary_240p4s/sample_3589.mp4',
        'samples/benchmark/JavisDiT-preliminary_240p4s/sample_7223.mp4',
        'samples/benchmark/JavisDiT-preliminary_240p4s/sample_8672.mp4',
        'samples/benchmark/JavisDiT-preliminary_240p4s/sample_10138.mp4',
        'samples/benchmark/JavisDiT-preliminary_240p4s/sample_8672.mp4',
    ]:
        frames = extract_video_frames(
            # 'samples/benchmark/JavisDiT-preliminary/sample_0067.mp4'
            # 'samples/benchmark/JavisDiT-preliminary/sample_0087.mp4'
            path
        )
        save_dir = f'debug/frames_demo3/{osp.basename(path)[:-4]}'
        os.makedirs(save_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            cv2.imwrite(f'{save_dir}/{i:03d}.jpg', frame)


def extract_xcond_demos():
    names = ['sample_0099', 'sample_0125']
    for name in names:
        for mode in ['a2v', 'ai2v', 'av_ext', 'i2av', 'v2a']:
            path = f'samples/x_cond/{mode}/{name}.mp4'
            frames = extract_video_frames(path)
            save_dir = f'debug/frames_demo_xcond/{mode}/{name}'
            os.makedirs(save_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                cv2.imwrite(f'{save_dir}/{i:03d}.jpg', frame)


if __name__ == '__main__':
    # extract_av()
    # video_path = 'samples/demo/v1_preliminary_720p2s/sample_0581.mp4'
    video_path = '../SyncVA-Bench/data/clip/youtube_clip__fnadg-fgw4_scene-0.mp4'
    save_dir = f'debug/{osp.splitext(osp.basename(video_path))[0]}'
    os.makedirs(save_dir, exist_ok=True)
    frames = extract_video_frames(video_path, sample_rate=6)
    for i, frame in enumerate(frames):
        cv2.imwrite(f'{save_dir}/{i:03d}.jpg', frame)

    # vis_attn()

    # extract_xcond_demos()
    pass
