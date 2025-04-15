import os
from glob import glob
from tqdm import tqdm
from multiprocessing.pool import Pool
import argparse
import math

from funasr import AutoModel

model_path = './third_party/FunAudioLLM/SenseVoiceSmall'
# model_path = './third_party/funasr/fsmn-vad'


def eval_res(res_list):
    cnt = 0
    for i, res in enumerate(res_list):
        # print(i, res['value'])
        print(i, res['text'])
        cnt += int('<|nospeech|>' in res['text'])
    print(cnt, len(res_list))


def run_funasr(model_path, inputs, device='cuda:0'):
    model = AutoModel(model=model_path, device=device)  # , fp16=True
    return model.generate(input=inputs, batch_size=16)
    

def dp_funasr(model_path, inputs, gpu_num):
    pool = Pool(gpu_num)
    n = len(inputs)
    d = int(math.ceil(n / gpu_num))
    res = []
    for i in range(gpu_num):
        device_i = f'cuda:{i}'
        inputs_i = inputs[d*i:d*i+d]
        res.append(pool.apply_async(run_funasr, (model_path, inputs_i, device_i, )))
    pool.close()
    pool.join()

    ret = []
    for r in res:
        ret.extend(r.get())
    
    return ret


def filter_speech():
    from tools.datasets.datautil import read_data, apply

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, nargs="+", help="path to the input dataset")
    parser.add_argument("--ngpus", type=int, default=1, help="GPU number")
    args = parser.parse_args()

    data, input_name = read_data(args.input)
    save_path = os.path.dirname(args.input[0]) + '/' + input_name

    file_list = list(data['audio_path'])
    res = dp_funasr(model_path, file_list, args.ngpus)
    assert len(res) == len(file_list), f'{len(res)}, {len(file_list)}'

    speech_flag = [int('<|nospeech|>' not in r['text']) for r in res]
    data['speech'] = speech_flag
    data.to_csv(save_path+'_detspeech.csv', index=False)

    data = data[data['speech'] < 1]
    data.to_csv(save_path+'_nospeech.csv', index=False)

    print(f'{len(data)} samples left.')


def test_model():
    file_list = [
        # speech
        'data/raw/MMTrail-20M/data/video_dataset_1/-_EVfrXqkEo_0000018.wav',
        'data/raw/MMTrail-20M/data/video_dataset_1/-YVLAooFpXU_0000302.wav',
        'data/raw/MMTrail-20M/data/video_dataset_1/-YKn-grvBYY_0000140.wav',
        'data/raw/MMTrail-20M/data/video_dataset_1/-UlGiV2WHG0_0000050.wav',
        'data/raw/MMTrail-20M/data/video_dataset_1/-S5ZDRwXnd4_0000013.wav',
        # non speech
        'data/raw/MMTrail-20M/data/video_dataset_1/-_4wD4B0cwc_0000003.wav',
        'data/raw/MMTrail-20M/data/video_dataset_1/--mVgqxHZzw_0000023.wav',
        'data/raw/MMTrail-20M/data/video_dataset_1/--tnMF5bX_o_0000049.wav',
        'data/raw/MMTrail-20M/data/video_dataset_1/-DfrgDzde9w_0000694.wav',
        'data/raw/MMTrail-20M/data/video_dataset_1/-NmbFRdmmqQ_0000000.wav'
    ]

    # data_root = 'data/raw/MMTrail-20M/data'  # 2284/7811 without speech
    # file_list = list(glob(f'{data_root}/*/*.wav'))

    res = run_funasr(model_path, file_list, 'cuda:0')
    # res = dp_funasr(model_path, file_list, 8)
    eval_res(res)


if __name__ == '__main__':
    # test_model()
    filter_speech()