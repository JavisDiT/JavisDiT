import argparse
import os
import os.path as osp
import pandas as pd
import json
from typing import List, Dict, Literal
from glob import glob

import logging
logging.warning = lambda *args, **kwargs: None

import torch

from .src.metrics import (
    calc_fvd_kvd_fad, calc_imagebind_score, 
    calc_clip_score, calc_clap_score, calc_cavp_score,
    calc_av_align,  calc_av_score,
    calc_audio_score
)

class JavisBenchCategory(object):
    def __init__(self, cfg: str):
        self.cfg = cfg
        
        with open(cfg, 'r') as f:
            data = json.load(f)
        
        category_matrix = []
        for aspect in data:
            category_list = []
            for category in aspect['categories']:
                category_list.append(category['title'])
            category_matrix.append(category_list)

        self.category_cfg = data
        self.aspect_list = [aspect['aspect'] for aspect in data]
        self.category_matrix = category_matrix
        self.aspect_num = len(self.category_matrix)


class JavisEvaluator(object):
    def __init__(self, input_file: str, category_cfg: str, metrics: List[str], output_file: str, **kwargs):
        self.input_file = input_file
        self.df = pd.read_csv(input_file) #.loc[:100]

        if category_cfg and osp.isfile(category_cfg) and kwargs.get('verbose'):
            self.parse_aspect_dict(category_cfg)
        else:
            self.cat2indices = None

        self.output_file = output_file

        self.total_metrics = [
            'fvd+kvd+fad',   # quality
            'imagebind-score', 'cxxp-score',  # semantic consistency
            'av-align',  # av alignment
            'av-score',  #'avh-score', 'javis-score'
            # 'audio-score', 
        ]
        if metrics == ['all']:
            metrics = self.total_metrics
        self.metrics = metrics
        self.metric2items = {
            'fvd+kvd+fad': ['fvd', 'kvd', 'fad'],
            'imagebind-score': ['ib_tv', 'ib_ta', 'ib_av'],
            'cxxp-score': ['clip_score', 'clap_score', 'cavp_score'],
            'av-align': ['av_align'],
            'av-score': ['avh_score', 'javis_score'],
            'audio-score': ['fad', 'ib_ta', 'clap'],
        }

        self.eval_kwargs = kwargs

        self.gather_audio_video_data()
    
    def parse_aspect_dict(self, category_cfg:str):
        self.category = JavisBenchCategory(category_cfg)
        cat2indices: List[List[List[int]]] = []
        for ai in range(self.category.aspect_num):
            index_list = [[] for _ in range(len(self.category.category_matrix[ai]))]
            for pi, cat_str in enumerate(self.df[f'cat{ai}_ind'].tolist()):
                for ci in (cat_str.split(',') if isinstance(cat_str, str) else [cat_str]):
                    index_list[int(ci)].append(pi)
            cat2indices.append(index_list)
        
        self.cat2indices = cat2indices
    
    @torch.no_grad()
    def __call__(self, *args, **kwds):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        exist_metrics = self.load_metric()

        prompt_list = self.df['text'].tolist()
        # video_prompt_list = self.df.get('video_text', self.df['text']).tolist()
        video_prompt_list = prompt_list
        audio_prompt_list = self.df.get('audio_text', self.df['text']).tolist()

        gt_video_list = self.df['path'].tolist()
        gt_audio_list = self.df.get('audio_path', self.df['path']).tolist()

        pred_video_list = self.df['pred_video_path']
        pred_audio_list = self.df['pred_audio_path']
        
        for metric in self.metrics:
            if all(item in exist_metrics for item in self.metric2items[metric]):
                print(f'{metric} calculated. skip.')
                continue
            
            if metric == 'fvd+kvd+fad':
                mode = self.eval_kwargs.get('fvd_mode', 'vanilla')
                eval_num = self.eval_kwargs.get('fvd_eval_num', None)
                exist_metrics["fvd"], exist_metrics["kvd"], exist_metrics["fad"] = \
                    calc_fvd_kvd_fad(gt_video_list, pred_video_list, gt_audio_list, pred_audio_list, 
                                     device, self.cat2indices, eval_num, mode=mode, **self.eval_kwargs)
                self.write_metric(exist_metrics, metric)
            
            elif metric == 'imagebind-score':
                exist_metrics["ib_tv"], exist_metrics["ib_ta"], exist_metrics["ib_av"] = \
                    calc_imagebind_score(pred_video_list, pred_audio_list, video_prompt_list, audio_prompt_list,
                                              device, self.cat2indices)
                self.write_metric(exist_metrics, metric)
            
            elif metric == 'cxxp-score':
                if "clip_score" not in exist_metrics:
                    exist_metrics["clip_score"] = calc_clip_score(pred_video_list, video_prompt_list, device, self.cat2indices)
                if "clap_score" not in exist_metrics:
                    exist_metrics["clap_score"] = calc_clap_score(pred_audio_list, audio_prompt_list, device, self.cat2indices)
                if "cavp_score" not in exist_metrics:
                    exist_metrics["cavp_score"] = calc_cavp_score(pred_video_list, pred_audio_list, device, self.cat2indices,
                                                                  cavp_config_path=self.eval_kwargs['cavp_config_path'])
                self.write_metric(exist_metrics, metric)
            
            elif metric == 'av-align':
                if "av_align" not in exist_metrics:
                    exist_metrics["av_align"], av_align_scores = \
                        calc_av_align(pred_video_list, pred_audio_list, self.cat2indices, return_score_list=True)
                else:
                    av_align_scores = self.df.get('av_align_scores', [None] * len(self.df))
                self.write_metric(exist_metrics, metric)
                if self.eval_kwargs.get('save_avalign_scores', False):
                    assert len(av_align_scores) == len(self.df)
                    self.df['av_align_scores'] = av_align_scores

            elif metric == 'av-score':
                if any(flag not in exist_metrics for flag in ["avh_score", "javis_score"]):
                    exist_metrics["avh_score"], exist_metrics["javis_score"], avh_scores, javis_scores = \
                        calc_av_score(pred_video_list, pred_audio_list, prompt_list, device, self.cat2indices,
                                        window_size_s=self.eval_kwargs.get("window_size_s", 2.0),
                                        window_overlap_s=self.eval_kwargs.get("window_overlap_s", 1.5),
                                        return_score_list=True)
                else:
                    avh_scores = self.df.get('avh_scores', [None] * len(self.df))
                    javis_scores = self.df.get('javis_scores', [None] * len(self.df))
                if self.eval_kwargs.get('save_avalign_scores', False):
                    assert len(avh_scores) == len(javis_scores) == len(self.df)
                    self.df['avh_scores'] = avh_scores
                    self.df['javis_scores'] = javis_scores
                    save_path = osp.splitext(self.output_file)[0] + '_avalign.csv'
                    self.df.to_csv(save_path, index=False)
                self.write_metric(exist_metrics, metric)
            
            elif metric == 'audio-score':
                audio_prompt_list = self.df['audio_text'].tolist()
                exist_metrics["fad"], exist_metrics['ib_ta'], exist_metrics['clap'] = \
                    calc_audio_score(gt_audio_list, pred_audio_list, audio_prompt_list, 
                                     device, **self.eval_kwargs)
                self.write_metric(exist_metrics, metric)

    def write_metric(self, metric:dict, metric_type:str):
        os.makedirs(osp.dirname(self.output_file), exist_ok=True)
        for item in self.metric2items[metric_type]:
            score = metric[item]
            if isinstance(score, dict):
                score = score['overall']
            print(f'{item}: {score:.4f}', end='; ')
        print()
        with open(self.output_file, 'w+') as f:
            json.dump(metric, f, indent=4, ensure_ascii=False)

    def load_metric(self):
        metric = {}
        if osp.exists(self.output_file) and osp.getsize(self.output_file) > 0:
            with open(self.output_file, 'r') as f:
                metric = json.load(f)
        return metric

    def gather_audio_video_data(self):
        infer_data_dir = self.eval_kwargs['infer_data_dir']
        if not infer_data_dir:
            return
        assert osp.isdir(infer_data_dir), infer_data_dir
        audio_only = self.metrics == ['audio-score']
        sample_num = len(self.df)
        if audio_only:
            pred_audio_list = sorted(glob(f'{infer_data_dir}/*.wav'))
            pred_video_list = [''] * sample_num
            assert len(pred_audio_list) == sample_num
            self.df['text'] = self.df['audio_text']
            self.df['path'] = self.df['audio_path']
        else:
            pred_audio_list = sorted(glob(f'{infer_data_dir}/*.wav'))
            pred_video_list = sorted(glob(f'{infer_data_dir}/*.mp4'))
            assert len(pred_audio_list) == sample_num
            assert len(pred_video_list) == sample_num

        self.df['pred_video_path'] = pred_video_list
        self.df['pred_audio_path'] = pred_audio_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None, help="path to input csv file", required=True)
    parser.add_argument("--infer_data_dir", type=str, default=None, help="directory to audio-video inference results")
    parser.add_argument("--output_file", type=str, default=None, help="path to output json file", required=True)
    parser.add_argument("--category_cfg", type=str, default='./eval/javisbench/configs/category.json')
    parser.add_argument("--metrics", type=str, nargs='+', default='all', help="metrics to calculate, default as `all`")
    parser.add_argument("--verbose", action='store_true', default=False, help="whether to present category-specific score list")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for data loading")
    # parameters for evaluation
    parser.add_argument("--max_frames", type=int, default=24, help="size of the input video")
    parser.add_argument("--max_audio_len_s", type=float, default=None, help="maximum length of the audio")
    parser.add_argument("--video_fps", type=int, default=24, help="frame rate of the input video")
    parser.add_argument("--audio_sr", type=int, default=16000, help="sampling rate of the audio")
    parser.add_argument("--image_size", type=int, default=224, help="size of the input image")
    parser.add_argument("--fvd_eval_num", type=int, default=None, help="number of videos to evaluate")
    parser.add_argument("--fvd_avcache_path", type=str, default=None, help="path to the audio-video cache file for FVD/KVD/FAD evaluation")
    parser.add_argument("--fvd_mode", type=str, default='vanilla', choices=['vanilla', 'mmdiffusion'], help="mode of fvd calculation, `video` or `audio`")
    # hyper-parameters for metrics
    parser.add_argument("--window_size_s", type=float, default=2.0, help="JavisScore window size")
    parser.add_argument("--window_overlap_s", type=float, default=1.5, help="JavisScore overlap size")
    parser.add_argument("--cavp_config_path", type=str, default='./eval/javisbench/configs/Stage1_CAVP.yaml', help="JavisScore overlap size")
    parser.add_argument("--save_avalign_scores", action='store_true', default=False, help="whether to return score list for AV-Align evaluation")
    args = parser.parse_args()

    cache_dir = f'{osp.dirname(args.output_file)}/cache/{osp.basename(osp.splitext(args.output_file)[0])}'
    setattr(args, "cache_dir", cache_dir)

    evaluator = JavisEvaluator(**vars(args))
    evaluator()
