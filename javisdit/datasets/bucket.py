from collections import OrderedDict

import numpy as np

from javisdit.utils.misc import get_logger, is_main_process

from .aspect import ASPECT_RATIOS, get_closest_ratio


def find_approximate_hw(hw, hw_dict, approx=0.8):
    for k, v in hw_dict.items():
        if hw >= v * approx:
            return k
    return None


def find_closet_smaller_bucket(t, t_dict, frame_interval):
    # process image
    if t == 1:
        if 1 in t_dict:
            return 1
        else:
            return None
    # process video
    for k, v in t_dict.items():
        if t >= v * frame_interval and v != 1:
            return k
    return None


class Bucket:
    def __init__(self, bucket_config):
        for key in bucket_config:
            assert key in ASPECT_RATIOS, f"Aspect ratio {key} not found."
        # wrap config with OrderedDict
        bucket_probs = OrderedDict()
        bucket_bs = OrderedDict()
        # pixel num, descend
        bucket_names = sorted(bucket_config.keys(), key=lambda x: ASPECT_RATIOS[x][0], reverse=True)
        for key in bucket_names:
            # frame sizes
            bucket_time_names = sorted(bucket_config[key].keys(), key=lambda x: x, reverse=True)
            # { res_type: { frame_num: probs } } accept probabolity to take sample to current bucket
            bucket_probs[key] = OrderedDict({k: bucket_config[key][k][0] for k in bucket_time_names})
            # { res_type: { frame_num: bs } } pre-defined batch size for each type of video 
            bucket_bs[key] = OrderedDict({k: bucket_config[key][k][1] for k in bucket_time_names})

        # first level: HW
        num_bucket = 0
        hw_criteria = dict()
        t_criteria = dict()
        ar_criteria = dict()
        bucket_id = OrderedDict()
        bucket_id_cnt = 0
        for k1, v1 in bucket_probs.items():
            # k1: resolution type
            hw_criteria[k1] = ASPECT_RATIOS[k1][0]  # pixel num; resolution
            t_criteria[k1] = dict()
            ar_criteria[k1] = dict()
            bucket_id[k1] = dict()
            for k2, _ in v1.items():
                # k2: frame num; v2: probs
                t_criteria[k1][k2] = k2  # t_criteria[res_type][frame_num] = frame_num
                bucket_id[k1][k2] = bucket_id_cnt  # bucket_id[res_type][frame_num] = bucket_id_cnt
                bucket_id_cnt += 1
                ar_criteria[k1][k2] = dict()
                for k3, v3 in ASPECT_RATIOS[k1][1].items():
                    ar_criteria[k1][k2][k3] = v3  # ar_criteria[res_type][frame_num][ar] = [h, w]
                    num_bucket += 1  # splitting buckets according to spatial size?

        self.bucket_probs = bucket_probs
        self.bucket_bs = bucket_bs
        self.bucket_id = bucket_id
        self.hw_criteria = hw_criteria
        self.t_criteria = t_criteria
        self.ar_criteria = ar_criteria
        self.num_bucket = num_bucket
        get_logger().info("Number of buckets: %s", num_bucket)

    def get_bucket_id(self, T, H, W, frame_interval=1, seed=None):
        # randomly assigning raw videos to pre-defined and proper buckets
        resolution = H * W
        approx = 0.8

        fail = True
        for hw_id, t_criteria in self.bucket_probs.items():
            if resolution < self.hw_criteria[hw_id] * approx:
                continue

            # if sample is an image
            if T == 1:
                if 1 in t_criteria:
                    rng = np.random.default_rng(seed + self.bucket_id[hw_id][1])
                    if rng.random() < t_criteria[1]:
                        fail = False
                        t_id = 1
                        break
                else:
                    continue

            # otherwise, find suitable t_id for video
            t_fail = True
            for t_id, prob in t_criteria.items():
                rng = np.random.default_rng(seed + self.bucket_id[hw_id][t_id])
                if isinstance(prob, tuple):
                    prob_t = prob[1]  # accept prob 1
                    if rng.random() > prob_t:
                        continue
                if T > t_id * frame_interval and t_id != 1:
                    # ok when total frames more than pre-defined frame_num
                    t_fail = False
                    break
            if t_fail:
                continue

            # leave the loop if prob is high enough
            if isinstance(prob, tuple):
                prob = prob[0]  # accept prob 2
            if prob >= 1 or rng.random() < prob:
                fail = False
                break
        if fail:
            return None

        # get aspect ratio id
        ar_criteria = self.ar_criteria[hw_id][t_id]
        ar_id = get_closest_ratio(H, W, ar_criteria)
        return hw_id, t_id, ar_id

    def get_thw(self, bucket_id):
        assert len(bucket_id) == 3
        T = self.t_criteria[bucket_id[0]][bucket_id[1]]
        H, W = self.ar_criteria[bucket_id[0]][bucket_id[1]][bucket_id[2]]
        return T, H, W

    def get_prob(self, bucket_id):
        return self.bucket_probs[bucket_id[0]][bucket_id[1]]

    def get_batch_size(self, bucket_id):
        return self.bucket_bs[bucket_id[0]][bucket_id[1]]

    def __len__(self):
        return self.num_bucket


def closet_smaller_bucket(value, bucket):
    for i in range(1, len(bucket)):
        if value < bucket[i]:
            return bucket[i - 1]
    return bucket[-1]
