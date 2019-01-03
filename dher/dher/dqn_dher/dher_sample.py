import numpy as np
import random


def make_sample_her_experience(episode_batch, batch_size):
    """Creates a sample function that can be used for HER experience replay.
    """

    def hash_memory(data):
        achieved_hash = {}
        desired_hash = {}
        for idx in range(len(data)):
            ac_x = data[idx][0][0]
            ac_y = data[idx][0][1]
            de_x = data[idx][0][2]
            de_y = data[idx][0][3]
            achieved_hash[(ac_x, ac_y)] = idx
            desired_hash[(de_x, de_y)] = idx
        return achieved_hash, desired_hash

    def search_memory(data):
        intersection_set = []
        achieved_hash, desired_hash = hash_memory(data)
        for key, val in achieved_hash.items():
            temp_x, temp_y = key
            ac_idx = val
            if (temp_x, temp_y) in desired_hash:
                sel = desired_hash[(temp_x, temp_y)]
                de_idx = sel
                intersection_set.append((ac_idx, de_idx))
        return intersection_set

    intersection = search_memory(episode_batch)
    if len(intersection) == 0:
        return []

    future_t = 8
    sample_batch = []
    data = episode_batch
    for i in range(batch_size):
        rand_i = random.randint(0, len(intersection) - 1)
        ac_idx, de_idx = intersection[rand_i]
        future_t = 8
        new_ft = 0
        for j in range(future_t):
            ac_done = data[ac_idx - j][4]
            de_done = data[de_idx - j][4]
            if ac_done == 1 or de_done == 1:
                break
            new_ft += 1
        new_ft = new_ft - 2
        de_idx = de_idx - new_ft
        ac_idx = ac_idx - new_ft
        new_exp = []
        for j in range(new_ft):
            ac_x = data[ac_idx][0][0]
            ac_y = data[ac_idx][0][1]
            de_x = data[de_idx][0][2]
            de_y = data[de_idx][0][3]
            ac_x2 = data[ac_idx + 1][0][0]
            ac_y2 = data[ac_idx + 1][0][1]
            de_x2 = data[de_idx + 1][0][2]
            de_y2 = data[de_idx + 1][0][3]
            ne_obs = [ac_x, ac_y, de_x, de_y, ac_x - de_x, ac_y - de_y]
            ne_act = data[ac_idx][1]
            ne_reward = -1.0
            ne_done = 0
            if j == new_ft - 1:
                ne_reward = 0.0
                ne_done = 1
            ne_obs2 = [
                ac_x2, ac_y2, de_x2, de_y2, ac_x2 - de_x2, ac_y2 - de_y2
            ]
            new_exp.append((np.asarray(ne_obs), ne_act, ne_reward,
                            np.asarray(ne_obs2), ne_done))
            ac_idx += 1
            de_idx += 1
        sample_batch.append(new_exp)

    return sample_batch