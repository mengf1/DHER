import numpy as np
import random


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def search_memory(episode_batch, intersection_hash, success_flags):
        intersection = []
        for key, val in intersection_hash.items():
            ac_pos = key
            ac_i, ac_j, find_i, find_j = val
            if not success_flags[find_i]:
                intersection.append((ac_i, ac_j, find_i, find_j))
        return np.array(intersection)

    def _sample_her_transitions(episode_batch, batch_size_in_transitions, intersection_hash = {}, success_rate = 0.0):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """

        episode_batch['info_is_success'] = np.reshape(episode_batch['info_is_success'], episode_batch['info_is_success'].shape[:2])
        successful = np.amax(episode_batch['info_is_success'], axis = 1)
        
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions
        size2 = batch_size

        # Select which episodes and time steps to use
        intersection = np.array(search_memory(episode_batch, intersection_hash, successful))
        if len(intersection) > 0:
            joint_rate = len(set(intersection[:, 1])) / (len(successful) - np.sum(successful))
        else:
            joint_rate = 0.0
        
        inter_transitions = None
        # print(success_rate)
        if len(intersection) > 0:
            steps = np.minimum(intersection[:, 1], intersection[:, 3]) + 1.0
            steps_array = (np.random.uniform(size=len(intersection)) * steps).astype(int)
            achieve_idxs, desired_idxs = intersection[:, 0], intersection[:, 2]
            achieve_t, desired_t = intersection[:, 1] - steps_array, intersection[:, 3] - steps_array
        
            inter_transitions = {key: episode_batch[key][achieve_idxs, achieve_t].copy() for key in episode_batch.keys()}

            new_g = episode_batch ['g'][desired_idxs, desired_t]
            new_next_g = episode_batch['dg_2'][desired_idxs, desired_t]
            inter_transitions['g'] = new_g
            inter_transitions['dg_2'] = new_next_g

            if replay_strategy == 'future':
                future_p = 1 - (1. / (1 + replay_k))
            else:  # 'replay_strategy' == 'none'
                future_p = 0

            size1 = int(future_p * batch_size * (1.0 - success_rate))
            # size1 = int(future_p * batch_size)
            size2 = batch_size - size1
            idxs1 = np.random.randint(0, len(inter_transitions['u']), size1)
            inter_transitions = {key: inter_transitions[key][idxs1] for key in inter_transitions.keys()}
            # print(inter_transitions)

        idxs2 = np.random.randint(0, rollout_batch_size, size2)
        t_samples = np.random.randint(T, size=size2)
        transitions = {key: episode_batch[key][idxs2, t_samples].copy() for key in episode_batch.keys()}

        if inter_transitions:
            all_transitions = {key: np.concatenate([inter_transitions[key], transitions[key]])for key in episode_batch.keys()}
        else:
            all_transitions = transitions

        info = {}
        for key, value in all_transitions.items():
            if key.startswith('info_'):
                    info[key.replace('info_', '')] = value

        reward_params = {k: all_transitions[k] for k in ['ag_2', 'dg_2']}
        reward_params['info'] = info
        all_transitions['r'] = reward_fun(**reward_params)
        all_transitions = {k: all_transitions[k].reshape(batch_size, *all_transitions[k].shape[1:]) for k in all_transitions.keys()}

        assert(all_transitions['u'].shape[0] == batch_size_in_transitions)
        return all_transitions
    return _sample_her_transitions
