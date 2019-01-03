import threading
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions

        self.achieve_hash = {}
        self.desire_hash = {}
        self.inter_hash = {}

        self.recent_history = deque(maxlen=100)
        
        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}

        self.buffers['o_2'] = self.buffers['o'][:, 1:, :]
        self.buffers['ag_2'] = self.buffers['ag'][:, 1:, :]
        self.buffers['dg_2'] = self.buffers['g'][:, 1:, :]


        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]
        buffers['dg_2'] = buffers['g'][:, 1:, :]


        # print(self.inter_hash)
        successful = np.amax(np.array(self.recent_history), axis = 1)
        success_rate = np.mean(successful)
            
        transitions = self.sample_transitions(buffers, batch_size, intersection_hash = self.inter_hash, success_rate=success_rate)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]
        achieve_key = 'ag_2'
        desire_key = 'dg_2'

        with self.lock:
            idxs = self._get_storage_idx(batch_size)
            if self.current_size >= self.size:
                old_achieve_points = self.buffers[achieve_key][idxs]
                old_desire_points = self.buffers[desire_key][idxs]
                # delete old episodes from hash tables

                for i in idxs:
                    for j in range(self.T):
                        ac_pos = self.buffers[achieve_key][i, j]
                        de_pos = self.buffers[desire_key][i, j]
                        ac_pos = ("%.2f,%.2f,%.2f" % (ac_pos[0], ac_pos[1], ac_pos[2]))
                        de_pos = ("%.2f,%.2f,%.2f" % (de_pos[0], de_pos[1], de_pos[2]))

                        if ac_pos in self.achieve_hash and self.achieve_hash[ac_pos] == (i, j):
                            del self.achieve_hash[ac_pos]
                            if ac_pos in self.inter_hash and self.inter_hash[ac_pos][0] == i and self.inter_hash[ac_pos][1] == j:
                                del self.inter_hash[ac_pos]
                        if de_pos in self.desire_hash and self.desire_hash[de_pos] == (i, j):
                            del self.desire_hash[de_pos]
                            if de_pos in self.inter_hash and self.inter_hash[de_pos][2] == i and self.inter_hash[de_pos][3] == j:
                                del self.inter_hash[de_pos]

            # load inputs into buffers
            for key in episode_batch.keys():
                self.buffers[key][idxs] = episode_batch[key]
            self.buffers['o_2'][idxs] = episode_batch['o'][:, 1:, :]
            self.buffers['ag_2'][idxs] = episode_batch['ag'][:, 1:, :]
            self.buffers['dg_2'][idxs] = episode_batch['g'][:, 1:, :]


            # add new episodes into hash tables                
            for i in idxs:
                self.recent_history.append(self.buffers['info_is_success'][i])
                for j in range(self.T):
                    ac_pos = self.buffers[achieve_key][i, j]
                    de_pos = self.buffers[desire_key][i, j]
                    ac_pos = ("%.2f,%.2f,%.2f" % (ac_pos[0], ac_pos[1], ac_pos[2]))
                    de_pos = ("%.2f,%.2f,%.2f" % (de_pos[0], de_pos[1], de_pos[2]))
                    self.achieve_hash[ac_pos] = (i, j)
                    self.desire_hash[de_pos] = (i, j)

            for i in idxs:
                for j in range(self.T):
                    ac_pos = self.buffers[achieve_key][i, j]
                    de_pos = self.buffers[desire_key][i, j]
                    ac_pos = ("%.2f,%.2f,%.2f" % (ac_pos[0], ac_pos[1], ac_pos[2]))
                    de_pos = ("%.2f,%.2f,%.2f" % (de_pos[0], de_pos[1], de_pos[2]))
                    if ac_pos in self.desire_hash and i != self.desire_hash[ac_pos][0]:
                        self.inter_hash[ac_pos] = (i, j, self.desire_hash[ac_pos][0], self.desire_hash[ac_pos][1])
                    if de_pos in self.achieve_hash and i != self.achieve_hash[de_pos][0]:
                        self.inter_hash[de_pos] = (self.achieve_hash[de_pos][0], self.achieve_hash[de_pos][1], i, j)

            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0
            self.achieve_hash = {}
            self.desire_hash = {}
            self.inter_hash = {}


    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size + inc)

        if inc == 1:
            idx = idx[0]
        return idx
