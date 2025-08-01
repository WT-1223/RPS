import random
import numpy as np
import collections
import pickle


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def save_replay_buffer(replay_buffer, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(replay_buffer.buffer, f)


def load_replay_buffer(file_path, buffersize):
    with open(file_path, 'rb') as f:
        buffer = pickle.load(f)
    replay_buffer = ReplayBuffer(buffersize)
    replay_buffer.buffer = collections.deque(buffer, maxlen=buffersize)
    return replay_buffer