import pickle
import os
from .io import check_path

class Recorder:
    def __init__(self, save_dir, max_buffer=2000):
        self.save_dir = check_path(save_dir)
        self.clear_buffer()
        self.count = 0
        self.max_buffer = max_buffer

    def record(self, action, obs, reward, done, info):
        self.buffer.append((action, obs, reward, done, info))
        if self.overflow():
            self.save()

    def save(self):
        file_name = os.path.join(self.save_dir, str(self.count) + '.pkl')
        with open(file_name, 'wb') as f:
            pickle.dump(self.buffer, f, protocol=2)
        self.count += 1
        self.clear_buffer()

    def clear_buffer(self):
        self.buffer = []

    def overflow(self):
        return len(self.buffer) >= self.max_buffer
