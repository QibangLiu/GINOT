# %%
import torch

# %%


class SlidingWindow():
    def __init__(self, window_size, overlap=0):
        self.window_size = window_size
        self.stride = window_size - overlap

    def set_window(self, window_size, overlap=0):
        self.window_size = window_size
        self.stride = window_size - overlap

    def __call__(self, sequence, fun: callable, *args, **kwargs):
        sequence_length = sequence.size(1)
        outputs = []
        for start in range(0, sequence_length, self.stride):
            end = min(start + self.window_size, sequence_length)
            window = sequence[:, start:end]
            output = fun(window, *args, **kwargs)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        return outputs
