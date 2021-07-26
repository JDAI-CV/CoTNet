import numpy as np
import torch
from torch import nn as nn

class Shiftlution(nn.Module):
    def __init__(self, channels, kernel_h, kernel_w, max_height = 400, max_width = 400):
        super(Shiftlution, self).__init__()
        assert kernel_w % 2 == 1 and kernel_h % 2 == 1
        self.pad_w = (kernel_w - 1) // 2
        self.pad_h = (kernel_h - 1) // 2

        ########################## get shift list ##########################
        sort_value = []
        shift_list = []
        shift_h = list(range(-(kernel_h-1)//2,(kernel_h-1)//2+1, 1)) if kernel_h != 1 else [0]
        shift_w = list(range(-(kernel_w-1)//2,(kernel_w-1)//2+1, 1)) if kernel_w != 1 else [0]
        for h in shift_h:
            for w in shift_w:
                shift_list.append((h, w))
                sort_value.append(max(abs(h) + abs(h)/10.0 + abs(w)/100.0 + h/1000.0 + w/10000.0, 
                                      abs(w) + abs(h)/20.0 + abs(w)/200.0 + h/2000.0 + w/20000.0))

        sort_idx = np.argsort(sort_value)
        shift_list = np.array(shift_list)
        shift_list = shift_list[sort_idx]

        ########################## set shift array ##########################
        index_w = np.zeros((channels, max_width), dtype=int)
        index_h = np.zeros((channels, max_height), dtype=int)
        range_w = np.array(range(max_width))
        range_h = np.array(range(max_height))
        c_span = channels // (kernel_w * kernel_h)
        for i, shift in enumerate(shift_list):
            h_offset = shift[0]
            w_offset = shift[1]

            index_w[i*c_span:(i+1)*c_span, :] = range_w + w_offset + self.pad_w
            index_h[i*c_span:(i+1)*c_span, :] = range_h + h_offset + self.pad_h

        index_w[len(shift_list)*c_span:, :] = range_w + self.pad_w
        index_h[len(shift_list)*c_span:, :] = range_h + self.pad_h

        self.register_buffer('index_w', torch.from_numpy(index_w))
        self.register_buffer('index_h', torch.from_numpy(index_h))
        self.shift_list = shift_list

        #self.jit_shift = JITShiftlution(channels, kernel_h, kernel_w)
        
    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, H*W)
        y = torch.zeros(B, C, (H + 2*self.pad_h) * (W + 2*self.pad_w), dtype=x.dtype, device=x.device)
        index_h = self.index_h[:, 0:H]
        index_w = self.index_w[:, 0:W]
        index = index_h.unsqueeze(-1) * (W + 2*self.pad_w) + index_w.unsqueeze(1)
        index = index.unsqueeze(0).expand(B, C, H, W).view(B, C, H*W)
        y = y.scatter(2, index, x)
        y = y.view(B, C, H + 2*self.pad_h, W + 2*self.pad_w)
        y = y[:, : , self.pad_h:self.pad_h + H, self.pad_w:self.pad_w + W]
        #y = self.jit_shift(x, self.index_h, self.index_w)
        return y