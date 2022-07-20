import torch
import torch.nn as nn
import torch.nn.functional as F

def pixel_shuffle(input, upsacle_factor):
    '''
        input: (batchSize, c, w, h)
        downscale_factor: k
        (batchSize, c, w, h) -> (batchSize, c/k/k, k*w, k*h)
    '''
    c, w, h = input.shape[1], input.shape[2], input.shape[3]
    kernel = torch.zeros(size = [c, 1, upsacle_factor, upsacle_factor], device=input.device)
    for y in range(upsacle_factor):
        for x in range(upsacle_factor):
            kernel[x + y * upsacle_factor::upsacle_factor * upsacle_factor, 0, y, x] = 1
    # print('kernel:', kernel, kernel.shape)
    return F.conv_transpose2d(input, kernel, stride=upsacle_factor, groups=int(c / upsacle_factor / upsacle_factor))

class Pixel_Shuffle(nn.Module):
    def __init__(self, upsacle_factor):
        super(Pixel_Shuffle, self).__init__()
        self.upsacle_factor = upsacle_factor

    def forward(self, input):
        '''
        input: (batchSize, c, w, h)
        downscale_factor: k
        (batchSize, c, w, h) -> (batchSize, c/k/k, k*w, k*h)
        '''
        return pixel_shuffle(input, self.upsacle_factor)

def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    downscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]
    kernel = torch.zeros(size = [downscale_factor * downscale_factor * c, 1, downscale_factor, downscale_factor],
                        device = input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride = downscale_factor, groups = c)

class Pixel_UnShuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(Pixel_UnShuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        downscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''
        return pixel_unshuffle(input, self.downscale_factor)
