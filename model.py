import numpy as np

class CNN:
    def __init__(self,kernel_size = (3,3), eps = 1e-9):
        self.kernel = np.zeros((kernel_size)) + eps
        
    
    def convolution(self, input, stride = 1):
        batch_size , H, W = input.shape
        kh, kw = self.kernel.shape

        out_h = (H - kh) // stride + 1
        out_w = (W - kw) // stride + 1

        outputs = np.zeros((batch_size, out_h, out_w))

        for b, img in enumerate(input):
            for j in range(0, out_h):
                for k in range(0, out_w):
                    region = img[j* stride : j * stride +kh,
                               k * stride : k * stride + kw]
                    
                    outputs[b,j,k] = np.sum(region * self.kernel)
        
        return outputs
        
    def activation(self, conv_outputs):
        return np.maximum(0, conv_outputs)
    



