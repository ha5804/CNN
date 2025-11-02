import numpy as np

class CNN:
    def __init__(self, input_data , kernel_size = (3,3), eps = 1e-9, num_classes = 4):
        self.kernel = np.random.randn((kernel_size)) + eps
        self.weight = np.random.randn(((input_data.reshape(input_data.shape[0], -1)).shape[1], num_classes))
        self.bias = np.zeros(num_classes)
        
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
    
    def max_pooling(self, input , pooling_size = 2, stride = 2):
        #input = output of convolution
        #stride is general same pooling_size
        batch_size , h, w = input.shape
        out_h = (h - pooling_size) // stride + 1
        out_w = (w - pooling_size) // stride + 1

        outputs = np.zeros((batch_size, out_h, out_w))
        for b in range(batch_size):
            for i in range(0, out_h):
                for j in range(0, out_w):
                    region = input[b, i * stride : i * stride + pooling_size, j * stride : j * stride + pooling_size]
                    outputs[b,i,j] = np.max(region)
        return outputs

    def fullyconnected_layer(self, input):
        flatten = input.reshape(input.shape[0], -1)
        logits = np.dot(flatten, self.weight)

        exp = np.exp(logits - np.max(logits, axis = 1, keepdims = True))
        softmax = exp / np.sum(exp , axis = 1, keepdims = True)
        return softmax

    def forword(self):

    



