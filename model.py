import numpy as np

class CNN:
    def __init__(self, input_data , kernel_size=(3,3), eps=1e-9, num_classes=4):
        self.input_data = input_data
        self.kernel = np.random.randn(*kernel_size) * eps   
        self.weight = None
        self.bias = None
        self.last_flatten = None

    def convolution(self, input, stride = 1):
        input = np.pad(input, ((0,0),(1,1),(1,1)), mode = 'constant', constant_values = 0)
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
        self.last_flatten = flatten

        if self.weight is None:
            in_dim = flatten.shape[1]
            self.weight = np.random.randn(in_dim , 4)
            self.bias = np.zeros(4)
        logits = np.dot(flatten, self.weight) + self.bias


        exp = np.exp(logits - np.max(logits, axis = 1, keepdims = True))
        softmax = exp / np.sum(exp , axis = 1, keepdims = True)
        return softmax

    def forward(self ,stride_k = 1, p_size = 2, stride_p = 2):
        conv_out = self.convolution(self.input_data, stride_k)
        act_out = self.activation(conv_out)
        conv2_out = self.convolution(act_out , stride_k)
        act2_out = self.activation(conv2_out)
        conv3_out = self.convolution(act2_out, stride_k)
        act3_out = self.activation(conv3_out)
        pool_out = self.max_pooling(act3_out, p_size , stride_p)
        f_result = self.fullyconnected_layer(pool_out)
        y_pred = np.argmax(f_result, axis = 1)
        return f_result, y_pred

    def cross_entropy(self, y_pred, y_true, eps=1e-9):
        # y_true가 정수 라벨이면 one-hot으로 변환
        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true.astype(int)]

        loss = -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))
        return y_true, loss

    
    def backward(self, x, y_true, y_pred):
        batch_size = x.shape[0]
        dz = (y_pred - y_true) / batch_size
        dw = np.dot(x.T, dz)
        db = np.sum(dz, axis = 0)
        return dw, db
    
    def gd(self, dw, db, lr = 0.1):
        self.weight -= lr * dw
        self.bias -= lr * db

    def conv_backward(self, x, y_true , y_pred):
        

