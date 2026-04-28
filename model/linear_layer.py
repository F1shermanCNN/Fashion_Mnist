import numpy as np
class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)  # He初始化
        self.b = np.zeros(out_features)  

    def forward(self, x):
        """
        前向传播：计算线性变换 z = Wx + b
        """
        self.x = x  
        self.z = np.dot(x, self.W) + self.b 
        return self.z

    def backward(self, grad_output):
        """
        反向传播：计算 W、b 和输入 x 的梯度
        """
        self.grad_W = np.dot(self.x.T, grad_output)  # W 的梯度
        self.grad_b = np.sum(grad_output, axis=0)    # b 的梯度
        self.grad_x = np.dot(grad_output, self.W.T)  # x 的梯度

        return self.grad_x