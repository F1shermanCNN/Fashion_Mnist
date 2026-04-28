import numpy as np
class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        """
        x: 输入 (N, D)
        
        return:
            out: 激活后的输出
        """
        # 记录<=0的位置
        self.mask = (x <= 0)
        # 计算 ReLU
        out = x.copy()

        out[self.mask] = 0
        
        return out

    def backward(self, grad_output):
        """
        grad_output: 上一层传回来的梯度 (dL/da)
        
        return:
            grad_input: 传给前一层的梯度 (dL/dx)
        """
        # 链式法则：grad_output * 导数
        # ReLu的导数在x>0处为1，其余为0
        grad_input = grad_output.copy()
        grad_input[self.mask] = 0
        
        return grad_input
    
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        """
        x: 输入 (N, D)
        
        return:
            out: 激活后的输出
        """
        out = 1 / (1 + np.exp(-x))
        # 保留结果，避免重复计算
        self.out = out
        return out

    def backward(self, grad_output):
        """
        grad_output: 上一层传回来的梯度 (dL/da)
        
        return:
            grad_input: 传给前一层的梯度 (dL/dx)
        """
        # 链式法则：grad_output * 导数
        # Sigmoid的导数为f(x)*(1-f(x))
        grad_input = grad_output.copy()
        grad_input *= self.out * (1 - self.out)
        
        return grad_input
    
class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, x):
        """
        x: 输入 (N, D)
        
        return:
            out: 激活后的输出
        """
        out = np.tanh(x)
        # 保留结果，避免重复计算
        self.out = out
        return out

    def backward(self, grad_output):
        """
        grad_output: 上一层传回来的梯度 (dL/da)
        
        return:
            grad_input: 传给前一层的梯度 (dL/dx)
        """
        # 链式法则：grad_output * 导数
        # Tanh的导数为1-tanh(x)^2
        grad_input = grad_output.copy()
        grad_input *= (1 - self.out ** 2)
        
        return grad_input