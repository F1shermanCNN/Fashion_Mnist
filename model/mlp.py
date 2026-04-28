from .linear_layer import Linear
from .activation import ReLU, Sigmoid, Tanh
class MLP:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, activation='relu'):

        self.fc1 = Linear(input_dim, hidden_dim1)
        self.fc2 = Linear(hidden_dim1, hidden_dim2)
        self.fc3 = Linear(hidden_dim2, output_dim)

        # 选择激活函数
        if activation == 'relu':
            self.act1 = ReLU()
            self.act2 = ReLU()
        elif activation == 'sigmoid':
            self.act1 = Sigmoid()
            self.act2 = Sigmoid()
        elif activation == 'tanh':
            self.act1 = Tanh()
            self.act2 = Tanh()
        else:
            raise ValueError("Unsupported activation")

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.act1.forward(x)

        x = self.fc2.forward(x)
        x = self.act2.forward(x)

        x = self.fc3.forward(x)
        return x

    def backward(self, grad):
        grad = self.fc3.backward(grad)

        grad = self.act2.backward(grad)
        grad = self.fc2.backward(grad)

        grad = self.act1.backward(grad)
        grad = self.fc1.backward(grad)

        return grad