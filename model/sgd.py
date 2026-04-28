class SGD:
    def __init__(self, model, lr=0.01, weight_decay=0.0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for layer in [self.model.fc1, self.model.fc2, self.model.fc3]:

            layer.grad_W += self.weight_decay * layer.W

            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b
    def set_lr(self, new_lr):
        self.lr = new_lr