import numpy as np
class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.labels = None

    def forward(self, logits, labels):
        """
        logits: (N, C) —— 模型输出
        labels: (N,)   —— 每个样本的类别

        return:
            loss: 标量
        """
        N = logits.shape[0]
        # 数值稳定处理
        altered_logits = logits - np.max(logits, axis=1, keepdims=True)
        # 计算 softmax
        expo_logits = np.exp(altered_logits)
        probs = expo_logits / (np.sum(expo_logits, axis=1, keepdims=True))
        # 计算 cross entropy loss (+1e-12避免数值问题)  -Σp_iln(q_i)，此处pi表示真实标签，q_i表示分到对应类的概率
        corect_logprobs = -np.log(probs[range(N), labels] + 1e-12)
        loss = np.sum(corect_logprobs) / N

        # 缓存中间变量
        self.probs = probs
        self.labels = labels
        return loss

    def backward(self):
        """
        return:
            grad_logits: (N, C) —— dL/dz
        """
        N = self.probs.shape[0]
        # dL/dz_k = probs_k - y_k 
        grad_logits = self.probs.copy()
        grad_logits[range(N), self.labels] -= 1
        # forward除以 N
        grad_logits /= N
        return grad_logits