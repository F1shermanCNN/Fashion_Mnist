import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model.mlp import MLP
from training.data_separation import load_data

X_train, X_val, y_train, y_val, X_test, y_test = load_data()


## 加载最优参数的路径，保存在根目录即可
data_load = np.load("best_params_1.npz")

# 初始化模型结构 对应我的最优参数h1=512, h2=512；可根据不同读入数据修改
best_model = MLP(
    input_dim=784, 
    hidden_dim1=512, 
    hidden_dim2=512, 
    output_dim=10, 
    activation='relu'
)

# 将权重注入模型实例
best_model.fc1.W = data_load["fc1_W"]
best_model.fc1.b = data_load["fc1_b"]
best_model.fc2.W = data_load["fc2_W"]
best_model.fc2.b = data_load["fc2_b"]
best_model.fc3.W = data_load["fc3_W"]
best_model.fc3.b = data_load["fc3_b"]


logits = best_model.forward(X_test)
y_pred = np.argmax(logits, axis=1)


test_acc = np.mean(y_pred == y_test)
print(f"Test Set Performance")
print(f"Overall Accuracy: {test_acc:.4%}\n")

# 打印各分类的详细报告
fashion_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=fashion_labels))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=fashion_labels, 
    yticklabels=fashion_labels
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


print("Raw Confusion Matrix:")
print(cm)