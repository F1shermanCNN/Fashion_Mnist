import numpy as np
import matplotlib.pyplot as plt
from model.cross_entrophy_loss import SoftmaxCrossEntropyLoss
from model.mlp import MLP
from training.data_separation import load_data
from testing.test import test
from .Train import train
X_train, X_val, y_train, y_val, X_test, y_test = load_data()


## 超参数列表
param_list = [
    {'lr': 0.2, 'hidden_dim1': 512, 'hidden_dim2': 512, 'weight_decay': 1e-5},
    {'lr': 0.2, 'hidden_dim1': 512, 'hidden_dim2': 256, 'weight_decay': 1e-5},
    {'lr': 0.2, 'hidden_dim1': 512, 'hidden_dim2': 128, 'weight_decay': 1e-5},
    {'lr': 0.1, 'hidden_dim1': 512, 'hidden_dim2': 512, 'weight_decay': 1e-5},
    {'lr': 0.1, 'hidden_dim1': 512, 'hidden_dim2': 256, 'weight_decay': 1e-5},
    {'lr': 0.1, 'hidden_dim1': 512, 'hidden_dim2': 128, 'weight_decay': 1e-5}
]

epochs = 5
batch_size = 64
all_results = []


for i, params in enumerate(param_list, 1):

    print(f"\n训练第 {i} 组")
    print(params)

    model = MLP(
        input_dim=784,
        hidden_dim1=params['hidden_dim1'],
        hidden_dim2=params['hidden_dim2'],
        output_dim=10,
        activation='relu'
    )

    loss_fn = SoftmaxCrossEntropyLoss()


    best_params, train_loss_history, val_loss_history, val_acc_history, lr_history = train(
        X_train, y_train,
        X_val, y_val,
        model,
        loss_fn,
        epochs=epochs,
        lr=params['lr'],
        weight_decay=params['weight_decay'],
        batch_size=batch_size,
        lr_schedule="step",
        patience=40
    )

    test_acc = test(X_test, y_test, model, best_params)
    print("Test Accuracy:", test_acc)
    
    # 保存训练结果
    np.savez(
        f"best_params_{i}.npz",
        fc1_W=best_params["fc1_W"], fc1_b=best_params["fc1_b"],
        fc2_W=best_params["fc2_W"], fc2_b=best_params["fc2_b"],
        fc3_W=best_params["fc3_W"], fc3_b=best_params["fc3_b"]
    )


    np.savez(
        f"history_{i}.npz",
        train_loss=train_loss_history,
        val_loss=val_loss_history,   
        val_acc=val_acc_history,
        lr=lr_history
    )


    plt.figure(figsize=(12,5))


    plt.subplot(1,2,1)
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss", linestyle='--') 
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve (Exp {i})")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(val_acc_history, label="Val Accuracy", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Val Accuracy (Exp {i})")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"curve_{i}.png")
    plt.close()


    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(train_loss_history, label="Train Loss", color="blue")
    ax1.plot(val_loss_history, label="Val Loss", color="cyan", linestyle=":") 
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="blue")
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(lr_history, color="red", linestyle="--")
    ax2.set_ylabel("LR", color="red")

    plt.title(f"Loss vs LR (Exp {i})")
    plt.tight_layout()
    plt.savefig(f"lr_curve_{i}.png")
    plt.close()


    all_results.append({
        'id': i,
        'params': params,
        'test_acc': float(test_acc),
        'best_val_acc': float(max(val_acc_history)),
        'min_val_loss': float(min(val_loss_history)) 
    })


np.save("all_results.npy", all_results)


all_results_sorted = sorted(all_results, key=lambda x: x['test_acc'], reverse=True)
print("\n===== 最终排名 =====")
for r in all_results_sorted:
    print(r)