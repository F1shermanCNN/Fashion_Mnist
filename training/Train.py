from model.sgd import SGD
import numpy as np
from testing.Evaluate import evaluate
def train(X_train, y_train,
          X_val, y_val,
          model, loss_fn,
          epochs, lr, weight_decay,
          batch_size=64,
          lr_schedule="constant",
          patience=10):

    optimizer = SGD(model, lr=lr, weight_decay=weight_decay)

    N = X_train.shape[0]
    N_val = X_val.shape[0] 

    best_val_acc = 0
    best_params = None
    no_improve = 0

    train_loss_history = []
    val_loss_history = []  
    val_acc_history = []
    lr_history = []

    decay_points = [30, 60, 90, 120, 150, 180, 210]

    for epoch in range(epochs):
        # 学习率衰减策略
        if lr_schedule == "constant":
            current_lr = lr
        elif lr_schedule == "step":
            decay_count = sum(epoch >= dp for dp in decay_points)
            current_lr = lr * (0.1 ** decay_count)
        elif lr_schedule == "linear":
            current_lr = lr * (1 - epoch / epochs)
        elif lr_schedule == "inverse_sqrt":
            current_lr = lr / np.sqrt(epoch + 1)
        else:
            raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

        optimizer.set_lr(current_lr)
        lr_history.append(current_lr)

        indices = np.random.permutation(N)
        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0


        for i in range(0, N, batch_size):
            x_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            logits = model.forward(x_batch)
            loss = loss_fn.forward(logits, y_batch)
            epoch_loss += loss * x_batch.shape[0]

            grad = loss_fn.backward()
            model.backward(grad)
            optimizer.step()

        epoch_loss /= N
        train_loss_history.append(epoch_loss)


        epoch_val_loss = 0
        for i in range(0, N_val, batch_size):
            x_val_batch = X_val[i:i+batch_size]
            y_val_batch = y_val[i:i+batch_size]
            
            val_logits = model.forward(x_val_batch)
            v_loss = loss_fn.forward(val_logits, y_val_batch)
            epoch_val_loss += v_loss * x_val_batch.shape[0]

        epoch_val_loss /= N_val
        val_loss_history.append(epoch_val_loss)


        val_acc = evaluate(X_val, y_val, model)
        val_acc_history.append(val_acc)

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | " 
              f"Val Acc: {val_acc:.4f}")

        # 保存最优模型+Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {
                "fc1_W": model.fc1.W.copy(),
                "fc1_b": model.fc1.b.copy(),
                "fc2_W": model.fc2.W.copy(),
                "fc2_b": model.fc2.b.copy(),
                "fc3_W": model.fc3.W.copy(),
                "fc3_b": model.fc3.b.copy()
            }
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    print("\nBest Validation Accuracy:", best_val_acc)

    return best_params, train_loss_history, val_loss_history, val_acc_history, lr_history