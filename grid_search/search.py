import itertools
from model.mlp import MLP
from model.cross_entrophy_loss import SoftmaxCrossEntropyLoss
from training.Train import train
from training.data_separation import load_data


X_train, X_val, y_train, y_val, X_test, y_test = load_data()

lr_list = [0.4, 0.2, 1e-1, 1e-2]
wd_list = [1e-4, 1e-5, 0]

hidden_pairs = [
    (1024, 512),
    (1024, 256),
    (1024, 128),
    (512, 512),
    (512, 256),
    (512, 128),
    (256, 256),
    (256, 128),
    (128, 128)
]

results_allgrid = []



for lr, (h1, h2), wd in itertools.product(
        lr_list,
        hidden_pairs,
        wd_list):

    print(f"\n=== lr={lr}, h1={h1}, h2={h2}, wd={wd} ===")


    model = MLP(
        input_dim=784,
        hidden_dim1=h1,
        hidden_dim2=h2,
        output_dim=10,
        activation='relu'
    )

    loss_fn = SoftmaxCrossEntropyLoss()


    best_params, loss_hist, _, val_hist, _ = train(
        X_train, y_train,
        X_val, y_val,
        model,
        loss_fn,
        epochs=30,
        lr=lr,
        weight_decay=wd,
        batch_size=64,
        lr_schedule="constant"
    )

    best_val_acc = max(val_hist)


    results_allgrid.append({
        "lr": lr,
        "hidden_dim1": h1,
        "hidden_dim2": h2,
        "weight_decay": wd,
        "val_acc": best_val_acc
    })



best = max(results_allgrid, key=lambda x: x["val_acc"])

print("\n=========================")
print("Best config:")
print(best)
print("=========================")



results_sorted = sorted(results_allgrid, key=lambda x: x["val_acc"], reverse=True)

print("\nTop 10 configurations:")
for r in results_sorted[:10]:
    print(r)