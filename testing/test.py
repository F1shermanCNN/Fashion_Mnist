from testing.Evaluate import evaluate
def test(X_test, y_test, model, best_params):

    model.fc1.W = best_params["fc1_W"]
    model.fc1.b = best_params["fc1_b"]

    model.fc2.W = best_params["fc2_W"]
    model.fc2.b = best_params["fc2_b"]

    model.fc3.W = best_params["fc3_W"]
    model.fc3.b = best_params["fc3_b"]


    acc = evaluate(X_test, y_test, model)
    print("Test Accuracy:", acc)

    return acc