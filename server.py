# server.py
import torch, flwr as fl
from model import Net, set_parameters
from data  import load_global_test, BATCH_SIZE
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
NUM_CLIENTS = 4
ROUNDS      = 50

# ----- global evaluation loader -------------------------------------------
test_loader = load_global_test(BATCH_SIZE)
INPUT_DIM   = test_loader.dataset.tensors[0].shape[1]
NUM_CLASSES = len(torch.unique(test_loader.dataset.tensors[1]))
DEVICE      = torch.device("cpu")
print(f"[Server ] test_features={INPUT_DIM}  classes={NUM_CLASSES}")


def get_eval_fn():
    model = Net(INPUT_DIM, NUM_CLASSES).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()


    def evaluate(server_round, params, _config):
        # 1) load parameters & switch to eval
        set_parameters(model, params)
        model.eval()

        # 2) reset accumulators
        y_true_all = []
        y_pred_all = []
        loss_sum, total = 0.0, 0

        # 3) run through the global test set
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb).item()
                preds = torch.argmax(logits, dim=1)

                # accumulate loss & counts
                loss_sum += loss * yb.size(0)
                total   += yb.size(0)

                # accumulate for metrics
                y_true_all.extend( yb.cpu().numpy().tolist() )
                y_pred_all.extend(preds.cpu().numpy().tolist())

        # 4) compute overall metrics
        y_true = np.array(y_true_all)
        y_pred = np.array(y_pred_all)
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # 5) start your metrics dict *before* per-class
        metrics = {
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1_score":  f1,
        }

        # 6) now add per-class one-vs-rest metrics
        for cls in range(NUM_CLASSES):
            y_true_bin = (y_true == cls).astype(int)
            y_pred_bin = (y_pred == cls).astype(int)
            metrics[f"class_{cls}_prec"] = precision_score(
                y_true_bin, y_pred_bin, zero_division=0
            )
            metrics[f"class_{cls}_rec"]  = recall_score(
                y_true_bin, y_pred_bin, zero_division=0
            )
            metrics[f"class_{cls}_f1"]   = f1_score(
                y_true_bin, y_pred_bin, zero_division=0
            )

        # 7) optional: print the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"[Server] Confusion matrix:\n{cm}")

        # 8) return average loss and the full metrics dict
        return loss_sum / total, metrics

    return evaluate


strategy = fl.server.strategy.FedAvg(
    fraction_fit       = 1.0,
    fraction_evaluate  = 1.0,
    min_evaluate_clients    = NUM_CLIENTS,   # NEW: demand all 4 clients
    min_fit_clients    = NUM_CLIENTS,
    min_available_clients = NUM_CLIENTS,
    evaluate_fn            = get_eval_fn(),
)



if __name__ == "__main__":
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )
    import matplotlib.pyplot as plt
    
    # hist.metrics_centralized is a dict like {"accuracy": [(0, …), (1, …), …], "precision": […], …}
    metrics = history.metrics_centralized
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(8, 18))
    
    # 1) Global metrics
    ax = axes[0]
    for name in ["accuracy", "precision", "recall", "f1_score"]:
        if name in metrics:
            rounds, vals = zip(*metrics[name])
            ax.plot(rounds, vals, marker='o', label=name)
    ax.set_title("Global Metrics Over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    
    # 2) Per-class F1
    ax = axes[1]
    for cls in range(NUM_CLASSES):
        key = f"class_{cls}_f1"
        if key in metrics:
            rounds, vals = zip(*metrics[key])
            ax.plot(rounds, vals, marker='.', label=f"Class {cls}")
    ax.set_title("Per-Class F1 Over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("F1 Score")
    ax.legend(ncol=2)
    ax.grid(True)

        # 3) Per-class Recall
    ax = axes[2]
    for cls in range(NUM_CLASSES):
        key = f"class_{cls}_rec"
        if key in metrics:
            rounds, vals = zip(*metrics[key])
            ax.plot(rounds, vals, marker='.', label=f"Class {cls}")
    ax.set_title("Per-Class Recall Over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Recall")
    ax.legend(ncol=2, fontsize="small")
    ax.grid(True)

    # 4) Per-class Precision
    ax = axes[3]
    for cls in range(NUM_CLASSES):
        key = f"class_{cls}_prec"
        if key in metrics:
            rounds, vals = zip(*metrics[key])
            ax.plot(rounds, vals, marker='.', label=f"Class {cls}")
    ax.set_title("Per-Class Precision Over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Precision")
    ax.legend(ncol=2, fontsize="small")
    ax.grid(True)

    # 3) Class 3 Precision & Recall
    # ax = axes[2]
    # for prefix, style in [("prec", "s-"), ("rec", "d-")]:
    #     key = f"class_3_{prefix}"
    #     if key in metrics:
    #         rounds, vals = zip(*metrics[key])
    #         ax.plot(rounds, vals, style, label=prefix.capitalize())
    # ax.set_title("Class 3 Precision & Recall")
    # ax.set_xlabel("Round")
    # ax.set_ylabel("Value")
    # ax.legend()
    # ax.grid(True)
    
    plt.tight_layout()
    plt.show()   # one blocking call; stays open until you close it

