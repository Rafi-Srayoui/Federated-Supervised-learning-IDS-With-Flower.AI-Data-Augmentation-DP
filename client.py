# client.py
import sys, torch, flwr as fl
import torch.nn as nn
import torch.optim as optim
from model import Net, get_parameters, set_parameters
from data  import load_partition, BATCH_SIZE
from opacus import PrivacyEngine
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

#INPUT_DIM, NUM_CLASSES = global_shapes()
NUM_CLIENTS = 4              # keep in sync with run_fl_ids.bat / server.py
EPOCHS_PER_ROUND = 3         # small value → faster FL rounds

cid = int(sys.argv[1])       # e.g. python client.py 0
train_loader, val_loader = load_partition(cid, NUM_CLIENTS)
INPUT_DIM   = train_loader.dataset.tensors[0].shape[1]
NUM_CLASSES = 10 #len(torch.unique(train_loader.dataset.tensors[1]))
DEVICE      = torch.device("cpu")             # or "cuda" if available
print(f"[Client {cid}] features={INPUT_DIM}  classes={NUM_CLASSES}")

model      = Net(INPUT_DIM, NUM_CLASSES).to(DEVICE)
criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.parameters(), lr=1e-3)

# privacy_engine = PrivacyEngine()
# model, optimizer, train_loader = privacy_engine.make_private(
#     module=model,
#     optimizer=optimizer,
#     data_loader=train_loader,
#     noise_multiplier=1,   # Tune this later
#     max_grad_norm=1.0,
# )



def train():
    model.train()
    for _ in range(EPOCHS_PER_ROUND):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()


def test(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss   = criterion(logits, yb).item()
            _, pred = torch.max(logits, 1)
            correct += (pred == yb).sum().item()
            total   += yb.size(0)
            loss_sum += loss * xb.size(0)
    return loss_sum / total, correct / total


# helper to run the model on a DataLoader and collect y_true, y_pred
def collect_preds(loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = torch.argmax(model(xb), dim=1)
            y_true.extend( yb.cpu().numpy().tolist() )
            y_pred.extend(preds.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred)




class IDSClient(fl.client.NumPyClient):
    def get_parameters(self, config):                   # → List[np.ndarray]
        return get_parameters(model)

    def fit(self, params, config):
        set_parameters(model, params)
        train()
        # epsilon = privacy_engine.get_epsilon(delta=1e-5)
        # print(f"[Client {cid}] ε = {epsilon:.2f}, δ = 1e-5")
        # ---- local eval ----
        y_true, y_pred = collect_preds(val_loader)
        acc   = accuracy_score(y_true, y_pred)
        prec  = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec   = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1    = f1_score(y_true, y_pred, average="macro", zero_division=0)
        # per-class:
        per_class = {}
        for cls in range(NUM_CLASSES):
            # Build binary labels: 1 if this sample is class `cls`, else 0
            y_true_bin = (y_true == cls).astype(int)
            y_pred_bin = (y_pred == cls).astype(int)

            per_class[f"class_{cls}_prec"] = precision_score(
                y_true_bin, y_pred_bin, zero_division=0
            )
            per_class[f"class_{cls}_rec"]  = recall_score(
                y_true_bin, y_pred_bin, zero_division=0
            )
            per_class[f"class_{cls}_f1"]   = f1_score(
                y_true_bin, y_pred_bin, zero_division=0
            )

        print(f"[Client {cid}] acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")
        print("[Client per-class]", per_class)
        # you can also return them in the dict, e.g.:
        metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, **per_class}
        return get_parameters(model), len(train_loader.dataset), metrics

        # return get_parameters(model), len(train_loader.dataset), {}

    def evaluate(self, params, config):
        set_parameters(model, params)
        # Compute standard loss + full suite of metrics on val_loader
        # 1) loss
        loss_sum, total = 0.0, 0
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss_sum += criterion(logits, yb).item() * yb.size(0)
                total   += yb.size(0)
        loss = loss_sum / total

        # 2) predictions
        y_true, y_pred = collect_preds(val_loader)
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # 3) per-class
        per_class = {}
        for cls in range(NUM_CLASSES):
            # Build binary labels: 1 if this sample is class `cls`, else 0
            y_true_bin = (y_true == cls).astype(int)
            y_pred_bin = (y_pred == cls).astype(int)
        
            per_class[f"class_{cls}_prec"] = precision_score(
                y_true_bin, y_pred_bin, zero_division=0
            )
            per_class[f"class_{cls}_rec"]  = recall_score(
                y_true_bin, y_pred_bin, zero_division=0
            )
            per_class[f"class_{cls}_f1"]   = f1_score(
                y_true_bin, y_pred_bin, zero_division=0
            )

        metrics = {
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1_score":  f1,
            **per_class,
        }
        return loss, len(val_loader.dataset), metrics


if __name__ == "__main__":
    # NB: start_client is deprecated in 1.13+, but still functional. :contentReference[oaicite:0]{index=0}
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=IDSClient().to_client(),        # convert NumPyClient → Client
    )
