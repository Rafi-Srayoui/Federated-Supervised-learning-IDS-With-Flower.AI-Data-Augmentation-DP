# FL-IDS: Federated Learning Intrusion Detection System (UNSW-NB15)

This project implements a **Federated Learning (FL)** system for **Intrusion Detection** using the [Flower](https://flower.dev/) framework and the **UNSW-NB15** dataset. It supports **non-IID label distributions**, centralized evaluation with **per-class metrics**, and optional visualization tools for data skew analysis.

---

## 📁 Project Structure

FL-IDS - non-iid - Full Metrics/

├── dataset/ # Directory containing UNSW-NB15 CSVs

├── client.py # Flower client logic (training, validation, local metrics)

├── data.py # Data preprocessing and Dirichlet skewed partitioning

├── model.py # Neural network model and FL utility functions

├── server.py # Flower server with full evaluation metrics + plots

├── run_fl_ids.bat # Script to launch server and multiple clients


---

## 🚀 How It Works

- **Dataset**: Uses the **UNSW-NB15** intrusion detection dataset (numeric features only).
- **Model**: A simple **Multilayer Perceptron (MLP)** classifier.
- **Partitioning**: Clients receive data via **Dirichlet non-IID splits** (controlled via α).
- **Evaluation**:
  - Global test set is held out centrally.
  - Both **global** and **per-class** metrics (accuracy, precision, recall, F1) are computed.
  - Confusion matrix and plots are generated on the server side.
- **Training**:
  - Local training for 3 epochs per round.
  - Run for 50 rounds.
  - Default: 4 clients (modifiable).

---

## 📊 Performance Summary (FedAvg)

Using **non-IID data with α = 0.1**, the system achieved:

| Metric                | Value     |
|-----------------------|-----------|
| Accuracy              | 81.9%     |
| Macro F1 Score        | 0.371     |
| Macro Precision       | 0.735     |
| Macro Recall          | 0.381     |
| Best-class F1 (Normal)| 1.000     |
| Weakest-class F1      | 0.002–0.04|

> See `Results - FedAVG.docx` for full run details (alpha sweeps, per-class trends, etc.)

---

## ⚙️ Setup & Running

### 1. Install Dependencies

```bash
pip install flwr torch scikit-learn pandas matplotlib opacus
Optional: opacus for privacy support (commented in code)

3. Run the System
Use the provided .bat file (Windows) or adapt the commands manually:


# In terminal 1
python server.py

# In terminals 2–5 (one per client)
python client.py 0
python client.py 1
python client.py 2
python client.py 3
You can change the number of clients by adjusting:

NUM_CLIENTS in server.py, client.py

run_fl_ids.bat to launch more clients

📌 Highlights
✅ Full per-class precision, recall, and F1 scores

✅ Handles non-IID client data (Dirichlet α skew)

✅ Centralized confusion matrix and plots

✅ Easy to customize and extend (e.g., privacy via Opacus)

✅ Modular design (data/model/client/server are decoupled)

🔬 Dataset Overview
Rows (after cleaning): 257,673

Features used: 41 (numeric only)

Classes: 10 attack categories:

Normal, Exploits, Generic, Reconnaissance, DoS, etc.

🧪 Experimental Runs
Available in Results - FedAVG.docx:

16 runs with different α values (from IID to extreme non-IID)

Also includes 10-client experiments and fractional participation tests

