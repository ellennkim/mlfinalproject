# ML Final Project
Final Project for Ewha Machine Learning
- Group Name: Exchange Team
- Group Members: Ellen Kim, Alexandre Roger Privat, Bianca Dina Marine Muccini, Alice Juliette Loustau, Maya Oh Holmen, Jennifer Jang​

## Data Preprocessing

This project uses two datasets provided as pickle files:

- `mon_standard.pkl` — 19,000 **monitored** website traces  
- `unmon_standard10.pkl` — 10,000 **unmonitored** website traces  

Each trace corresponds to a single website visit and is represented as a sequence of encrypted packets (integers encoding both direction and timing).

---

### 1. Loading and Basic Representation

We first load the pickle files and convert them into separate arrays:

- **`X1_mon`, `X1_unmon`**: timestamp sequences for each trace  
- **`X2_mon`, `X2_unmon`**: directional size sequences, where:
  - outgoing packets are mapped to `+512`
  - incoming packets are mapped to `-512`
- **`y_mon`**: labels for monitored traces, in `{0, 1, ..., 94}`  
  (unmonitored traces are unlabeled at this stage)

Each list element in `X1_*` and `X2_*` represents one complete website visit.

---

### 2. Removing Corrupted or Incomplete Traces

To ensure data quality, we filter out any traces that:

- are empty (`len(X1[i]) == 0`), or  
- have inconsistent lengths between timestamps and sizes  
  (`len(X1[i]) != len(X2[i])`)

Only traces with non-empty, length-matching `(timestamp, size)` sequences are kept.  
This cleaning step is applied separately to monitored and unmonitored data.

---

### 3. Timestamp Normalization

For each trace, timestamps are normalized to start at zero by subtracting the first timestamp:

```python
t_normalized = t - t[0]

## Random Forest Classification (Closed World vs Open World)

This section trains and evaluates Random Forest models on both closed-world and open-world datasets. For each setting, the code:

- Runs a RandomForestClassifier with fixed hyperparameters  
- Performs 5-fold cross-validation  
- Computes test accuracy  
- Extracts the top 5 most important features  
- Plots the top 20 feature importances
