# ML Final Project
Final Project for Ewha Machine Learning
- Group Name: Exchange Team
- Group Members: Ellen Kim, Alexandre Roger Privat, Bianca Dina Marine Muccini, Alice Juliette Loustau, Maya Oh Holmen, Jennifer Jang​


## Setup and Requirements

### Prerequisites
- **Python Version**: 3.8 or higher
- **Jupyter Notebook** or **Google Colab** environment

### Dependencies
To run the code, you need to install the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`

You can install all dependencies using the following command:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
```

### How to Run
1.  Ensure the dataset files (`mon_standard.pkl` and `unmon_standard10.pkl`) are located in the same directory as the notebook/script.
2.  Launch the notebook environment:
    ```bash
    jupyter notebook
    ```
3.  Open the project file and **run all cells sequentially** to reproduce the preprocessing, training, and visualization steps.

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

```

### 4. Sequence Truncation and Padding

To create uniform input vectors for our models, all sequences are standardized to a fixed length:

- **Maximum Length**: 10,000 packets
- **Method**:
  - Sequences longer than 10,000 are truncated.
  - Sequences shorter than 10,000 are padded with zeros at the end.

## Feature Engineering

Instead of using raw timestamp/size arrays directly, we extract statistical and behavioral features to capture unique traffic patterns. A total of **65 handcrafted features** are extracted for each trace, including:

1.  **Statistical Features**: Total packets, incoming/outgoing counts, total bytes, mean/std/median/min/max packet sizes.
2.  **Burst Features**: Analysis of consecutive packets moving in the same direction (bursts). We calculate the number of bursts, max/avg/std burst length, size, and duration.
3.  **Timing Features**: Session duration, inter-packet delays (mean, std, median, min, max), and specific percentiles (10th, 25th, 50th, 75th, 90th).
4.  **Directional Patterns**: Number of direction changes and N-gram counts (e.g., bigrams like "Outgoing-Incoming", trigrams like "Out-Out-In").
5.  **Traffic Concentration**: Packet counts in the first and last 20% of the trace, and cumulative byte sums across 10 time windows.

**Scaling**: All features are standardized using `StandardScaler` (zero mean, unit variance) to ensure optimal performance for distance-based algorithms like KNN and SVM.

## Experimental Scenarios

We evaluated our models under two distinct scenarios:

### 1. Closed World Scenario (Multi-class Classification)
- **Objective**: Identify exactly which monitored website the user visited.
- **Classes**: 95 distinct websites (Labels 0-94).
- **Data**: Only the 19,000 monitored traces.
- **Split**: 70% Train / 15% Validation / 15% Test.

### 2. Open World Scenario (Binary Classification)
- **Objective**: Determine *if* the user is visiting a monitored website or an unmonitored one.
- **Classes**: Binary (1 = Monitored, 0 = Unmonitored).
- **Data**: Combined dataset of 19,000 monitored and 10,000 unmonitored traces.
- **Split**: 70% Train / 15% Validation / 15% Test.

## Models and Hyperparameter Tuning

We implemented and compared three different machine learning algorithms. For KNN and SVM, we utilized `GridSearchCV` to optimize hyperparameters.

### K-Nearest Neighbors (KNN)
- **Closed World Results**:
  - Base Accuracy: ~67.8%
  - **Tuned Accuracy**: **~71.1%** (Parameters: `n_neighbors=3`, `weights='distance'`)
- **Open World Results**:
  - **Accuracy**: **~92.1%** (Hyperparameter tuning yielded negligible improvement over the base model).

### Support Vector Machine (SVM)
- **Closed World Results**:
  - Base Accuracy: ~59.7%
  - **Tuned Accuracy**: **~77.7%** (Parameters: `C=100`, `kernel='rbf'`)
  - *Observation*: Significant improvement (+18%) with parameter tuning.
- **Open World Results**:
  - Base Accuracy: ~99.1%
  - **Tuned Accuracy**: **~99.4%** (Parameters: `C=10`, `kernel='rbf'`)

### Random Forest (Tree Ensemble)
- **Closed World Results**:
  - **Test Accuracy**: **~84.2%**
  - *Top Features*: `first_k_incoming`, `first_k_outgoing`, `early_out_in_ratio`, `bigram_OO`.
- **Open World Results**:
  - **Test Accuracy**: **~99.9%**
  - *Top Features*: `time_to_first_incoming`, `first_k_incoming`, `cumulative_bytes_window_0`.


## Random Forest Classification (Closed World vs Open World)

This section trains and evaluates Random Forest models on both closed-world and open-world datasets. For each setting, the code:

- Runs a RandomForestClassifier with fixed hyperparameters  
- Performs 5-fold cross-validation  
- Computes test accuracy  
- Extracts the top 5 most important features  
- Plots the top 20 feature importances

## Resource Estimation

Based on the dataset size and computational complexity of the algorithms used, we estimate the following resource consumption for reproducing these results:

- **Memory (RAM)**: **~8 GB to 16 GB**
  - The raw arrays (`int32` timestamps and `int16` sizes) for ~30,000 samples padded to length 10,000 require approximately 1.5 GB - 2 GB of RAM just for storage.
  - Feature extraction and model training (especially Random Forest with 1,000 trees and Grid Search with cross-validation) significantly increase peak memory usage.
- **CPU Cores**: **4+ Cores recommended**
  - The code utilizes `n_jobs=-1` in `KNeighborsClassifier` and `GridSearchCV`, leveraging all available CPU cores for parallel processing.
- **Compute Time**:
  - *Loading datafile*: ~2 minutes.
  - *Preprocessing*: ~30-60 seconds.
  - *Feature Extraction*: ~2-5 minutes.
  - *Model Training KNN and SVM (Base)*: ~1 minute per model.
  - *Grid Search (Tuning)*: ~2 minutes.
  - *Model Training Tree Ensemble (Base)*: ~12 minutes depending on CPU speed.

## Conclusion

Our analysis shows that **Random Forest** outperforms both SVM and KNN in both scenarios, achieving **84.2%** accuracy in the Closed World task and near-perfect **99.9%** accuracy in the Open World task.  
It would be interesting to explore in future work the extent to which the Closed World performance could be further enhanced through rigorous hyperparameter tuning (e.g., via GridSearchCV) and by testing a broader range of hyperparameters such as tree depth and split criteria.

Feature importance analysis reveals that **burst patterns** and **early packet timing** are the most critical indicators for fingerprinting websites, suggesting that the initial handshake and loading phase contains the most distinguishing information. 




