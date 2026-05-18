# Real-Time Anomaly Detection in Financial Transactions

## 📌 Project Overview
Financial fraud is a multi-billion dollar problem. For FinTech companies, distinguishing between a legitimate purchase and a fraudulent one in milliseconds is critical. However, fraud is statistically rare—representing just 0.17% of all transactions in this dataset.

This project builds a robust **Unsupervised Anomaly Detection System** designed to handle extreme class imbalance. Instead of teaching a supervised model what fraud looks like (which changes constantly), this system learns what "normal" behavior looks like and flags significant deviations in real-time.

## 📊 The Dataset
The model was trained and evaluated on the standard **Credit Card Fraud Detection** dataset provided by the Machine Learning Group - ULB (via Kaggle).
* **Size:** 284,807 transactions.
* **Features:** 28 PCA-transformed numerical features (`V1` to `V28`), `Time`, and `Amount`.
* **Imbalance:** Only 492 frauds out of 284,807 transactions (0.17%).

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Environment:** Jupyter Notebook
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
* **Algorithms:** Isolation Forest (Tree-based), Local Outlier Factor (Density-based)

## 🧠 Methodology & Execution Pipeline

### 1. Data Preprocessing & EDA
* Applied `RobustScaler` to the `Amount` and `Time` features. Standard scaling is highly sensitive to extreme outliers (like massive fraudulent purchases), so `RobustScaler` was used to scale data based on the median and Interquartile Range (IQR).

### 2. Model Implementation & Memory Optimization
* **Isolation Forest:** Proved highly scalable and efficient for this dataset. To prevent memory exhaustion (RAM blowout) on the 284k+ rows, the algorithm was optimized using `max_samples='auto'` and `n_jobs=-1` for multi-core processing.
* **Local Outlier Factor (LOF):** As a density-based algorithm relying on nearest-neighbor distances, LOF possesses an $O(n^2)$ computational complexity. To prevent system freezing, the dataset was strategically down-sampled to 10% specifically for the LOF evaluation.

### 3. Evaluation Strategy (The Metric Trap)
Traditional `Accuracy` metrics are useless here; a model that blindly guesses "Not Fraud" 100% of the time would achieve 99.8% accuracy. Instead, the models were evaluated strictly on:
* **Precision:** What percentage of flagged transactions were actually fraud? (Minimizing false alarms).
* **Recall:** What percentage of total fraud did we catch?
* **Precision-Recall Curve:** Plotted to find the optimal operating threshold that maximizes the F1-Score.

## 📈 Key Findings & Business Recommendation
The **Isolation Forest** proved to be the superior model for a real-time, high-volume production environment due to its computational efficiency and scalability. 

By analyzing the Precision-Recall curve, we identified the mathematical point that best balances catching fraudulent actors while minimizing the disruption of false alarms for legitimate customers.

**Final Recommendation:** We recommend implementing the Isolation Forest model into the production pipeline using an **Optimal Threshold of -0.0047**, which yields an **Expected Precision of 28.8%** and an **Expected Recall of 31.9%**. This operating point maximizes our ability to intercept fraudulent activity purely through unsupervised learning while maintaining a manageable False Positive rate for customer support operations.

## 🚀 How to Run Locally
1. Clone this repository.
2. Download the `creditcard.csv` dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
3. Extract the CSV file and place it in the same directory as the Jupyter Notebook.
4. Run all cells in `Fraud_Detection_project_2.ipynb`.
