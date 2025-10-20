# 🧠 Network Intrusion Detection System (NIDS) – Machine Learning Approach


## 📄 Overview


This project develops a **Network Intrusion Detection System (NIDS)** using **machine learning** to detect and classify various network attacks on the **UNSW-NB15 dataset**.

It covers the complete data science pipeline — from **data exploration and preprocessing** to **model training**, **hyperparameter optimization**, and **attack-type-based performance evaluation**.

---

## 📊 Project Structure

```
├── EDA.ipynb                      # Exploratory Data Analysis: cleaning, visualization, and feature insights
├── ML_Model_Building_.ipynb       # Model training, feature scaling, balancing, and hyperparameter tuning
├── detection_by_attack_type.ipynb # Evaluation of model performance per attack type (DoS, Exploit, Fuzzers, etc.)
├── requirements.txt               # Required Python packages
├── .gitignore                     # Files and folders to ignore in Git
├── LICENSE                        # License for the project
└── README.md                      # Project documentation
```

---

## 🧩 Features

* In-depth EDA with correlation heatmaps, feature selection, and class distribution analysis
* Implementation of **Random Forest**, **Decision Tree**, and **Logistic Regression** classifiers
* **SMOTE** balancing to handle imbalanced attack classes
* **GridSearchCV**-based hyperparameter optimization
* Attack-type breakdown via `detection_by_attack_type.ipynb`
* Evaluation with **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC** metrics
* Modular Jupyter notebooks for transparency and reproducibility

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/GokhanYavuzz/Network-Intrusion-Detection-System.git
cd nids-adversarial
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run


1. Launch Jupyter Notebook:
   

2. Run notebooks **in the following order**:
   

   1. `EDA.ipynb` – Data preprocessing & visualization
   2. `ML_Model_Building_.ipynb` – Model training & evaluation
   3. `detection_by_attack_type.ipynb` – Analyze per-attack performance

---

## 📦 Dependencies

Core libraries used:

* `pandas`, `numpy`
* `scikit-learn`
* `imbalanced-learn`
* `matplotlib`, `seaborn`
* `tqdm`
* `jupyter / notebook`

*(Full list in `requirements.txt`.)*

---

## 📈 Results

* Achieved **high detection accuracy** across multiple attack types
* Visual comparisons of **true vs predicted labels** and **confusion matrices**
* Performance visualization for each attack category improves interpretability
