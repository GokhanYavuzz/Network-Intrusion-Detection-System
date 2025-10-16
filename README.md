🧠 Network Intrusion Detection System (NIDS) – Machine Learning Approach
📄 Overview

This project aims to build a Network Intrusion Detection System (NIDS) using machine learning algorithms to identify and classify various types of network attacks based on the UNSW-NB15 dataset.
It involves a full data science workflow, from exploratory data analysis (EDA) to model training, optimization, and evaluation.

📊 Project Structure
│
├── EDA.ipynb                   # Exploratory Data Analysis: data cleaning, visualization, and feature insights
├── ML_Model_Building_.ipynb    # Model training, hyperparameter tuning, evaluation metrics
├── requirements.txt            # Required Python packages
├── .gitignore                  # Files and folders to ignore in Git
├── LICENSE                     # License for the project
└── README.md                   # Project documentation

🧩 Features

Detailed EDA including correlation heatmaps, missing value analysis, and feature importance

Implementation of Random Forest Classifier and other ML algorithms

Class balancing using imblearn to address data imbalance

Model evaluation with cross-validation, accuracy, precision, recall, F1-score, and ROC-AUC

Modular design for easy experimentation and reproducibility

⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/nids-adversarial.git
cd nids-adversarial


Install the required packages:

pip install -r requirements.txt

🚀 How to Run

Launch Jupyter Notebook:

jupyter notebook


Open and run notebooks in order:

EDA.ipynb

ML_Model_Building_.ipynb

📦 Dependencies

The main libraries used in this project:

pandas

numpy

scikit-learn

imbalanced-learn

matplotlib

tqdm

jupyter / notebook

(See requirements.txt for full list.)

📈 Results

Achieved high accuracy and robust generalization in detecting various network attack categories.

Visualizations illustrate feature distributions and model performance across different classes.

🧪 Future Work

Integrate deep learning models (e.g., LSTM, CNN) for sequential packet analysis

Deploy a real-time detection system using Flask or FastAPI

Optimize for adversarial robustness

📜 License

This project is licensed under the terms specified in the LICENSE
 file.

👤 Author

Gökhan Yavuz
