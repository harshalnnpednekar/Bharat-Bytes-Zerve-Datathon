# 🥉 Bharat Bytes - Zerve AI Datathon 2025 3rd Prize Winner

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Optimized-green.svg)](https://catboost.ai/)
[![Zerve AI](https://img.shields.io/badge/Platform-Zerve_AI-black.svg)](https://zerve.ai/)

> **3rd Prize Winning Solution for the Zerve AI Datathon 2025 held at Techfest, IIT Bombay.**
> 
> **Team Bharat Bytes:**
> - Vedang Mendhurwar
> - Ruturaj Rajwade
> - Harshal Pednekar

---

## 📖 Project Overview

This repository contains the 3rd prize winning machine learning pipeline and exploratory data analysis (EDA) suite developed by **Team Bharat Bytes** for the Zerve AI Datathon. The core challenge involved predicting a highly imbalanced binary classification target (with a severe 26.4:1 ratio) based on a complex, high-dimensional dataset containing both numerical and categorical features.

Our solution focuses on robust data stratification, intelligent categorical feature processing, and a highly optimized **CatBoost Classifier**. The model was specifically tuned to capture non-linear relationships while effectively mitigating overfitting on the minority class. Furthermore, we engineered a comprehensive, beautifully styled visualization suite that matched the premium Zerve Design System aesthetic, giving us a distinct edge in presentation and interpretability.

## 🚀 Key Features and Innovations

### 1. Robust Modeling Pipeline (`Final_Model.py`)
- **Rigorous Stratification:** Implemented a strict 60/20/20 (Train/Validation/Holdout) dataset split strategy to ensure that the severe class imbalance is consistently and accurately represented across all sets.
- **Automated Preprocessing:** Intelligent detection and string-casting of categorical variables to natively leverage CatBoost's powerful, out-of-the-box categorical handling capabilities.
- **Optimized CatBoost Classifier:** Fine-tuned hyperparameters (`depth=8`, `learning_rate=0.02`, `loss_function="Logloss"`, `eval_metric="AUC"`) combined with early stopping (`early_stopping_rounds=150`) to maximize the Area Under the ROC Curve (AUC).
- **In-depth Log Analysis & Performance:** Automated parsing of `catboost_info` TSV files to dynamically identify best iterations and learning trajectories post-training. The model ultimately achieved a highly competitive **Gini score of ~0.31**.

### 🧠 Detailed Machine Learning Model Architecture

The core of our predictive pipeline is built upon the **CatBoost Classifier**, a state-of-the-art gradient boosting algorithm on decision trees. We selected CatBoost over other tree-based models (like XGBoost or LightGBM) primarily for its native, highly efficient handling of categorical features and its built-in mechanisms to combat overfitting—which is crucial when dealing with a severe 26.4:1 class imbalance.

**Hyperparameter Engineering & Rationale:**
- **`iterations=3000` & `early_stopping_rounds=150`:** We initialized a massive forest of 3000 trees but utilized early stopping based on validation performance. If the validation score didn't improve for 150 consecutive rounds, training halted. This guarantees we extract maximum predictive power without overfitting.
- **`depth=8` & `grow_policy="Lossguide"`:** A tree depth of 8 allows the model to capture deep, complex non-linear feature interactions. The `Lossguide` grow policy (leaf-wise growth) prioritizes expanding the nodes that yield the highest loss reduction, optimizing performance.
- **`learning_rate=0.02`:** A slow, deliberate learning rate ensures stable convergence and finer parameter adjustments across the boosting stages.
- **`loss_function="Logloss"` & `eval_metric="AUC"`:** While Logloss drives the gradient descent, we specifically evaluated on **AUC (Area Under the ROC Curve)**. For highly imbalanced datasets, accuracy is misleading; AUC accurately measures the model's ability to rank positive events higher than negative ones. Ultimately, this rigorous tuning and evaluation pipeline yielded an impressive **Gini score of ~0.31**.
- **Regularization (`l2_leaf_reg=10` & `random_strength=2.0`):** We applied heavy L2 regularization to leaf weights and injected randomness into the split scoring. This acts as a robust defense mechanism against the model memorizing the training data.
- **Subsampling (`bootstrap_type="Bernoulli"`, `subsample=0.85`, `rsm=0.85`):** By randomly sampling 85% of the data and 85% of the features (Random Subspace Method) for each tree, we increased the diversity of the trees, significantly improving the ensemble's generalization to unseen test data.
- **Categorical Handling (`one_hot_max_size=2`, `max_ctr_complexity=3`):** Specifically tuned how CatBoost natively processes and combines categorical features to prevent data leakage and dimensionality explosion.

### 2. Premium Visualization Suite (`visualization_insights.ipynb`)
We created a dedicated Jupyter Notebook engineered to provide deep exploratory insights and model evaluation metrics, all rendered with a striking, dark-mode Zerve-themed aesthetic:
- **Target Distribution Chart:** A stylized bar chart that immediately highlights the extreme class imbalance.
- **Missing Value Matrix:** A seaborn-powered heatmap detailing data sparsity and density across all features.
- **Feature Correlations:** Clear horizontal bar charts of the top 20 numerical features Pearson-correlated with the target variable.
- **Stratified Feature Distributions:** Overlaid KDE density plots demonstrating exactly how feature distributions shift between `Class 0` and `Class 1`.
- **Interactive Learning Curves:** Visualizations of training vs. validation logloss and AUC progression to transparently demonstrate the model's resistance to overfitting over thousands of iterations.

## 📂 Repository Structure

- `Final_Model.py` - The core end-to-end Python script covering data loading, processing, model training, and metric logging.
- `visualization_insights.ipynb` - The standalone Jupyter Notebook containing the premium Zerve-styled EDA and evaluation charts.
- `training_data.csv` & `test_data.csv` - The core datasets provided for the challenge.
- `catboost_info/` - Directory automatically generated by CatBoost during training, tracking iteration-wise metrics.
- `ans.csv` - The final prediction probability file outputted by the best model iteration.
- `Zerve ai datathon presentation.pdf` - The final presentation deck delivered at Techfest, IIT Bombay outlining our methodology, data pipeline, and findings.

## 🛠️ Getting Started

### Prerequisites
To reproduce this environment locally, ensure you have Python 3.8+ installed along with the following crucial dependencies:
```bash
pip install pandas numpy scikit-learn catboost matplotlib seaborn scipy jupyter nbconvert
```

### Execution
1. **Model Training:**
   Place `training_data.csv` and `test_data.csv` in the root directory. Execute the training script to build the CatBoost model and generate the underlying training logs:
   ```bash
   python Final_Model.py
   ```

2. **Generate Visualization Insights:**
   After training the model, run the Jupyter Notebook to explore the data and evaluate the performance learning curves:
   ```bash
   python -m nbconvert --to notebook --execute --inplace visualization_insights.ipynb
   ```
   You can then open `visualization_insights.ipynb` to view all of the generated charts directly inline.

## 🏆 Acknowledgements
We express our immense gratitude to the organizers of **Techfest, IIT Bombay** and the sponsors from **Zerve AI** for conceptualizing and hosting such a challenging and engaging datathon. This achievement is a proud testament to the rigorous experimentation, late-night coding, and seamless synergy of Team Bharat Bytes.

---
*Built with ❤️ by Team Bharat Bytes.*
