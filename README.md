# KNN-Based Breast Cancer Classification

## Overview
This project demonstrates the use of the k-Nearest Neighbors (KNN) algorithm to classify breast cancer tumors as benign or malignant based on microscopic characteristics. The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which provides tumor characteristics such as clump thickness, cell uniformity, and more.

## Features
- Classification of tumors into **benign** (label 2) or **malignant** (label 4).
- Optimal selection of the number of neighbors (k) using cross-validation.
- Data visualization to understand feature relationships and class distributions.
- Evaluation metrics like accuracy, confusion matrix, and classification report.
- Testing with real and simulated data samples.

---

## Dataset Details
The dataset contains the following attributes:

1. **Sample Code Number**: Unique identifier for each tissue sample (excluded from analysis).
2. **Clump Thickness**: Assessment of tumor cell cluster thickness (1-10).
3. **Uniformity of Cell Size**: Uniformity in tumor cell size (1-10).
4. **Uniformity of Cell Shape**: Uniformity in tumor cell shape (1-10).
5. **Marginal Adhesion**: Adhesion level of tumor cells to surrounding tissue (1-10).
6. **Single Epithelial Cell Size**: Size of individual tumor cells (1-10).
7. **Bare Nuclei**: Count of nuclei without surrounding cytoplasm (1-10).
8. **Bland Chromatin**: Chromatin structure assessment (1-10).
9. **Normal Nucleoli**: Presence of normal-looking nucleoli (1-10).
10. **Mitoses**: Frequency of mitotic cell divisions (1-10).
11. **Class**: Tumor classification - **2 (Benign)** or **4 (Malignant)**.

---

## Workflow
1. **Data Preprocessing**:
   - Handle missing values in the `Bare Nuclei` feature by dropping rows with `NaN` values.
   - Split the dataset into features (`X`) and target variable (`y`).
   - Standardize the features to ensure equal weighting during distance calculations.

2. **Model Training**:
   - Train a KNN model using training data.
   - Use cross-validation to identify the optimal number of neighbors (`k`).

3. **Model Evaluation**:
   - Evaluate the model's performance using accuracy, confusion matrix, and classification report.

4. **Testing**:
   - Test the model on multiple real and simulated samples to validate predictions.

5. **Visualization**:
   - Plot the class distribution.
   - Display a correlation heatmap to understand feature relationships.
   - Use boxplots to compare feature distributions between benign and malignant classes.

---

## Code Implementation
The implementation is organized into the following steps:

1. **Loading and Cleaning Data**: Handle missing values and prepare features for analysis.
2. **Data Visualization**: Explore feature distributions and relationships.
3. **Model Training and Tuning**:
   - Train KNN with an initial `k` value.
   - Tune `k` using cross-validation.
4. **Model Testing**:
   - Use real and simulated data samples to validate predictions.
   - Evaluate results using metrics and visualizations.

---

## Visualizations
1. **Class Distribution**:
   - Bar plot showing the count of benign vs. malignant cases.

2. **Correlation Heatmap**:
   - Matrix displaying the relationships between features, highlighting redundant or influential features.

3. **Boxplots**:
   - Visual representation of feature distributions grouped by tumor class.

4. **Optimal k Plot**:
   - Line graph showing cross-validated accuracy for different values of `k` to identify the best neighbor count.

---

## How to Run
1. Place the dataset file (`breast-cancer-wisconsin.data`) in the working directory.
2. Install required Python libraries if not already installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Execute the Python script in a compatible environment (e.g., Jupyter Notebook, VSCode).

---

## Key Results
- **Optimal k**: Cross-validation determined the best number of neighbors for KNN.
- **Model Accuracy**: Achieved high accuracy in predicting tumor classes.
- **Feature Insights**:
  - Features like `Bare Nuclei` and `Uniformity of Cell Size/Shape` strongly correlate with tumor malignancy.

---

## Future Improvements
- Implement alternative algorithms (e.g., SVM, Random Forest) for comparison.
- Explore feature engineering techniques to enhance classification accuracy.
- Address potential dataset imbalances through resampling techniques.

---

## Author
- Nagesh Sortur

---

## References
- [UCI Machine Learning Repository - Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28original%29)
- Documentation for libraries like Pandas, NumPy, Matplotlib, and Scikit-learn.
