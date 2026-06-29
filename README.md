# Regression-Analysis-business-Use-case
Company: Lumina HealthPath, A HealthTech &amp; Preventative Diagnostics
# Project: Patient Risk Prediction using Logistic Regression

## Overview
This project aims to develop a machine learning model to predict a binary `Risk_Category` for synthetic patients based on various health-related features. The process involves data generation, rigorous Exploratory Data Analysis (EDA), feature engineering, model training, evaluation, and hyperparameter tuning of a Logistic Regression classifier.

## Dataset
A synthetic dataset of 50,000 patient records was generated, comprising the following features:
- `Patient_ID`: Unique identifier for each patient.
- `Age`: Age of the patient.
- `Gender`: Gender of the patient (Male/Female).
- `BMI`: Body Mass Index.
- `Glucose`: Blood Glucose level.
- `Smoking`: Binary indicator (0/1) for smoking status.
- `Alcohol`: Binary indicator (0/1) for alcohol consumption.
- `Exercise`: Binary indicator (0/1) for regular exercise.
- `Risk_Category`: Binary target variable (0/1) assigned based on the rule `(BMI > 30) | (Glucose > 140)`. This indicates whether a patient is at high risk.

## Project Steps & Findings

### 1. Data Generation
- A synthetic dataset of 50,000 patient records was created with a predefined rule for `Risk_Category` assignment to simulate a health risk scenario.

### 2. Exploratory Data Analysis (EDA)
- **Missing Values:** No missing values were found, negating the need for imputation.
- **Data Types & Structure:** All features had appropriate data types, and the dataset contained 50,000 non-null entries.
- **Descriptive Statistics:** Mean, Median, and Variance were explicitly calculated for numerical features (`Age`, `BMI`, `Glucose`), providing insights into their central tendency and spread.
    - **Age:** Mean: 53.51, Median: 53.00, Variance: 433.11
    - **BMI:** Mean: 29.94, Median: 29.90, Variance: 75.20
    - **Glucose:** Mean: 135.19, Median: 135.50, Variance: 1410.18
- **Feature Correlation Heatmap:** A heatmap revealed strong positive correlations between `BMI`, `Glucose`, and the `Risk_Category`, consistent with the data generation logic. Other features showed negligible correlation with the target.
- **Data Visualization:** Histograms for numerical features showed relatively uniform distributions. Count plots for categorical features displayed their respective distributions. Box plots and grouped count plots highlighted the strong relationship between `Risk_Category` and high `BMI`/`Glucose` values.

### 3. Feature Engineering and Data Preparation
- **One-Hot Encoding:** The `Gender` categorical feature was one-hot encoded into `Gender_Male`.
- **Train-Test Split:** The dataset was split into 70% training and 30% testing sets, with stratification to maintain the `Risk_Category` distribution across both sets.
- **Feature Scaling:** Numerical features (`Age`, `BMI`, `Glucose`) were scaled using `StandardScaler` to normalize their ranges, which is crucial for many machine learning algorithms.

### 4. Model Selection and Training
- A Logistic Regression model was chosen as a baseline binary classifier and trained on the preprocessed training data.

### 5. Model Evaluation
- **Initial Model Evaluation (before scaling):** The model achieved an accuracy of ~91.0%, with strong recall (0.95) for `Risk_Category=1`.
- **Re-evaluation with Scaled Features:** The model was re-trained and re-evaluated with scaled features. The performance metrics remained largely consistent, indicating that for this specific Logistic Regression model and dataset, scaling did not significantly alter the predictive power. The recall for `Risk_Category 1` was 0.9493, meeting the target of `>= 0.85`.
- **Confusion Matrix Visualization:** A heatmap of the confusion matrix was generated, providing a clear visual breakdown of true positives, true negatives, false positives, and false negatives.
- **Precision-Recall Curve Analysis:** A Precision-Recall curve was plotted, showing a high AUC of 0.99, demonstrating the model's excellent ability to distinguish positive cases.
- **Final Model Performance Report:** A comprehensive report detailed accuracy, recall (for `Risk_Category 1`), F1-score, and the full classification report.
- **Threshold Adjustment Recommendations:** The model's current recall (0.9493) already exceeded the target of 0.85, so no threshold adjustment was deemed necessary to meet this specific criterion.

### 6. Hyperparameter Tuning
- **GridSearchCV** was employed to optimize the Logistic Regression model's hyperparameters (`C` and `solver`). The `scoring='recall'` metric was used during the search to prioritize identifying high-risk individuals.
- **Best Parameters Found:** `{'C': 0.001, 'solver': 'liblinear'}`.
- **Best Recall Score (from cross-validation):** 0.9803.
- **Optimized Model Performance:** The model was re-trained with the best hyperparameters and re-evaluated:
    - **Accuracy:** 0.8085
    - **Recall (for `Risk_Category 1`):** 0.9787. This significantly exceeds the target recall of 0.85.
    - It was noted that while recall improved, there was a trade-off with precision for `Risk_Category 0`, leading to a lower overall accuracy compared to the unoptimized model. This highlights the impact of optimizing for a specific metric (recall in this case).

## Conclusion
The project successfully built and optimized a Logistic Regression model for patient risk prediction. The model demonstrates a very high recall for identifying high-risk patients, which was a key objective. Although hyperparameter tuning increased the recall further, it also introduced a trade-off with the precision of the lower risk category, a common consideration in imbalanced classification or when specific performance metrics are prioritized.
