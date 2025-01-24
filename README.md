# grid_stability
This repository contains the code and analysis for predicting the stability of electrical grids using simulated data. The project leverages machine learning models to classify grid stability based on various electrical grid features, with the goal of providing insights into how these features influence grid behavior and performance.

## Overview
The dataset used in this project simulates the stability of an electrical grid under various conditions. The main task is to predict whether the grid will remain stable or become unstable based on these conditions. The project uses both Logistic Regression and Random Forest models for classification, evaluates model performance with several metrics, and visualizes the results to draw insights.

## Features of the Project

### Data Preprocessing
- **Data Cleaning:** Handling missing values in the dataset to ensure accurate model training.
- **Label Encoding:** The target variable `stabf` is encoded using `LabelEncoder` to convert categorical values into numerical format.
- **Feature Scaling:** Features are scaled using `StandardScaler` to normalize the data, making it compatible for training the models.

### Model Training and Evaluation
#### Models Used:
- **Logistic Regression:**
  - A basic model for binary classification.
  - The model is trained on the scaled training data and evaluated on the test set.
- **Random Forest Classifier:**
  - An ensemble model that can capture complex relationships in the data.
  - The model is trained on the raw (non-scaled) features and evaluated similarly to the logistic regression model.

#### Model Evaluation Metrics:
- **Accuracy:** The proportion of correct predictions made by the model.
- **ROC-AUC:** The area under the receiver operating characteristic curve, indicating the model's ability to distinguish between the stable and unstable grid classes.
- **Precision, Recall, and F1-Score:** These metrics provide insight into the model's performance for each class (stable/unstable).
- **Confusion Matrix:** Visualized for both models, this matrix helps identify the true positives, false positives, true negatives, and false negatives.

### Feature Importance
- **Random Forest Feature Importance:** The importance of each feature in predicting the grid stability is displayed for the Random Forest model. This helps in understanding which features contribute the most to the model's predictions.

### Visualization
- **Confusion Matrix:** A visual representation of the confusion matrix for both models.
- **Feature Importance:** A chart displaying the importance of each feature, generated from the Random Forest model.

---

## Dataset

The dataset used in this project is a simulation of electrical grid stability, available as `Data_for_UCI_named.csv`. It contains various features that describe the state of the grid, with the target variable being `stabf`, which represents the stability of the grid (0 for stable, 1 for unstable).

---

## Preprocessing the Data

- **Label Encoding:**
  - The target variable `stabf` is encoded using `LabelEncoder` to convert categorical values (stable/unstable) into numerical values (0/1).
  
- **Feature Scaling:**
  - Features are scaled using `StandardScaler` to normalize the data and make it suitable for machine learning algorithms that require normalized inputs, such as Logistic Regression.

---

## Model Training

### Logistic Regression
- Logistic Regression is used as a baseline model for binary classification.
- The model is trained on the scaled training data and evaluated on the test set to predict grid stability.

### Random Forest Classifier
- Random Forest is an ensemble model that can handle complex relationships between features and target variables.
- It is trained on the raw features (without scaling) and evaluated on the test set.

---

## Model Evaluation

### Confusion Matrix
- The confusion matrix is used to visualize the true positives, false positives, true negatives, and false negatives for both models. This helps in understanding how well each model is performing.

### ROC-AUC Score
- The ROC-AUC score provides a measure of the model's ability to discriminate between the two classes (stable/unstable grid). The closer the score is to 1, the better the model performs.

### Classification Report
- The classification report includes the following metrics for both models:
  - **Precision:** The proportion of positive predictions that are correct.
  - **Recall:** The proportion of actual positive cases that are correctly identified.
  - **F1-Score:** The harmonic mean of precision and recall.
  - **Support:** The number of occurrences of each class in the test set.

### Feature Importance
- The feature importance is calculated for the **Random Forest** model. This shows which features (columns) contribute the most to the prediction of grid stability. The higher the importance score, the more influence the feature has on the model’s prediction.

---

## Results

After training and evaluating both models, the following metrics are displayed for each model:

- **Accuracy:** The proportion of correct predictions made by the model.
- **ROC-AUC Score:** A measure of the model's ability to distinguish between the two classes (stable and unstable).
- **Confusion Matrix:** A visual representation of how well the models predicted the grid stability.
- **Classification Report:** Detailed metrics for each class, including precision, recall, and F1-score.
- **Feature Importance:** Only available for the Random Forest model, showing the contribution of each feature to the model’s predictions.
