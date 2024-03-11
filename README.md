# Machine Learning Task: Phishing Website Detection

## Overview
This project focuses on detecting phishing websites using various machine learning algorithms. The goal is to analyze and predict whether a website is phishing or not based on different attributes.

## Data
The dataset is sourced from the UCI Machine Learning Repository and includes a range of website attributes labeled as phishing or legitimate. It is pre-processed to handle missing values and convert categorical features into numerical ones.

## Algorithms Used
- Decision Tree Classifier
- Naive Bayes (CategoricalNB)
- Multi-layer Perceptron (MLP) Classifier
- Custom implementations of Perceptron and KNN

## Libraries Required
- Numpy
- Pandas
- Scipy
- Seaborn
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras

## Setup
Ensure you have Python 3.x installed along with the above libraries. You can install them using pip:
```bash
pip install numpy pandas scipy seaborn matplotlib scikit-learn tensorflow keras
```

## Execution
Run the Jupyter Notebook to train and evaluate models on the phishing website dataset. The notebook includes:
- Data loading and preprocessing
- Exploratory data analysis with visualizations such as heatmaps for attribute correlations
- Model training, evaluation, and comparison

## Models and Evaluation
The project evaluates the performance of each model based on accuracy, ROC curve, AUC, confusion matrix, and classification report. Custom functions are utilized for better model evaluation and comparison.

## Conclusion
The notebook provides a comprehensive analysis of phishing website detection using machine learning techniques. It aims to deliver insights into model performance and the importance of various website attributes in phishing detection.
