📌 **Project: Customer Churn Prediction using Machine Learning**

This project focuses on predicting customer churn in the banking sector using machine learning models. The primary objective is to help banks identify customers who are likely to leave, enabling them to take proactive retention measures.

🔍** Key Features:**

Data Preprocessing & Feature Engineering

Encoded categorical variables (Gender, Education, Income, Card Type, etc.)

Created dummy variables for Marital_Status

Normalized numerical features using StandardScaler

Handled class imbalance with SMOTE (Synthetic Minority Over-sampling Technique)

**Model Training & Hyperparameter Tuning**

Implemented Random Forest Classifier with GridSearchCV to optimize estimators.

Implemented Support Vector Classifier (SVC) across multiple kernels (Linear, Poly, RBF, Sigmoid).

Used 10-fold Cross Validation with scoring based on Recall, to minimize false negatives.

**Performance**

Random Forest Classifier achieved ~83.27% accuracy.

Support Vector Classifier with Polynomial kernel (C=0.001) achieved the highest accuracy of 88.16%.

Final model selected: SVC with Poly Kernel.

**📊 Technologies & Libraries:**

Python (NumPy, Pandas, Scikit-learn)

Imbalanced-learn (SMOTE)

Random Forest Classifier

Support Vector Classifier (SVC)

GridSearchCV & Cross Validation
