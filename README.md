Customer Churn Prediction using Machine Learning

This project leverages machine learning to predict customer churn in the banking sector, enabling proactive customer retention strategies and minimizing revenue leakage.

🔍 Business Problem

Customer churn is a critical challenge for banks, directly impacting revenue, market share, and customer lifetime value (CLV). Retaining existing customers is significantly more cost-effective than acquiring new ones, making churn prediction an essential capability for strategic decision-making.

⚙️ Approach
1. Data Preparation & Feature Engineering

- Converted categorical variables (e.g., Gender, Education, Income, Card Type) into machine-readable formats.

- Created dummy variables for Marital Status to capture nuanced customer behavior.

- Standardized numerical features for fair model comparison.

- Applied SMOTE to address class imbalance, ensuring the model identifies churners effectively (high recall focus).

2. Model Development & Validation

- Trained Random Forest Classifier with GridSearchCV to optimize hyperparameters.

- Explored Support Vector Classifier (SVC) with multiple kernels to capture complex decision boundaries.

- Evaluated models using 10-fold Cross Validation, prioritizing recall to minimize false negatives (i.e., ensuring at-risk customers are not missed).

📊 Results & Insights

- Random Forest Classifier: ~83.27% accuracy.

- SVC (Polynomial Kernel, C=0.001): 88.16% accuracy (highest performance).

- Final model chosen: SVC (Poly Kernel) for its superior ability to capture churn patterns.

Business Impact:

- By correctly identifying likely churners, banks can target retention campaigns more effectively, reducing customer attrition.

- Focus on recall-driven modeling ensures fewer “missed churners,” allowing banks to intervene early.

- The model provides an opportunity to segment customers based on churn probability and allocate retention budgets strategically (e.g., offering tailored promotions or personalized financial advice).

🛠️ Technologies & Tools

- Python: NumPy, Pandas, Scikit-learn

- Imbalanced-learn: SMOTE

- Models: Random Forest, SVC

- Optimization & Validation: GridSearchCV, Cross Validation

📈 Business Analyst Perspective

- This solution equips decision-makers with a predictive lens into customer behavior, shifting churn management from reactive to proactive.

- The churn model can be integrated with CRM systems to trigger automated retention workflows (e.g., sending retention offers when churn probability crosses a threshold).

- Financially, reducing churn by even 5–10% could translate into millions in retained revenue annually, given the high lifetime value of banking customers.
