import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE  # imblearn library can be installed using pip install imblearn
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Importing dataset and examining it
dataset = pd.read_csv("C:\Program Files\JetBrains\PyCharm Community Edition 2022.2.2\Bin\CustomerChurn.csv")
pd.set_option('display.max_columns', None) # to make sure you can see all the columns in output window
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())



# Converting Categorical features into Numerical features
dataset['Gender'] = dataset['Gender'].map({'F': 1, 'M': 0})
dataset['Education_Level'] = dataset['Education_Level'].map({'Doctorate': 4, 'Graduate': 2, 'High School': 1, 'Post-Graduate': 3, 'Uneducated' : 0 })
dataset['Income_Category'] = dataset['Income_Category'].map({'Less than $40K': 0, '$40K - $60K': 1, '$60K - $80K': 2, '$80K - $120K': 3, '$120K +': 4 })
dataset['Card_Category'] = dataset['Card_Category'].map({'Blue': 0, 'Gold': 2, 'Platinum': 3, 'Silver': 1,})
dataset['Attrition_Flag'] = dataset['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1,})
print(dataset.info())


categorical_features = ['Marital_Status']
final_data = pd.get_dummies(dataset, columns = categorical_features)
print(final_data.info())
print(final_data.head(2))


# Dividing dataset into label and feature sets
X = final_data.drop('Attrition_Flag', axis = 1) # Features
Y = final_data['Attrition_Flag'] # Labels
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)



#X_ = final_data[['Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal']]
X_ = final_data[['Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal','Total_Relationship_Count','Months_Inactive','Contacts_Count','Credit_Limit','Customer_Age','Months_on_book','Dependent_count']]
# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X_)

# Implementing Random Forest Classifier
# Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
model = Pipeline([
        ('balancing', SMOTE(random_state = 101)),
        ('classification', RandomForestClassifier(criterion='entropy', max_features='auto', random_state=1) )
    ])
grid_param = {'classification__n_estimators': [10,20,30,40,100]}

gd_sr = GridSearchCV(estimator=model, param_grid=grid_param, scoring='recall', cv=10)

#In the above GridSearchCV(), scoring parameter should be set as follows:
#scoring = 'accuracy' when you want to maximize prediction accuracy
#scoring = 'recall' when you want to minimize false negatives
#scoring = 'precision' when you want to minimize false positives
#scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)


gd_sr.fit(X_scaled, Y)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)

#featimp = pd.Series(gd_sr.best_estimator_.named_steps["classification"].feature_importances_, index=list(X)).sort_values(ascending=False) # Getting feature importances list for the best model
#print(featimp)


#Selecting features with higher sifnificance and redefining feature set
#X_ = final_data[['Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal']]
X_ = final_data[['Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal','Total_Relationship_Count','Months_Inactive','Contacts_Count','Credit_Limit','Customer_Age','Months_on_book','Dependent_count']]

#X_ = final_data[['Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal']]
feature_scaler = StandardScaler()
X_scaled_ = feature_scaler.fit_transform(X_)

#Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
model = Pipeline([
        ('balancing', SMOTE(random_state = 101)),
        ('classification', RandomForestClassifier(criterion='entropy', max_features='auto', random_state=1) )
    ])
grid_param = {'classification__n_estimators': [150,200,250,300,350]}

gd_sr = GridSearchCV(estimator=model, param_grid=grid_param, scoring='recall', cv=10)


#In the above GridSearchCV(), scoring parameter should be set as follows:
#scoring = 'accuracy' when you want to maximize prediction accuracy
#scoring = 'recall' when you want to minimize false negatives
#scoring = 'precision' when you want to minimize false positives
#scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)


gd_sr.fit(X_scaled_, Y)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)

#featimp = pd.Series(gd_sr.best_estimator_.named_steps["classification"].feature_importances_, index=list(X)).sort_values(ascending=False) # Getting feature importances list for the best model
#print(featimp)


X_ = final_data[['Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal','Total_Relationship_Count','Months_Inactive','Contacts_Count','Credit_Limit','Customer_Age','Months_on_book','Dependent_count']]
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X_)
print("------------------------------------Linear------------------------------------------------")
##################################################################################
# Implementing Support Vector Classifier
# Tuning the kernel parameter and implementing cross-validation using Grid Search
model = Pipeline([
        ('balancing', SMOTE(random_state = 101)),
        ('classification', SVC(random_state=1))
    ])
grid_param = {'classification__kernel': ['linear'], 'classification__C': [.001,.01,.1,1,10,100]}

gd_sr = GridSearchCV(estimator=model, param_grid=grid_param, scoring='recall', cv=10)

"""
In the above GridSearchCV(), scoring parameter should be set as follows:
scoring = 'accuracy' when you want to maximize prediction accuracy
scoring = 'recall' when you want to minimize false negatives
scoring = 'precision' when you want to minimize false positives
scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
"""

gd_sr.fit(X_scaled, Y)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)

###########################ImplementingSVC#####################################################

print("----------------------------Poly-----------------------------------")
# Normalizing numerical features so that each feature has mean 0 and variance 1
#X_ = final_data[['Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal']]
#X_ = final_data[['Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal','Total_Relationship_Count','Months_Inactive','Contacts_Count','Credit_Limit','Customer_Age','Months_on_book','Dependent_count','Income_Category','Education_Level','Gender','Marital_Status_Married','Marital_Status_Single','Card_Category','Marital_Status_Divorced']]
X_ = final_data[['Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal','Total_Relationship_Count','Months_Inactive','Contacts_Count','Credit_Limit','Customer_Age','Months_on_book','Dependent_count']]
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X_)

##################################################################################
# Implementing Support Vector Classifier
# Tuning the kernel parameter and implementing cross-validation using Grid Search
model = Pipeline([
        ('balancing', SMOTE(random_state = 101)),
        ('classification', SVC(random_state=1))
    ])
grid_param = {'classification__kernel': ['poly'], 'classification__C': [.001,.01,.1,1,10,100]}

gd_sr = GridSearchCV(estimator=model, param_grid=grid_param, scoring='recall', cv=10)

"""
In the above GridSearchCV(), scoring parameter should be set as follows:
scoring = 'accuracy' when you want to maximize prediction accuracy
scoring = 'recall' when you want to minimize false negatives
scoring = 'precision' when you want to minimize false positives
scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
"""

gd_sr.fit(X_scaled, Y)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)


print("-------------------------------------RBF-----------------------------------------------------------")


# Normalizing numerical features so that each feature has mean 0 and variance 1
#X_ = final_data[['Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal']]
#X_ = final_data[['Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal','Total_Relationship_Count','Months_Inactive','Contacts_Count','Credit_Limit','Customer_Age','Months_on_book','Dependent_count','Income_Category','Education_Level','Gender','Marital_Status_Married','Marital_Status_Single','Card_Category','Marital_Status_Divorced']]
X_ = final_data[['Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal','Total_Relationship_Count','Months_Inactive','Contacts_Count','Credit_Limit','Customer_Age','Months_on_book','Dependent_count']]
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X_)

##################################################################################
# Implementing Support Vector Classifier
# Tuning the kernel parameter and implementing cross-validation using Grid Search
model = Pipeline([
        ('balancing', SMOTE(random_state = 101)),
        ('classification', SVC(random_state=1) )
    ])
grid_param = {'classification__kernel': ['rbf'], 'classification__C': [.001,.01,.1,1,10,100]}

gd_sr = GridSearchCV(estimator=model, param_grid=grid_param, scoring='recall', cv=10)

"""
In the above GridSearchCV(), scoring parameter should be set as follows:
scoring = 'accuracy' when you want to maximize prediction accuracy
scoring = 'recall' when you want to minimize false negatives
scoring = 'precision' when you want to minimize false positives
scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
"""

gd_sr.fit(X_scaled, Y)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)

print("----------------------------------------Sigmoid--------------------------------------------------------------")

# Normalizing numerical features so that each feature has mean 0 and variance 1
#X_ = final_data[['Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal']]
#X_ = final_data[['Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal','Total_Relationship_Count','Months_Inactive','Contacts_Count','Credit_Limit','Customer_Age','Months_on_book','Dependent_count','Income_Category','Education_Level','Gender','Marital_Status_Married','Marital_Status_Single','Card_Category','Marital_Status_Divorced']]
X_ = final_data[['Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal','Total_Relationship_Count','Months_Inactive','Contacts_Count','Credit_Limit','Customer_Age','Months_on_book','Dependent_count']]
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X_)

##################################################################################
# Implementing Support Vector Classifier
# Tuning the kernel parameter and implementing cross-validation using Grid Search
model = Pipeline([
        ('balancing', SMOTE(random_state = 101)),
        ('classification', SVC(random_state=1) )
    ])
grid_param = {'classification__kernel': ['sigmoid'], 'classification__C': [.001,.01,.1,1,10,100]}

gd_sr = GridSearchCV(estimator=model, param_grid=grid_param, scoring='recall', cv=10)

"""
In the above GridSearchCV(), scoring parameter should be set as follows:
scoring = 'accuracy' when you want to maximize prediction accuracy
scoring = 'recall' when you want to minimize false negatives
scoring = 'precision' when you want to minimize false positives
scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
"""

gd_sr.fit(X_scaled, Y)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)



