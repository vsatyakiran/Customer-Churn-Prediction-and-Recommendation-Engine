import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,  f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

# Import DataSet
df = pd.read_csv("DataSet.csv")

# Drop CustomerID column as it is unique for each customer
df.drop('CustomerID', axis=1, inplace=True)

# Feature Engineering
# Create a new feature 'AvgMonthlySpend' by dividing 'TotalCharges' by 'Tenure'
df['AvgMonthlySpend'] = df['TotalCharges'] / df['Tenure']

# Create a TotalServiceUsage feature by summing up all the ServiceUsage columns
df['TotalServiceUsage'] = df[['ServiceUsage1', 'ServiceUsage2', 'ServiceUsage3']].sum(axis=1)

# Encoding Categorical Columns
df['Churn'] = df['Churn'].map({'No':0, 'Yes':1})
df['Gender'] = df['Gender'].map({'Female':0, 'Male':1})

# One Hot Encoding for PaymentMethod
ohe = OneHotEncoder(drop='first', sparse_output=False)

ohe.fit(df[['PaymentMethod']])
df_ohe = pd.DataFrame(ohe.transform(df[['PaymentMethod']]), columns=ohe.get_feature_names_out(['PaymentMethod']))


# Save the OneHotEncoder object
pickle.dump(ohe, open('ohe.pkl', 'wb'))

# Oversampling using SMOTE to handle class imbalance
oversample = SMOTE()

X, y = df.drop('Churn', axis=1), df['Churn']

# Standardize the data
scaler = StandardScaler()

# Fit and transform the data
scaler.fit(X[['Tenure', 'MonthlyCharges', 'TotalCharges', 'ServiceUsage1','ServiceUsage2','ServiceUsage3', 'TotalServiceUsage','AvgMonthlySpend']])

X[['Tenure', 'MonthlyCharges', 'TotalCharges', 'ServiceUsage1','ServiceUsage2','ServiceUsage3', 'TotalServiceUsage','AvgMonthlySpend']] = scaler.transform(X[['Tenure','MonthlyCharges', 'TotalCharges', 'ServiceUsage1','ServiceUsage2','ServiceUsage3', 'TotalServiceUsage','AvgMonthlySpend']])

# Save scaler objet 
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Splitting the data into 80% training and 20% testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Differnt combination to find best parameters for the model using GridSearchCV
param_grid_rf = {
    'n_estimators': [50, 75, 100, 115],
    'max_depth': [None, 10, 15, 20, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
param_grid_lr = {
    'penalty': ['l1', 'l2'], 
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
}

# Create Objects for the models
lr = LogisticRegression(max_iter=2000, random_state=42)
rf = RandomForestClassifier()
dt = DecisionTreeClassifier(random_state=42)

# Applying GridSearchCV to find the best parameters for the model
print("Performing Grid Search for Random Forest...")
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
print("Best Score for Random Forest:", grid_search_rf.best_score_)


# Decision Tree
print("\nPerforming Grid Search for Decision Tree...")
grid_search_dt = GridSearchCV(estimator=dt, param_grid=param_grid_dt, cv=5, n_jobs=-1, scoring='accuracy')
grid_search_dt.fit(X_train, y_train)
print("Best Parameters for Decision Tree:", grid_search_dt.best_params_)
print("Best Score for Decision Tree:", grid_search_dt.best_score_)


# Logistic Regression
print("\nPerforming Grid Search for Logistic Regression...")
grid_search_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=5, n_jobs=-1, scoring='accuracy')
grid_search_lr.fit(X_train, y_train)
print("Best Parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best Score for Logistic Regression:", grid_search_lr.best_score_)


# Using the best parameters to train the model
lr = LogisticRegression(**{'C': 10, 'penalty': 'l1', 'solver': 'liblinear'}, max_iter=5000, random_state=42)
rf = RandomForestClassifier(**{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50})
dt = DecisionTreeClassifier(**{'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 10}, random_state=42)

# Training the model
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
dt.fit(X_train, y_train)

# Predicting the test data
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_dt = dt.predict(X_test)

acc_lst = []
pre_lst = []
rec_lst = []
f1_lst = []

# Calculating the performance metrics
for pred in [y_pred_lr, y_pred_rf, y_pred_dt]:
    acc_lst.append(accuracy_score(y_test, pred))
    pre_lst.append(precision_score(y_test, pred))
    rec_lst.append(recall_score(y_test, pred))
    f1_lst.append(f1_score(y_test, pred))

performance = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Decision Tree'],
    'Accuracy': acc_lst,
    'Precision': pre_lst,
    'Recall': rec_lst,
    'F1 Score': f1_lst
})

# Prints the results of the performance metrics
print(performance)

print("Accuracy of Logistic Regression:", accuracy_score(y_test, y_pred_lr))
print("Accuracy of Random Forest:", accuracy_score(y_test, y_pred_rf))
print("Accuracy of Decision Tree:", accuracy_score(y_test, y_pred_dt))

# Save Random Forest Model
pickle.dump(rf, open('rf.pkl', 'wb'))