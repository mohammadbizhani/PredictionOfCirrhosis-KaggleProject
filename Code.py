# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 12:01:06 2023

@author: mooha
"""

# importing librairies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier


# Loading datas
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Make a copy from datas
train_data = train.copy()
test_data = test.copy()

# Check for NaN values
train_data.isnull().sum()
test_data.isnull().sum()

# info
train_data.info()
test_data.info()

# describe the datas
train_data.describe()
test_data.describe()

# Remove Id feature
train_data.drop(['id'], axis=1, inplace=True)
test_data.drop(['id'], axis=1, inplace=True)

# plot
plt.figure('count of status')
sns.countplot(data=train_data, x='Status', palette='YlGnBu')
plt.title('Count of each Status')
plt.show()


plt.figure('Heatmap')
sns.heatmap(train_data.select_dtypes('number').corr() , annot=True, cmap='YlGnBu')
plt.title('Heatmap for numeric features', fontweight = "bold")
plt.show()


list_plot = ['Spiders', 'Ascites', 'Hepatomegaly', 'Edema', 'Drug', 'Sex']
plt.figure('Status vs each feature', figsize=(12,5))
for i, col in enumerate(list_plot):
    plt.subplot(2,3, i + 1)
    sns.countplot(data=train_data, x='Status' , hue=col, palette = 'YlGnBu')
    plt.title(f"Status vs {col}")
    plt.tight_layout()
    plt.show()



# standard scaler
SS = StandardScaler()
list_num = ['N_Days' ,'Age', 'Bilirubin', 'Cholesterol', 'Copper', 'Albumin'\
            , 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']
train_data[list_num] = SS.fit_transform(train_data[list_num])
test_data[list_num] = SS.fit_transform(test_data[list_num])

X = train_data.drop(['Status'], axis=1)
Y = train_data['Status']

encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

X_final = pd.get_dummies(X, dtype=float ,drop_first=True)
test_final = pd.get_dummies(test_data, dtype=float ,drop_first=True)


X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y, test_size=0.3 , random_state=1234, stratify=Y)



# XGB
Params = {'max_depth': 6,
          'learning_rate': 0.03711180731953549,
          'n_estimators': 390,
          'min_child_weight': 8,
          'gamma': 8.878825928653752e-05,
          'subsample': 0.5871365948757953,
          'colsample_bytree': 0.175999259700574,
          'reg_alpha': 1.0782222832975881e-08,
          'reg_lambda': 0.002131207411566484}

xgb = XGBClassifier(**Params,random_state=1234)
xgb.fit(X_train, Y_train)

Y_pred = xgb.predict(X_test)
Y_pred_prob = xgb.predict_proba(X_test)
Y_pred_test = xgb.predict(test_final)
Y_pred_prob_test = xgb.predict_proba(test_final)
Score = accuracy_score(Y_test,Y_pred)
   
         
            
# Cat
cat = CatBoostClassifier(learning_rate=0.2, max_depth=8 , iterations=1000, random_state=1234)
cat.fit(X_train, Y_train)

Y_pred_cat = cat.predict(X_test)
Y_pred_prob_cat = cat.predict_proba(X_test)
Y_pred_test_cat = cat.predict(test_final)
Y_pred_prob_test_cat = cat.predict_proba(test_final)
Score_cat = accuracy_score(Y_test, Y_pred_cat)
       


# MLP model
ML = MLPClassifier(random_state=1234, max_iter=1000, solver='adam', alpha=1, hidden_layer_sizes=(10,50))

ML.fit(X_train, Y_train)
Y_pred_ML = ML.predict(X_test)
Y_pred_prob_ML = ML.predict_proba(X_test)
Y_pred_test_ML = ML.predict(test_final)
Y_pred_prob_test_ML = ML.predict_proba(test_final)
score_ML = accuracy_score(Y_test, Y_pred_ML)


            
# Make a submission file           
assert Y_pred_prob_test.shape == (test.shape[0], 3)
submission_labels = ["Status_C", "Status_CL", "Status_D"]
submission = pd.DataFrame({"id": test.id, **dict(zip(submission_labels, Y_pred_prob_test.T))})  
submission.to_csv('submission4.csv', index=False)        
            
            
            
            
            
            