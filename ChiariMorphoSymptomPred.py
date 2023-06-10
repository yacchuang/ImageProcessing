# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:31:37 2023

@author: Ya-Chen.Chuang
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

df = pd.read_excel('C:/Users/ya-chen.chuang/Documents/Python Scripts/PYpractice/symptoms_correlation.xlsx', sheet_name='symptoms_new2')

print(df.head())

## Extraxt relavent data to train
#surgery_morpho = ['Surgery', 'TonsilV', 'CBLv ', 'BSv', '4thVentricle', 'Tonsil length', '(CMa+Ta)/FMa', 'Clivo-occipital', 'Boogard Angle', 'Occipital angle', 'Clivus canal angle']
#surgery_morpho = ['Surgery', 'Tonsil length', '(CMa+Ta)/FMa']
# df_surgery = df[surgery_morpho]

morpho = ['TonsilV', 'CBLv', 'BSv', '4thV', 'TonsilL', '(CMa+Ta)/FMa', 'Clivo-occipital', 'BoogardA', 'OccipitalA', 'Clivus-canalA', 'Surgery']

df_cor = df[morpho] 
df_clean = df_cor.dropna()
print(df_clean.head())

df_clean.Surgery[df_clean.Surgery == "Y"] =1
df_clean.Surgery[df_clean.Surgery == "N"] =0

# df_clean = df_surgery.dropna()

Y = df_clean['Surgery'].values
Y = Y.astype('int')

X = df_clean.drop(labels=['Surgery'], axis = 1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)


#Scale data, otherwise model will fail.
#Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# define the model
#Experiment with deeper and wider networks
model = Sequential()
model.add(Dense(128, input_dim=10, activation ='relu'))
model.add(Dense(64, activation='relu'))
#Output layer
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.summary()

history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs =500)

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['mean_absolute_error']
val_acc = history.history['val_mean_absolute_error']
plt.plot(epochs, acc, 'y', label='Training MAE')
plt.plot(epochs, val_acc, 'r', label='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


##############################################
#Predict on test data
predictions = model.predict(X_test_scaled[:10])
print("Predicted values are: ", predictions)
print("Real values are: ", y_test[:10])
##############################################


##############################################
# Logistic Regression
from sklearn.linear_model import LogisticRegression

LRmodel = LogisticRegression(random_state = None, solver='liblinear')
LRmodel.fit(X_train, y_train)

## Classification report
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

log_y_pred = LRmodel.predict(X_test)
log_p_pred = LRmodel.predict_proba(X_test)
log_score = LRmodel.score(X_train, y_train)
log_acc =  accuracy_score(y_test, log_y_pred)
log_conf_m = confusion_matrix(y_test, log_y_pred)
log_report = classification_report(y_test, log_y_pred)

print("Accuracy=", accuracy_score(y_test, log_y_pred))


##############################################
# Decision tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

tree = DecisionTreeRegressor()
tree.fit(X_train_scaled, y_train)
tree_y_pred = tree.predict(X_test_scaled)
mse_dt = mean_squared_error(y_test, tree_y_pred)
mae_dt = mean_absolute_error(y_test, tree_y_pred)
print('Mean squared error using decision tree: ', mse_dt)
print('Mean absolute error using decision tree: ', mae_dt)


##############################################
# Random Forest
# Increase number of tress and see the effect
from sklearn.ensemble import RandomForestRegressor
RFmodel = RandomForestRegressor(n_estimators = 30, random_state=30)
RFmodel.fit(X_train_scaled, y_train)

RF_y_pred = RFmodel.predict(X_test_scaled)

mse_RF = mean_squared_error(y_test, RF_y_pred)
mae_RF = mean_absolute_error(y_test, RF_y_pred)
print('Mean squared error using Random Forest: ', mse_RF)
print('Mean absolute error Using Random Forest: ', mae_RF)


'''
## matplotlib scatter funcion w/ logistic regression
plt.scatter(X,Y)
plt.plot(X,model.predict_proba(X_test), color='red')
plt.xlabel("all features")
plt.ylabel("Probability of Surgery")
'''


## Feature importancy list
feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)
