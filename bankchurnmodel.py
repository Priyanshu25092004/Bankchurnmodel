Import Library

## REQUIRED LIBRARIES
# For data wrangling 
import numpy as np
import pandas as pd

# For visualization
import matplotlib.pyplot as plt

import seaborn as sns


# Read the data frame
df = pd.read_csv('/content/Bank Churn Modelling.csv', delimiter=',')
df.shape

df.head()


df.info()

df.duplicated( 'CustomerId').sum()

df=df.set_index('CustomerId')

df.info()

df['Geography'].value_counts()

df.replace({'Geography': {'France': 2, 'Germany' :1, 'Spain' :0}}, inplace=True)

df['Gender'].value_counts()

df.replace({'Gender': {'Female': 1, 'Male' :0}}, inplace=True)

df['Num Of Products'].value_counts()

df.replace({'Num Of Products': {1: 0, 2 :1, 3 :1, 4 :1}}, inplace=True)

df['Has Credit Card'].value_counts()

df['Is Active Member'].value_counts()

df.loc[(df['Balance']==0), 'Churn'].value_counts()

df['Zero Balance']=np.where(df['Balance']>0, 1, 0)

df['Zero Balance'].hist()

df.groupby(['Churn', 'Geography']).count()

df.columns

X=df.drop(['Surname', 'Churn'], axis=1)



Y=df['Churn']



X.shape, Y.shape

df['Churn'].value_counts()

sns.countplot(x='Churn', data=df);

X.shape, Y.shape

from imblearn.under_sampling import RandomUnderSampler

rus=RandomUnderSampler(random_state=2529)

X_rus, Y_rus=rus.fit_resample(X, Y)

X_rus.shape, Y_rus.shape, X.shape, Y.shape

Y.value_counts()

Y_rus.value_counts()

Y_rus.plot(kind='hist')

from imblearn.over_sampling import RandomOverSampler

ros=RandomOverSampler(random_state=2529)

X_ros, Y_ros=ros.fit_resample(X, Y)

X_ros.shape, Y_ros.shape, X.shape, Y.shape

Y.value_counts()  

Y_ros.value_counts()

Y_ros.plot(kind='hist')

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test=train_test_split(X_ros, Y_ros, test_size=0.3, random_state=2529)

X_train_rus, X_test_rus, Y_train_rus, Y_test_rus=train_test_split(X_rus, Y_rus, test_size=0.3, random_state=2529)

X_train_ros, X_test_ros, Y_train_ros, Y_test_ros=train_test_split(X_ros, Y_ros, test_size=0.3, random_state=2529)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train[[ 'CreditScore','Age','Tenure', 'Balance','Estimated Salary']]=sc.fit_transform(X_train[[ 'CreditScore','Age','Tenure', 'Balance','Estimated Salary']])
X_test[[ 'CreditScore','Age','Tenure', 'Balance','Estimated Salary']]=sc.fit_transform(X_test[['CreditScore','Age','Tenure', 'Balance','Estimated Salary']])

X_train_rus[[ 'CreditScore','Age','Tenure', 'Balance','Estimated Salary']]=sc.fit_transform(X_train_rus[[ 'CreditScore','Age','Tenure', 'Balance','Estimated Salary']])
X_test_rus[[ 'CreditScore','Age','Tenure', 'Balance','Estimated Salary']]=sc.fit_transform(X_test_rus[['CreditScore','Age','Tenure', 'Balance','Estimated Salary']])

X_train_ros[[ 'CreditScore','Age','Tenure', 'Balance','Estimated Salary']]=sc.fit_transform(X_train_ros[[ 'CreditScore','Age','Tenure', 'Balance','Estimated Salary']])
X_test_ros[[ 'CreditScore','Age','Tenure', 'Balance','Estimated Salary']]=sc.fit_transform(X_test_ros[['CreditScore','Age','Tenure', 'Balance','Estimated Salary']])
#

from sklearn.svm import SVC

svc=SVC()
svc.fit(X_train, Y_train)
Y_pred=svc.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(Y_test, Y_pred)

print(classification_report(Y_test, Y_pred))

from sklearn.model_selection import GridSearchCV

param_grid={'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf'], 'class_weight': ['balanced']}

grid=GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=2)
grid.fit(X_train, Y_train)

print(grid.best_estimator_)

grid_predictions=grid.predict(X_test)

confusion_matrix(Y_test, grid_predictions)

print(classification_report(Y_test, grid_predictions))

svc_rus=SVC()
svc_rus.fit(X_train_rus, Y_train_rus)
Y_pred_rus=svc_rus.predict(X_test_rus)

confusion_matrix(Y_test_rus, Y_pred_rus)

print(classification_report(Y_test_rus, Y_pred_rus))

param_grid={'C': [0.1, 1, 10,], 'gamma': [1, 0.1, 0.01, 0], 'kernel': ['rbf'], 'class_weight': ['balanced']}

grid_rus=GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=2)
grid_rus.fit(X_train_rus, Y_train_rus)

print(grid_rus.best_estimator_)

grid_predictions_rus=grid_rus.predict(X_test_rus)

confusion_matrix(Y_test_rus, grid_predictions_rus)

print(classification_report(Y_test_rus, grid_predictions_rus))

svc_ros=SVC()
svc_ros.fit(X_train_ros, Y_train_ros)
Y_pred_ros=svc_ros.predict(X_test_ros)

confusion_matrix(Y_test_ros, Y_pred_ros)

print(classification_report(Y_test_ros, Y_pred_ros))

param_grid={'C': [0.1, 1, 10,], 'gamma': [1, 0.1, 0.01, 0], 'kernel': ['rbf'], 'class_weight': ['balanced']}

grid_ros=GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=2)
grid_ros.fit(X_train_ros, Y_train_ros)

print(grid_ros.best_estimator_)

grid_predictions_ros=grid_ros.predict(X_test_ros)

confusion_matrix(Y_test_ros, grid_predictions_ros)

print(classification_report(Y_test_ros, grid_predictions_ros))

print(classification_report(Y_test, Y_pred))

print(classification_report(Y_test, grid_predictions))

print(classification_report(Y_test_rus, Y_pred_rus))

print(classification_report(Y_test_rus, grid_predictions_rus))

print(classification_report(Y_test_ros, Y_pred_ros))

print(classification_report(Y_test_ros, grid_predictions_ros))  

