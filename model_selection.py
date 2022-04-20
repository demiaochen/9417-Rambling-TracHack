import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import BernoulliNB
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

################################################################################
##                        Import Preprocessed Data                            ##
################################################################################
df = pd.read_csv("preprocessedData.csv")
X= df.drop('ebb_eligible', axis=1)
y= df['ebb_eligible']
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


################################################################################
##                               Models                                       ##
################################################################################
# record predictions of each model
model_predictions = []


# Baseline model Dummy
baseline_model = DummyClassifier(strategy='constant', constant=1)
baseline_model.fit(X_train, y_train)
model_predictions.append(baseline_model.predict(X_test))


# LinearSVC
param_grid = {
    'penalty': ['l1', 'l2'],
    'dual': [False], # Prefer dual=False when n_samples > n_features
    'C': [0.1, 0.2, 0.5]
}
linearSVC = GridSearchCV(LinearSVC(), param_grid, cv=5, verbose=False)
linearSVC.fit(X_train, y_train)
print("LinearSVC best params:", linearSVC.best_params_)
model_predictions.append(linearSVC.predict(X_test))


# KNN
param_grid = {
    'n_neighbors':[20],
    'leaf_size': [20],
    'weights': ['uniform']
}
knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=2, verbose=False)
knn.fit(X_train, y_train)
print("KNN best params:", knn.best_params_)
model_predictions.append(knn.predict(X_test))


# BernoulliNB
bernoulliNB = BernoulliNB()
bernoulliNB.fit(X_train, y_train)
model_predictions.append(bernoulliNB.predict(X_test))


# Catboost
catboost = CatBoostClassifier(verbose=False,random_state=0)
catboost.fit(X_train, y_train,cat_features=[],eval_set=(X_test, y_test))
model_predictions.append(catboost.predict(X_test))


# MLP
mlp = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(5, 2), \
                    random_state=1, max_iter = 10000)
mlp.fit(X_train, y_train)
model_predictions.append(mlp.predict(X_test))


# Ramdom Forest
grid = {'n_estimators': [95]}
rf = GridSearchCV(RandomForestClassifier(random_state=23, max_depth=3), grid, cv=2, verbose=False)
rf.fit(X_train, y_train)
print("RF best params:", rf.best_params_)
model_predictions.append(rf.predict(X_test))


# ADAboost
grid = {'learning_rate': np.linspace(0.01,1.5,100), 'algorithm': ['SAMME.R']}
ada = GridSearchCV(AdaBoostClassifier(), grid, cv=2, n_jobs=-1, verbose=False)
ada.fit(X_train, y_train)
print("ADAboost best params:", ada.best_params_)
model_predictions.append(ada.predict(X_test))


# Logistic Regression
grid = {'C': np.logspace(-2,2,100), 'penalty': ['l1','l2'], 'solver': ['saga', 'liblinear']}
logreg = GridSearchCV(LogisticRegression(), grid, cv=2, n_jobs=-1, verbose=False)
model = logreg.fit(X_train, y_train)
print("Logistic Regression best params:", logreg.best_params_)
model_predictions.append(logreg.predict(X_test))


# Gradient Boosting
gb = GradientBoostingClassifier(random_state=0)
gb.fit(X_train, y_train)
model_predictions.append(gb.predict(X_test))


# Extra Trees
et =  ExtraTreesClassifier(random_state=0)
et.fit(X_train, y_train)
model_predictions.append(et.predict(X_test))


# XGBoost
xgbc = XGBClassifier(random_state=0)
xgbc.fit(X_train, y_train)
model_predictions.append(xgbc.predict(X_test))


# LightGBM
lgbmc=LGBMClassifier(random_state=0)
lgbmc.fit(X_train, y_train)
model_predictions.append(lgbmc.predict(X_test))

################################################################################
##                            Evaluations                                     ##
################################################################################
accuracy = []
f1_scores = []
model_names = [
               'Dummy', 'LinearSVC', 'KNN', 'BernoulliNB', \
               'Catboost', 'MLP', 'Ramdom Forest', \
               'Ada Boost', 'Logistic Regression',\
               'Gradient Boosting', 'Extre Trees', 'XGBoost', 'LightGBM'
              ]

# use test dataset to do evaluation
for y_test_pred in model_predictions:    
    test_accuracy_score = round(accuracy_score(y_test, y_test_pred), 4)
    accuracy.append(test_accuracy_score)
    test_f1_score = round(f1_score(y_test_pred, y_test, average='binary'), 4)
    f1_scores.append(test_f1_score)

# store results into a dataframe
result_df = pd.DataFrame({'Accuracy':accuracy, 'F1 score':f1_scores}, index=model_names)
print(result_df)
