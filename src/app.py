#Librerias
import pandas as pd
import numpy as np
from pickle import dump

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

#Obtenemos los datos

x_train_with_outliers = pd.read_csv("../data/processed/x_train_sel_with_outliers.csv")
x_test_with_outliers = pd.read_csv("../data/processed/x_test_sel_with_outliers.csv")
x_train_without_outliers = pd.read_csv("../data/processed/x_train_sel_without_outliers.csv")
x_test_without_outliers = pd.read_csv("../data/processed/x_test_sel_without_outliers.csv")
y_train = pd.read_csv("../data/processed/y_train.csv")
y_test = pd.read_csv("../data/processed/y_test.csv")

def trainingModel(x, y, device = "cuda"):
    model = XGBClassifier(tree_method = "hist", device = device, random_state = 42)
    model.fit(x, y)
    return model

pre_model = trainingModel(x_train_with_outliers, y_train)

hyperparameters = {"booster" : ["gbtree", "gblinear", "dart"], "eta" : np.linspace(0,1,3), "gamma" : np.random.randint(100, size = 3), "max_depth" : np.random.randint(8, size = 3), "enable_categorical" : [True, False], "n_estimators" : np.random.randint(1000, size = 3), "random_state" : np.random.randint(100, size = 3), "device" : ["cuda"]}

#Optimizamos los hiperparametros con RandomSearch porque con el GridSearch se me van los tiempos a dias.

randgrid = RandomizedSearchCV(pre_model, hyperparameters, n_iter = 10, cv = 10, verbose = 1, n_jobs = -4)

randgrid.fit(x_train_with_outliers, y_train)

clf = randgrid.best_estimator_

dump(clf, open(f"../models/boosting_tree.sav", "wb"))

y_test_predict = clf.predict(x_test_with_outliers)

score = accuracy_score(y_test, y_test_predict)

print (score)