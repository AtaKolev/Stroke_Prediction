from matplotlib.pyplot import cla
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pickle

lr = LogisticRegression()
rfc = RandomForestClassifier(n_estimators=1000)
gbc = GradientBoostingClassifier()

model_accuracy = {lr: 0,
                rfc : 0,
                gbc : 0}
models = list(model_accuracy.keys())

df_mean = pd.read_csv('mean_bmi_ready_for_model.csv')
df_nona = pd.read_csv('no_nas_ready_for_model.csv') 
dfs = [df_mean, df_nona]

for model in models:
    acc_dict = {'mean' : 0,
                'zero' : 0}
    for df, key in zip(dfs, list(acc_dict.keys())):
        acc_functions = {'f1_score': 0,
                        'accuracy_score' : 0}
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc_functions.update({'f1_score' : f1_score(y_test, y_pred)})
        acc_functions.update({'accuracy_score' : accuracy_score(y_test, y_pred)})
        #print("Report for " + str(model) + " " + str(df) + "\n" + classification_report(y_test, y_pred))
        acc_dict.update({key:acc_functions})
    model_accuracy.update({model:acc_dict})
    pickle.dump(model, open('model_'+str(model)[:-2]+".pkl", 'wb'))

for model in models:
    print(model_accuracy[model])

    








