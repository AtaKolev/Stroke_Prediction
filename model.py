from random import Random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

lr = LinearRegression()
rfc = RandomForestClassifier(n_estimators=1000)
gbc = GradientBoostingClassifier()

models = [lr, rfc, gbc]
model_accuracy = {lr: 0,
                rfc : 0,
                gbc : 0}

for model in models:
    pass