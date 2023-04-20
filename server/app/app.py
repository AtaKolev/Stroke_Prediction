import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import constants
import os

app = Flask(__name__)

def load_model(path = constants.model_name):

    if os.path.exists(path):
        model = pickle.load(path)
    else:
        raise FileNotFoundError('Specified model does not exist.')
    
    return model


def predict():
    pass
    

if __name__ == "__main__":
    app.run(debug=True)