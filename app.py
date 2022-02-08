import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = pickle.load(open('model_GradientBoostingClassifier.pkl', 'rb'))

@app.route('/')
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    output = model.predict(final_features)
    
    if output == 1:
        return render_template('index.html', prediction_text = 'Client will get a stroke')
    else:
        return render_template('index.html', prediction_text = 'Client will not get a stroke')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.get_json(force = True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)