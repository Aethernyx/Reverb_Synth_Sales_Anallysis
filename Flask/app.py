from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from catboost import CatBoostRegressor
model = CatBoostRegressor()

app = Flask(__name__)

model.load_model('log_model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    for rendering results on HTML GUI
    '''
    features = [str(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(np.exp(prediction[0]), 2)

    return render_template('index.html', prediction_text='Sale Price should be: ${}'.format(output))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    for direct API calls
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


app.run('127.0.0.1', debug=True)