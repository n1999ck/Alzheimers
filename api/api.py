import time
import torch
from flask import Flask, request, jsonify
from model11 import FNN
from model1 import FNN  # Feedforward Neural Network- PyTorch
from model2 import MLP  # Multilayer Perceptron     - PyTorch
from model3 import RFC  # Random Forest Classifier  - Sklearn
from model4 import SVM  # Support Vector Machine    - Sklearn
from model5 import XGB 
from main import get_prediction, get_test_acc

app = Flask(__name__, static_folder='../build', static_url_path='/')

print("App started...")

@app.errorhandler(404)
def not_found(e):
    return app.send_static_file('index.html')


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/time')
def get_current_time():
    return {'time': time.time()}

@app.route('/api/predict', methods=['POST'])
def predict():
    user_info = []
    # print("Request received")
    # print(type(request))
    # print("Request data: " + str(request.data) + "\n")
    # print("Request json:" + str(request.json) + "\n")
    for key, value in request.json.items():
        user_info.append(value)
    # print("User info:")
    # print(user_info)
    prediction = get_prediction(user_info);
    # print("Prediction results: " + str(prediction))
    # # print(str(type(prediction)))
    # print(prediction[2])
    prediction[2] = int(prediction[2][0])
    # print(prediction[3])
    prediction[3] = int(prediction[3][0])
    # print("Prediction results after conversion: " + str(prediction))

    return jsonify(prediction)

@app.route('/api/accuracies', methods=['GET'])
def get_accuracies():
    print("Getting accuracies...")
    accuracies = get_test_acc()
    print("Accuracies: " + str(accuracies))
    return jsonify(accuracies)