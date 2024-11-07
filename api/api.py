import time
import torch
from flask import Flask, request, jsonify
from model11 import FNN
from main import get_prediction

app = Flask(__name__, static_folder='../build', static_url_path='/')

model1 = FNN()
model1.eval()

# def get_prediction(user_info):
#     print("Getting prediction")
#     print(type(user_info))
#     print(type(user_info[0]))
#     user_info = [float(i) for i in user_info]
#     print(type(user_info)[0])
#     tensor = torch.tensor(user_info)
#     output1 = model1.forward(tensor)
#     print("Output: " + str(output1))
#     prediction1 = int(output1.round().item())
#     print("Prediction: " + str(prediction1))
#     predictions = [prediction1]
#     print("Predictions: " + str(predictions))
#     return predictions

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
    print("User info:")
    print(user_info)
    prediction = get_prediction(user_info);
    return jsonify(prediction)