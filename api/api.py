import time
import torch
from flask import Flask, request, jsonify
from model11 import FNN

app = Flask(__name__, static_folder='../build', static_url_path='/')

model1 = FNN()
model1.eval()

def get_prediction(user_info):
    tensor = torch.tensor(user_info)
    print(tensor)
    output1 = model1.forward(tensor)
    print(output1)
    prediction1 = int(output1.round().item())
    print(prediction1)
    predictions = [prediction1]
    print(predictions)
    return predictions

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
    return