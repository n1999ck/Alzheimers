'''
The saved sklearn models are with 323 samples not 322. Trying to get a new saved model of both will fix this. 
    ^Need a better model for it to be overridden

Validation Accuracy in FNN & MLP scripts are not accurate. Training and Testing are. 
    ^Not the biggest concern right now. Will fix eventually.
'''
import torch
import joblib
import os
from model1 import FNN  # Feedforward Neural Network- PyTorch
from model2 import MLP  # Multilayer Perceptron     - PyTorch
from model3 import RFC  # Random Forest Classifier  - Sklearn
from model4 import SVM  # Support Vector Machine    - Sklearn
from api.model5 import XGB  # Gradient Boosting         - Sklearn

# Creation of patient data
user_info = [72, 0, 0, 2, 33.28973831, 0, 7.890703151, 6.570993383, 7.941403884, 9.878710516, 0, 0, 0, 0, 0, 0, 166, 78, 283.3967969, 92.20006443, 81.92004333, 217.3968725, 11.11477737, 6.30754331, 0, 1, 8.327563008, 0, 1, 0, 0, 1]

# Max Absolute Normalization
for i, feature in enumerate(user_info):
    if abs(max(user_info, key=abs)) != 0:
        user_info[i] = feature/(abs(max(user_info, key=abs)))
    else:
        user_info[i] = 0

model_1 = FNN()
model_2 = MLP()
model_3 = RFC()
model_4 = SVM()
model_5 = XGB()

# Passing module classes into callable variables
state_dict_1 = torch.load("./model_fnn.pt", weights_only=True)
state_dict_2 = torch.load("./model_mlp.pt", weights_only=True)

model_1.load_state_dict(state_dict_1)
model_2.load_state_dict(state_dict_2)
model_3 = joblib.load("model_rfc.joblib")
model_4 = joblib.load("model_svm.joblib")
model_5 = joblib.load("model_xgb.joblib")

def get_weights()-> list[4]:
    model_sum = float(os.getenv('FNN_TESTING_ACCURACY')) + \
    float(os.getenv('FNN_RECALL'))*100 + \
    float(os.getenv('MLP_TESTING_ACCURACY')) + \
    float(os.getenv('MLP_RECALL'))*100 + \
    float(os.getenv('RFC_TESTING_ACCURACY')) + \
    float(os.getenv('RFC_RECALL'))*100 + \
    float(os.getenv('SVM_TESTING_ACCURACY')) + \
    float(os.getenv('SVM_RECALL'))*100
    float(os.getenv('XGB_TESTING_ACCURACY')) + \
    float(os.getenv('XGB_RECALL'))*100
    print(model_sum)
    weights = [
        (float(os.getenv('FNN_TESTING_ACCURACY')) + float(os.getenv('FNN_RECALL')))/ model_sum,
        (float(os.getenv('MLP_TESTING_ACCURACY')) + float(os.getenv('MLP_RECALL')))/ model_sum,
        (float(os.getenv('RFC_TESTING_ACCURACY')) + float(os.getenv('RFC_RECALL')))/ model_sum,
        (float(os.getenv('SVM_TESTING_ACCURACY')) + float(os.getenv('SVM_RECALL')))/ model_sum,
        (float(os.getenv('XGB_TESTING_ACCURACY')) + float(os.getenv('XGB_RECALL')))/ model_sum
    ]
    return weights

'''
Gets the prediction of all models. 
Parameters - User_info: array (of size 32)
'''
def poll_predictions(user_info):
    tensor = torch.tensor(user_info)
    output1 = model_1.forward(tensor)
    output2 = model_2.forward(tensor)
    prediction1 = int(output1.round().item())
    prediction2 = int(output2.round().item())
    prediction3 = model_3.rfc.predict(tensor.reshape(1,-1))
    prediction4 = model_4.bgc.predict(tensor.reshape(1,-1))
    prediction5 = model_5.xgb.predict(tensor.reshape(1,-1))
    predictions = [prediction1, prediction2, prediction3, prediction4, prediction5]
    return predictions

# This is the big daddy of them all
def meta_classifier(predictions, prediction_weights):
    prediction_num=0
    census_no = 0
    census_yes = 0
    # Evaluting each model based on the patients data in user_input
    for prediction in predictions: 
        if(prediction == 0):
            census_no += prediction_weights[prediction_num]
            print(f"Model {prediction_num+1} says...Alzheimers NOT detected :D")
        else:
            census_yes += prediction_weights[prediction_num] 
            print(f"Model {prediction_num+1} says...Alzheimers detected :'(")
        prediction_num+=1
    print("="*60)
    if(census_no > census_yes):
        print(f"Meta classifier says...Alzheimers NOT detected :D")
    else:
        print(f"Meta classifier says...Alzheimers detected :'(")

prediction = meta_classifier(poll_predictions(user_info), get_weights())



