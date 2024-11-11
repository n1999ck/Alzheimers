'''
TODO: Save models
TODO: Create meta classifier
TODO: Extract 'LOADING DATA' portion to its own file
'''

import torch
from model1 import FNN  # Feedforward Neural Network- PyTorch
from model2 import MLP  # Multilayer Perceptron     - PyTorch
from model3 import RFC  # Random Forest Classifier  - Sklearn
from model4 import SVM  # Support Vector Classifier - Sklearn

# Creation of patient data
user_info = [72, 0, 0, 2, 33.28973831, 0, 7.890703151, 6.570993383, 7.941403884, 9.878710516, 0, 0, 0, 0, 0, 0, 166, 78, 283.3967969, 92.20006443, 81.92004333, 217.3968725, 11.11477737, 6.30754331, 0, 1, 8.327563008, 0, 1, 0, 0, 1]

# Max Absolute Normalization
for i, feature in enumerate(user_info):
    user_info[i] = feature/(abs(max(user_info, key=abs)))

# Passing module classes into callable variables
model_1 = FNN()
model_2 = MLP()
model_3 = RFC()
model_4 = SVM()

# Putting PyTorch models into eval mode
model_1.eval()
model_2.eval()

'''
Gets the prediction of all models. 
Parameters - User_info: array (of size 32)
'''
def get_prediction(user_info):
    tensor = torch.tensor(user_info)
    output1 = model_1.forward(tensor)
    output2 = model_2.forward(tensor)
    prediction1 = int(output1.round().item())
    prediction2 = int(output2.round().item())
    prediction3 = model_3.rfc.predict(tensor.reshape(1,-1))
    prediction4 = model_4.svm.predict(tensor.reshape(1,-1))
    predictions = [prediction1, prediction2, prediction3, prediction4]
    return predictions

predictions = get_prediction(user_info)

# Evaluting each model based on the patients data in user_input
prediction_num=0
for prediction in predictions: 
    prediction_num +=1
    if(prediction == 0):
        print(f"Model {prediction_num} says...Alzheimers NOT detected :D")
    else:
        print(f"Model {prediction_num} says...Alzheimers detected :'(")