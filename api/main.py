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
models = [model_1, model_2, model_3, model_4]

# Putting PyTorch models into eval mode
model_1.eval()
model_2.eval()



'''
Gets the prediction of all models. 
Parameters - User_info: array (of size 32)
'''
def get_prediction(user_info):
    # print("User info full object data type:" + str(type(user_info)))
    # print("User info element 0 data type: " + str(type(user_info[0])))
    user_info = [float(i) for i in user_info]
    tensor = torch.tensor(user_info)
    # print("User info element 0 data type after conversion: " + str(type(user_info[0])))
    # print(tensor)
    # print("Running model 1...")
    output1 = model_1.forward(tensor)
    # print("Output 1: " + str(output1))
    # print("Running model 2...")
    output2 = model_2.forward(tensor)
    # print("Output 2: " + str(output2))
    prediction1 = int(output1.round().item())
    # print("Prediction 1: " + str(prediction1))
    prediction2 = int(output2.round().item())
    # print("Prediction 2: " + str(prediction2))
    # print("Running model 3...")
    prediction3 = model_3.rfc.predict(tensor.reshape(1,-1))
    # print("Prediction 3: " + str(prediction3))
    # print("Running model 4...")
    prediction4 = model_4.svm.predict(tensor.reshape(1,-1))
    # print("Prediction 4: " + str(prediction4))
    predictions = [prediction1, prediction2, prediction3, prediction4]
    print("Predictions: " + str(predictions))
    return predictions

def get_test_acc():
    accuracies = []
    for model in models:
        accuracies.append(model.get_test_acc())

    print("Model accuracies: " + str(accuracies))
    return accuracies

# predictions = get_prediction(user_info)

# # Evaluting each model based on the patients data in user_input
# prediction_num=0
# print("Getting predictions...")
# for prediction in predictions: 
#     prediction_num +=1
#     if(prediction == 0):
#         print(f"Model {prediction_num} says...Alzheimers NOT detected :D")
#     else:
#         print(f"Model {prediction_num} says...Alzheimers detected :'(")
        
