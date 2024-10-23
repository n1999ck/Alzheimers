import torch
from fnn import FNN
from model_test import MyModel

model1 = FNN(32, 500, 1)
model2 = MyModel()
model1.eval()
model2.eval()

user_info = [72, 0, 0, 2, 33.28973831, 0, 7.890703151, 6.570993383, 7.941403884, 9.878710516, 0, 0, 0, 0, 0, 0, 166, 78, 283.3967969, 92.20006443, 81.92004333, 217.3968725, 11.11477737, 6.30754331, 0, 1, 8.327563008, 0, 1, 0, 0, 1]
#Will be where we pull in user input
def get_user_input():
    return 

def get_prediction(user_info):
    tensor = torch.tensor(user_info)
    output1 = model1.forward(tensor)
    output2 = model2.forward(tensor)
    prediction1 = int(output1.round().item())
    prediction2 = int(output2.round().item())
    predictions = [prediction1, prediction2]
    return predictions

predictions = get_prediction(user_info)
prediction_num=0
for prediction in predictions: 
    prediction_num +=1
    if(prediction == 0):
        print(f"Model {prediction_num} says...Alzheimers NOT detected :D")
    else:
        print(f"Model {prediction_num} says...Alzheimers detected :'(")