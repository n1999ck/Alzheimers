import torch
from model11 import FNN


model1 = FNN()

model1.eval()

user_info = ['84', '1', '2', '0', '45', '1', '20', '0', '0', '0', '1', '1', '1', '1', '1', '1', '166', '78', '283', '92', '81', '217', '11', '0', '1', '1', '10', '1', '1', '1', '1', '1']


#Will be where we pull in user input
def get_user_input():
    return 

def get_prediction(user_info):
    print("User info full object data type:" + str(type(user_info)))
    print("User info element 0 data type: " + str(type(user_info[0])))
    user_info = [float(i) for i in user_info]
    tensor = torch.tensor(user_info)
    output1 = model1.forward(tensor)
    prediction1 = int(output1.round().item())
    predictions = [prediction1]
    print("Predictions: " + str(predictions))
    if(predictions == 0):
        return(f"Model says...Alzheimers NOT detected :D")
    else:
        return(f"Model says...Alzheimers detected :'(")
    #return predictions

predictions = get_prediction(user_info)
prediction_num=0
for prediction in predictions: 
    prediction_num +=1
    if(prediction == 0):
        print(f"Model {prediction_num} says...Alzheimers NOT detected :D")
    else:
        print(f"Model {prediction_num} says...Alzheimers detected :'(")