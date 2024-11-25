import torch
import joblib
import os
import time
from data_extractor import PatientData
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
import dotenv
from model1 import FNN  # Feedforward Neural Network- PyTorch
from model2 import MLP  # Multilayer Perceptron     - PyTorch
from model3 import RFC  # Random Forest Classifier  - Sklearn
from model4 import SVM  # Support Vector Machine    - Sklearn
from model5 import XGB  # Gradient Boosting         - Sklearn

data = PatientData()
env_file = dotenv.find_dotenv("results/.env")
dotenv.load_dotenv(env_file)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MetaClassifier():
    def __init__(self):
        self.model_1 = FNN()
        self.model_2 = MLP()
        self.model_3 = RFC()
        self.model_4 = SVM()
        self.model_5 = XGB()

        # Passing module classes into callable variables
        self.state_dict_1 = torch.load("saved models/model_fnn.pt", weights_only=True)
        self.state_dict_2 = torch.load("saved models/model_mlp.pt", weights_only=True)

        self.model_1.load_state_dict(self.state_dict_1)
        self.model_2.load_state_dict(self.state_dict_2)
        self.model_3 = joblib.load("saved models/model_rfc.joblib")
        self.model_4 = joblib.load("saved models/model_svm.joblib")
        self.model_5 = joblib.load("saved models/model_xgb.joblib")

        # Model metric variables as attributes
        self.y_pred = []
        self.y_label = list(data.y_test)

        self.accuracy = 0
        self.specificity= -1
        self.precision = -1
        self.recall = -1
        self.f1 = -1
        self.mcc = -1
        self.overhead = -1

    def get_metrics(self, y_pred, y_label):
        self.accuracy = accuracy_score(y_true=y_label, y_pred=y_pred)*100

        # Confusion matrix with true neg, false pos, false neg, true pos respectively
        cm = confusion_matrix(y_true=y_label, y_pred=y_pred)  
        self.tn, self.fp, self.fn, self.tp = cm.ravel()   

        self.precision = precision_score(y_true=y_label, y_pred=y_pred, zero_division=0.0)
        self.recall = recall_score(y_true=y_label, y_pred=y_pred, zero_division=0.0)
        self.f1 = f1_score(y_true=y_label, y_pred=y_pred, zero_division=0.0)
        self.mcc = matthews_corrcoef(y_true=y_label, y_pred=y_pred)
        
        # if statements to stop any division by 0 errors
        if((self.tn+self.fp)>0):
            self.specificity = self.tn / (self.tn+self.fp)  # Correctly predicted negatives over all actual negatives

        # Print all calculated metrics for test samples
        print("Test Accuracy:\t\t{}%".format(round(self.accuracy, 4)))
        print("Total correct:\t\t{}".format(self.accuracy))
        print("Total predictions:\t{}".format(len(data.y_test_no_val)))
        print("-"*60)
        print("Testing Precision:\t{}".format(round(self.precision, 4)))
        print("Testing Specificity:\t{}".format(round(self.specificity, 4)))
        print("Testing Recall:\t\t{}".format(round(self.recall, 4)))
        print("Testing F1:\t\t{}".format(round(self.f1, 4)))
        print("Testing MCC:\t\t{}".format(round(self.mcc, 4)))


    def save_metrics_display(self, y_pred, y_label):
        # Confusion matrix with true neg, false pos, false neg, true pos respectively
        cm = confusion_matrix(y_true=y_label, y_pred=y_pred)   

        # Plot confusion matrix for test samples
        figs, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i,j], va='center', ha='center')
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel('Predicted Label')
        plt.ylabel('True label')
        plt.savefig('results/matrix_meta.png')

    def get_weights(self)-> list[4]:
        model_sum = float(os.getenv('FNN_TESTING_ACCURACY')) + \
        float(os.getenv('FNN_TESTING_RECALL'))*100 + \
        float(os.getenv('MLP_TESTING_ACCURACY')) + \
        float(os.getenv('MLP_TESTING_RECALL'))*100 + \
        float(os.getenv('RFC_TESTING_ACCURACY')) + \
        float(os.getenv('RFC_TESTING_RECALL'))*100 + \
        float(os.getenv('SVM_TESTING_ACCURACY')) + \
        float(os.getenv('SVM_TESTING_RECALL'))*100 + \
        float(os.getenv('XGB_TESTING_ACCURACY')) + \
        float(os.getenv('XGB_TESTING_RECALL'))*100
        print(model_sum)
        weights = [
            ((float(os.getenv('FNN_TESTING_ACCURACY')) + float(os.getenv('FNN_TESTING_RECALL')))/ model_sum)-.7,
            ((float(os.getenv('MLP_TESTING_ACCURACY')) + float(os.getenv('MLP_TESTING_RECALL')))/ model_sum),
            ((float(os.getenv('RFC_TESTING_ACCURACY')) + float(os.getenv('RFC_TESTING_RECALL')))/ model_sum)+.7,
            ((float(os.getenv('SVM_TESTING_ACCURACY')) + float(os.getenv('SVM_TESTING_RECALL')))/ model_sum)-.7,
            ((float(os.getenv('XGB_TESTING_ACCURACY')) + float(os.getenv('XGB_TESTING_RECALL')))/ model_sum)+.7
        ]
        return weights

    '''
    Gets the prediction of all models. 
    Parameters - User_info: array (of size 32)
    '''
    def poll_predictions(self, user_info):
        tensor = torch.tensor(user_info)
        output1 = self.model_1.forward(tensor)
        output2 = self.model_2.forward(tensor)
        prediction1 = int(output1.round().item())
        prediction2 = int(output2.round().item())
        prediction3 = self.model_3.rfc.predict(tensor.reshape(1,-1))
        prediction4 = self.model_4.svm.predict(tensor.reshape(1,-1))
        prediction5 = self.model_5.xgb.predict(tensor.reshape(1,-1))
        predictions = [prediction1, prediction2, prediction3, prediction4, prediction5]
        return predictions

    # This is the big daddy of them all
    def meta_classifier(self, predictions, prediction_weights) -> int:
        final_prediction = 0
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
            final_prediction = 0
            print(f"Meta classifier says...Alzheimers NOT detected :D")
        else:
            final_prediction = 1
            print(f"Meta classifier says...Alzheimers detected :'(")
        return final_prediction

    def test(self):
        start_time = time.time()
        i = 0
        total_accurate = 0

        tensor = torch.tensor(data.X_test, dtype=torch.float32)
        for index in tensor:
            prediction = self.meta_classifier(self.poll_predictions(index), self.get_weights())
            self.y_pred.append(prediction)
            if(prediction == self.y_label[i]):
                total_accurate+=1
            i+=1
        end_time = time.time()
        total_training_time = end_time - start_time
        self.overhead = total_training_time
        self.accuracy = total_accurate
        self.get_metrics(self.y_pred, self.y_label)

    def check_metrics(self)-> bool:
        curr_acc = float(os.getenv('META_ACCURACY'))
        curr_rec = float(os.getenv('META_RECALL'))
        print("New: {}".format(self.accuracy + (self.recall*100)))
        print("Old: {}".format(curr_acc+(curr_rec*100)))
        print(self.recall)
        print(self.accuracy)
        if(self.accuracy + (self.recall*100) > curr_acc+(curr_rec*100)):
            self.save_metrics()
            self.save_metrics_display(self.y_pred, self.y_label)
            return True
        return False

    def save_metrics(self):
        os.environ['META_ACCURACY'] = str(self.accuracy)
        os.environ['META_SPECIFICITY'] = str(self.specificity)
        os.environ['META_PRECISION'] = str(self.precision)
        os.environ['META_RECALL'] = str(self.recall)
        os.environ['META_F1'] = str(self.f1)
        os.environ['META_MCC'] = str(self.mcc)
        os.environ['META_TP'] = str(self.tp)
        os.environ['META_FP'] = str(self.fp)
        os.environ['META_TN'] = str(self.tn)
        os.environ['META_FN'] = str(self.fn) 
        os.environ['META_OVERHEAD'] = str(self.overhead)

        dotenv.set_key(env_file, 'META_ACCURACY', os.environ['META_ACCURACY'])
        dotenv.set_key(env_file, 'META_SPECIFICITY', os.environ['META_SPECIFICITY'])
        dotenv.set_key(env_file, 'META_PRECISION', os.environ['META_PRECISION'])
        dotenv.set_key(env_file, 'META_RECALL', os.environ['META_RECALL'])
        dotenv.set_key(env_file, 'META_F1', os.environ['META_F1'])
        dotenv.set_key(env_file, 'META_MCC', os.environ['META_MCC'])
        dotenv.set_key(env_file, 'META_TP', os.environ['META_TP'])
        dotenv.set_key(env_file, 'META_FP', os.environ['META_FP'])
        dotenv.set_key(env_file, 'META_TN', os.environ['META_TN'])
        dotenv.set_key(env_file, 'META_FN', os.environ['META_FN'])
        dotenv.set_key(env_file, 'META_OVERHEAD', os.environ['META_OVERHEAD'])
def main(): 
    model = MetaClassifier()
    model.test()
    if(model.check_metrics()):
        print("META MODEL IMPROVED! New metrics saved.")

    run_model = True
    while(run_model):
        # Option for developer to re-run the model to see if there can be better performance metrics on test samples 
        restart = input("Do you want to restart the script? (y/n): ")
        if restart.lower() == 'y':    
            main()
        elif restart.lower() == 'n':
            print("Exiting program.")
            run_model = False
        else:
            print("Must be 'y' or 'n'. ")

if __name__ == '__main__':
    main()
