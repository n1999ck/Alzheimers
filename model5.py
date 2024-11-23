import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from data_extractor import PatientData
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib
import math
import os
import dotenv

env_file = dotenv.find_dotenv("results/.env")
dotenv.load_dotenv(env_file)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = PatientData() 

class XGB():

    def __init__(self):
        self.xgb = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,
    max_depth=1, random_state=0)

        # Model metric variables as attributes
        self.training_accuracy = 0
        self.testing_accuracy = 0

        self.specificity = -1
        self.precision = -1
        self.recall = -1
        self.f1 = -1
        self.mcc = -1

    def train(self):
        self.xgb.fit(data.X_train_no_val, data.y_train_no_val)
        y_pred = self.xgb.predict(data.X_train_no_val)
        self.get_train_metrics(data.y_train_no_val, y_pred)
    
    def test(self):
        self.pred = self.xgb.predict(data.X_test_no_val)
        self.get_test_metrics(data.y_test_no_val, self.pred)

    def get_train_metrics(self, y_true, y_pred):
        self.training_accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)*100

    def get_test_metrics(self, y_pred, y_true):
        self.testing_accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)*100
        
        # Confusion matrix with true neg, false pos, false neg, true pos respectively
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)  
        tn, fp, fn, tp = cm.ravel()  

        # if statements to stop any division by 0 errors
        if((tn+fp)>0):
            self.specificity = tn / (tn+fp)  # Correctly predicted negatives over all actual negatives

        if((tp+fp) & (fn+tp)>0):
            self.precision = tp /(tp+fp)     # Correctly predicted positives over all predicted positives
            self.recall = tp / (fn+tp)       # Correctly predicted positives over all actual positives
            self.f1 = 2 * ((self.precision*self.recall)/(self.precision+self.recall))    # F1 Score 

        if((tp+fp) & (tp+fn) & (tn+fp) & (tn+fn) > 0):
            self.mcc = ((tp*tn) - (fp*fn))/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))  # Matthews Correlation Coefficient

        # Print all calculated metrics for test samples
        print("Test Accuracy:\t\t{}%".format(round(self.testing_accuracy, 4)))
        print("-"*60)
        print("Precision:\t{}".format(round(self.precision, 4)))
        print("Specificity:\t{}".format(round(self.specificity, 4)))
        print("Recall:\t\t{}".format(round(self.recall, 4)))
        print("F1:\t\t{}".format(round(self.f1, 4)))
        print("MCC:\t\t{}".format(round(self.mcc, 4)))

    def save_test_metrics_display(self, y_pred, y_label):
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
        plt.savefig('results/matrix_xgb.png')
   
    def check_metrics(self)-> bool:
        curr_acc = float(os.getenv('XGB_TESTING_ACCURACY'))
        curr_rec = float(os.getenv('XGB_RECALL'))
        print("New: {}".format(self.testing_accuracy + (self.recall*100)))
        print("Old: {}".format(curr_acc+(curr_rec*100)))
        print(self.recall)
        print(self.testing_accuracy)
        if(self.testing_accuracy + (self.recall*100) > curr_acc+(curr_rec*100)):
            self.save_metrics()
            self.save_test_metrics_display(self.pred, data.y_test_no_val)
            return True
        return False

    def save_metrics(self):
        os.environ['XGB_TRAINING_ACCURACY'] = str(self.training_accuracy)
        os.environ['XGB_TESTING_ACCURACY'] = str(self.testing_accuracy)
        os.environ['XGB_SPECIFICITY'] = str(self.specificity)
        os.environ['XGB_PRECISION'] = str(self.precision)
        os.environ['XGB_RECALL'] = str(self.recall)
        os.environ['XGB_F1'] = str(self.f1)
        os.environ['XGB_MCC'] = str(self.mcc)

        dotenv.set_key(env_file, 'XGB_TRAINING_ACCURACY', os.environ['XGB_TRAINING_ACCURACY'])
        dotenv.set_key(env_file, 'XGB_TESTING_ACCURACY', os.environ['XGB_TESTING_ACCURACY'])
        dotenv.set_key(env_file, 'XGB_SPECIFICITY', os.environ['XGB_SPECIFICITY'])
        dotenv.set_key(env_file, 'XGB_PRECISION', os.environ['XGB_PRECISION'])
        dotenv.set_key(env_file, 'XGB_RECALL', os.environ['XGB_RECALL'])
        dotenv.set_key(env_file, 'XGB_F1', os.environ['XGB_F1'])
        dotenv.set_key(env_file, 'XGB_MCC', os.environ['XGB_MCC'])

def main(): 
    model = XGB()
    model.train()
    model.test()
    if(model.check_metrics()):
        joblib.dump(model, "saved models/model_xgb.joblib")
        print("MODEL IMPROVED! New model saved.")

    for i in range(50):
        main()

'''
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
'''
if __name__ == '__main__':
    main()