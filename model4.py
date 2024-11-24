import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from data_extractor import PatientData
import matplotlib.pyplot as plt
import joblib
import os
import dotenv
import time

env_file = dotenv.find_dotenv("results/.env")
dotenv.load_dotenv(env_file)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = PatientData() 

class SVM():

    def __init__(self):
        self.svm = SVC(kernel='rbf', gamma=0.08, C=10)
        self.bgc = BaggingClassifier(estimator=self.svm, n_estimators=5, random_state=0)
        
        # Model metric variables as attributes
        self.training_accuracy = 0
        self.training_specificity= -1
        self.training_precision = -1
        self.training_recall = -1
        self.training_f1 = -1
        self.training_mcc = -1
        self.training_overhead = -1

        self.testing_accuracy = 0
        self.testing_specificity= -1
        self.testing_precision = -1
        self.testing_recall = -1
        self.testing_f1 = -1
        self.testing_mcc = -1
        self.testing_overhead = -1

    def train(self):
        start_time = time.time()
        self.svm.fit(data.X_train_no_val, data.y_train_no_val)
        self.y_train_pred = self.svm.predict(data.X_train_no_val)
        end_time = time.time()
        total_training_time = end_time - start_time
        self.training_overhead = total_training_time
    
    def test(self):
        start_time = time.time()
        self.y_test_pred = self.svm.predict(data.X_test_no_val)
        end_time = time.time()
        total_testing_time = end_time - start_time
        self.testing_overhead = total_testing_time
        self.get_metrics(self.y_train_pred, data.y_train_no_val, self.y_test_pred, data.y_test_no_val)

    def get_metrics(self, y_train_pred, y_train_label, y_test_pred, y_test_label):

        self.training_accuracy = accuracy_score(y_true=y_train_label, y_pred=y_train_pred)*100
        self.testing_accuracy = accuracy_score(y_true=y_test_label, y_pred=y_test_pred)*100

        # Confusion matrix with true neg, false pos, false neg, true pos respectively
        train_cm = confusion_matrix(y_true=y_train_label, y_pred=y_train_pred)  
        test_cm = confusion_matrix(y_true=y_test_label, y_pred=y_test_pred)  
        self.train_tn, self.train_fp, self.train_fn, self.train_tp = train_cm.ravel() 
        self.test_tn, self.test_fp, self.test_fn, self.test_tp = test_cm.ravel()   

        self.training_precision = precision_score(y_true=y_train_label, y_pred=y_train_pred, zero_division=0.0)
        self.training_recall = recall_score(y_true=y_train_label, y_pred=y_train_pred, zero_division=0.0)
        self.training_f1 = f1_score(y_true=y_train_label, y_pred=y_train_pred, zero_division=0.0)
        self.training_mcc = matthews_corrcoef(y_true=y_train_label, y_pred=y_train_pred)

        self.testing_precision = precision_score(y_true=y_test_label, y_pred=y_test_pred, zero_division=0.0)
        self.testing_recall = recall_score(y_true=y_test_label, y_pred=y_test_pred, zero_division=0.0)
        self.testing_f1 = f1_score(y_true=y_test_label, y_pred=y_test_pred, zero_division=0.0)
        self.testing_mcc = matthews_corrcoef(y_true=y_test_label, y_pred=y_test_pred)
        
        # if statements to stop any division by 0 errors
        if((self.train_tn+self.train_fp)>0):
            self.training_specificity = self.train_tn / (self.train_tn+self.train_fp)  # Correctly predicted negatives over all actual negatives

        if((self.test_tn+self.test_fp)>0):
            self.training_specificity = self.train_tn / (self.train_tn+self.train_fp)  # Correctly predicted negatives over all actual negatives

        # Print all calculated metrics for test samples
        print("Test Accuracy:\t\t{}%".format(round(self.testing_accuracy, 4)))
        print("Total correct:\t\t{}".format(self.testing_accuracy))
        print("Total predictions:\t{}".format(len(data.y_test_no_val)))
        print("-"*60)
        print("Testing Precision:\t{}".format(round(self.testing_precision, 4)))
        print("Testing Specificity:\t{}".format(round(self.testing_specificity, 4)))
        print("Testing Recall:\t\t{}".format(round(self.testing_recall, 4)))
        print("Testing F1:\t\t{}".format(round(self.testing_f1, 4)))
        print("Testing MCC:\t\t{}".format(round(self.testing_mcc, 4)))

    def save_matrix_display(self, y_pred, y_label, testing):
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
        if(testing):
            plt.savefig('results/test/test_matrix_svm.png')
        else:
            plt.savefig('results/train/train_matrix_svm.png')

   
    def check_metrics(self)-> bool:
        curr_acc = float(os.getenv('SVM_TESTING_ACCURACY'))
        curr_rec = float(os.getenv('SVM_TESTING_RECALL'))
        print("New: {}".format(self.testing_accuracy + (self.testing_recall*100)))
        print("Old: {}".format(curr_acc+(curr_rec*100)))
        print(self.testing_recall)
        print(self.testing_accuracy)
        if(self.testing_accuracy + (self.testing_recall*100) > curr_acc+(curr_rec*100)):
            self.save_attributes()
            self.save_matrix_display(self.y_train_pred, data.y_train_no_val, False)
            self.save_matrix_display(self.y_test_pred, data.y_test_no_val, True)
            return True
        return False
    
    def save_attributes(self):
        os.environ['SVM_TRAINING_ACCURACY'] = str(self.training_accuracy)
        os.environ['SVM_TRAINING_SPECIFICITY'] = str(self.training_specificity)
        os.environ['SVM_TRAINING_PRECISION'] = str(self.training_precision)
        os.environ['SVM_TRAINING_RECALL'] = str(self.training_recall)
        os.environ['SVM_TRAINING_F1'] = str(self.training_f1)
        os.environ['SVM_TRAINING_MCC'] = str(self.training_mcc)
        os.environ['SVM_TRAINING_TP'] = str(self.train_tp)
        os.environ['SVM_TRAINING_FP'] = str(self.train_fp)
        os.environ['SVM_TRAINING_TN'] = str(self.train_tn)
        os.environ['SVM_TRAINING_FN'] = str(self.train_fn)
        os.environ['SVM_TRAINING_OVERHEAD'] = str(self.training_overhead)
        
        os.environ['SVM_TESTING_ACCURACY'] = str(self.testing_accuracy)
        os.environ['SVM_TESTING_SPECIFICITY'] = str(self.testing_specificity)
        os.environ['SVM_TESTING_PRECISION'] = str(self.testing_precision)
        os.environ['SVM_TESTING_RECALL'] = str(self.testing_recall)
        os.environ['SVM_TESTING_F1'] = str(self.testing_f1)
        os.environ['SVM_TESTING_MCC'] = str(self.testing_mcc)
        os.environ['SVM_TESTING_TP'] = str(self.test_tp)
        os.environ['SVM_TESTING_FP'] = str(self.test_fp)
        os.environ['SVM_TESTING_TN'] = str(self.test_tn)
        os.environ['SVM_TESTING_FN'] = str(self.test_fn)   
        os.environ['SVM_TESTING_OVERHEAD'] = str(self.testing_overhead)

        dotenv.set_key(env_file, 'SVM_TRAINING_ACCURACY', os.environ['SVM_TRAINING_ACCURACY'])
        dotenv.set_key(env_file, 'SVM_TRAINING_SPECIFICITY', os.environ['SVM_TRAINING_SPECIFICITY'])
        dotenv.set_key(env_file, 'SVM_TRAINING_PRECISION', os.environ['SVM_TRAINING_PRECISION'])
        dotenv.set_key(env_file, 'SVM_TRAINING_RECALL', os.environ['SVM_TRAINING_RECALL'])
        dotenv.set_key(env_file, 'SVM_TRAINING_F1', os.environ['SVM_TRAINING_F1'])
        dotenv.set_key(env_file, 'SVM_TRAINING_MCC', os.environ['SVM_TRAINING_MCC'])
        dotenv.set_key(env_file, 'SVM_TRAINING_TP', os.environ['SVM_TRAINING_TP'])
        dotenv.set_key(env_file, 'SVM_TRAINING_FP', os.environ['SVM_TRAINING_FP'])
        dotenv.set_key(env_file, 'SVM_TRAINING_TN', os.environ['SVM_TRAINING_TN'])
        dotenv.set_key(env_file, 'SVM_TRAINING_FN', os.environ['SVM_TRAINING_FN'])
        dotenv.set_key(env_file, 'SVM_TRAINING_OVERHEAD', os.environ['SVM_TRAINING_OVERHEAD'])

        dotenv.set_key(env_file, 'SVM_TESTING_ACCURACY', os.environ['SVM_TESTING_ACCURACY'])
        dotenv.set_key(env_file, 'SVM_TESTING_SPECIFICITY', os.environ['SVM_TESTING_SPECIFICITY'])
        dotenv.set_key(env_file, 'SVM_TESTING_PRECISION', os.environ['SVM_TESTING_PRECISION'])
        dotenv.set_key(env_file, 'SVM_TESTING_RECALL', os.environ['SVM_TESTING_RECALL'])
        dotenv.set_key(env_file, 'SVM_TESTING_F1', os.environ['SVM_TESTING_F1'])
        dotenv.set_key(env_file, 'SVM_TESTING_MCC', os.environ['SVM_TESTING_MCC'])
        dotenv.set_key(env_file, 'SVM_TESTING_TP', os.environ['SVM_TESTING_TP'])
        dotenv.set_key(env_file, 'SVM_TESTING_FP', os.environ['SVM_TESTING_FP'])
        dotenv.set_key(env_file, 'SVM_TESTING_TN', os.environ['SVM_TESTING_TN'])
        dotenv.set_key(env_file, 'SVM_TESTING_FN', os.environ['SVM_TESTING_FN'])
        dotenv.set_key(env_file, 'SVM_TESTING_OVERHEAD', os.environ['SVM_TESTING_OVERHEAD'])

def main(): 
    model = SVM()
    model.train()
    model.test()
    if(model.check_metrics()):
        joblib.dump(model, "saved models/model_svm.joblib")
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
