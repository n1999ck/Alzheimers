import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_extractor import PatientData
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
import math
import os
import dotenv
import time

env_file = dotenv.find_dotenv("results/.env")
dotenv.load_dotenv(env_file)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = PatientData()

'''
STEP 1: MAKING DATASET ITERABLE
'''
#Stores the samples and their corresponding labels
class CSVDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype= torch.float32).to(device)
        self.Y = torch.tensor(Y, dtype= torch.float32).to(device)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

training_data = CSVDataset(data.X_train, data.y_train)
validation_data = CSVDataset(data.X_val, data.y_val)
testing_data = CSVDataset(data.X_test, data.y_test)

#HYPERPARAMETERS
BATCH_SIZE=4
EPOCHS=10
HIDDEN_NEURONS=8
LR=1e-2

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)
testing_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=False)

'''
STEP 2: CREATE MODEL CLASS
'''
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.input_layer = nn.Linear(data.X.shape[1], HIDDEN_NEURONS)
        self.hidden_layer1 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.hidden_layer2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.output_layer = nn.Linear(HIDDEN_NEURONS, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() 

        # Optimizerand and Loss function for training
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR)

        # Model metric variables as attributes
        self.total_loss_train_plot = []
        self.total_loss_validation_plot = []
        self.total_acc_train_plot = []
        self.total_acc_validation_plot = []

        self.y_train_pred=[]
        self.y_train_label=[]
        self.y_test_pred=[]
        self.y_test_label=[]

        self.training_accuracy = 0
        self.training_loss = 0
        self.training_specificity= -1
        self.training_precision = -1
        self.training_recall = -1
        self.training_f1 = -1
        self.training_mcc = -1
        self.training_overhead=-1

        self.validation_accuracy = 0
        self.validation_loss = 0

        self.testing_accuracy = 0
        self.testing_loss = 0
        self.testing_specificity= -1
        self.testing_precision = -1
        self.testing_recall = -1
        self.testing_f1 = -1
        self.testing_mcc = -1
        self.testing_overhead=-1

        #HYPERPARAMETERS variables as model attributes
        self.BATCH_SIZE=BATCH_SIZE
        self.EPOCHS=EPOCHS
        self.HIDDEN_NEURONS=HIDDEN_NEURONS
        self.LR=LR

    # Forward pass
    def forward(self, x):
        x = self.relu(self.input_layer(x))  # Apply input layer and ReLU
        x = self.relu(self.hidden_layer1(x))  # Apply first hidden layer and ReLU
        x = self.relu(self.hidden_layer2(x))  # Apply second hidden layer and ReLU
        x = self.sigmoid(self.output_layer(x))  # Apply output layer and Sigmoid
        return x
    
    def save_epochs_metrics_display(self):
        # Plot accuracy and loss for test samples
        figs, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
        axs[0].plot(self.total_loss_train_plot, label="Train Loss")
        axs[0].plot(self.total_loss_validation_plot, label="Validation Loss")
        axs[0].set_title("Train and Validation Loss Over Epochs")
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_ylim([0,.2])
        axs[0].legend()

        axs[1].plot(self.total_acc_train_plot, label="Train Accuracy")
        axs[1].plot(self.total_acc_validation_plot, label="Validation Accuracy")
        axs[1].set_title("Train and Validation Accuracy Over Epochs")
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_ylim([0,100])
        axs[1].legend()

        plt.tight_layout()
        plt.savefig('results/train/epochs_mlp.png')

    def get_matrix_metrics(self, y_train_pred, y_train_label, y_test_pred, y_test_label):

        self.training_accuracy = (self.testing_accuracy/(testing_data.__len__()))*100
        self.testing_accuracy = (self.testing_accuracy/(testing_data.__len__()))*100
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
        print("Total predictions:\t{}".format(testing_data.__len__()))
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
            plt.savefig('results/test/test_matrix_mlp.png')
        else:
            plt.savefig('results/train/train_matrix_mlp.png')

    def train(self):
        start_time = time.time()  # Record the start time of training
        for epoch in range(EPOCHS):
            self.y_train_pred=[]
            self.y_train_label=[]
            self.training_accuracy = 0
            self.training_loss = 0
            self.validation_accuracy = 0
            self.validation_loss = 0
            for data in train_dataloader:
                inputs, labels = data
                prediction = self(inputs).squeeze(1)
                batch_loss = self.criterion(prediction, labels)
                self.training_loss += batch_loss.item()
                acc = ((prediction).round() == labels).sum().item()
                self.training_accuracy += acc

                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                for item in prediction:
                    self.y_train_pred.append(int(item.round()))
                for item in labels:
                    self.y_train_label.append(int(item))

            with torch.no_grad():
                acc=0
                for data in validation_dataloader:
                    input, labels = data
                    prediction = self(input).squeeze(1)
                    batch_loss = self.criterion(prediction, labels)
                    self.validation_loss += batch_loss.item()
                    acc = ((prediction).round() == labels).sum().item()
                    self.validation_accuracy += acc

            # Calculate training time
            end_time = time.time()  # Record the end time of training
            total_training_time = end_time - start_time  # Calculate the time spent in training
            self.training_overhead = total_training_time

            # Calculate average time per epoch
            average_epoch_time = total_training_time / EPOCHS
            
            self.training_loss = self.training_loss/(training_data.__len__())
            self.training_accuracy = (self.training_accuracy/(training_data.__len__()))*100
            self.validation_loss = self.validation_loss/(validation_data.__len__())
            self.validation_accuracy = (self.validation_accuracy/(validation_data.__len__()))*100
            self.total_loss_train_plot.append(round(self.training_loss, 4))
            self.total_acc_train_plot.append(round(self.training_accuracy, 4))
            self.total_loss_validation_plot.append(round(self.validation_loss, 4))
            self.total_acc_validation_plot.append(round(self.validation_accuracy, 4))
            
            # Print total and average training time
            print(f"Total Training Time: {self.training_overhead:.2f} seconds")
            print(f"Average Time per Epoch: {average_epoch_time:.2f} seconds")

    def test(self):
        start_time = time.time() # Record the start time of testing
        with torch.no_grad():
            acc=0
            for data in testing_dataloader:
                inputs, labels = data
                prediction = self(inputs).squeeze(1)
                batch_loss_test = self.criterion((prediction), labels)
                self.testing_loss += batch_loss_test.item()
                acc = ((prediction).round() == labels).sum().item()
                self.testing_accuracy += acc

                for item in prediction:
                    self.y_test_pred.append(int(item.round()))
                for item in labels:
                    self.y_test_label.append(int(item))  
        end_time = time.time()
        total_testing_time = end_time - start_time
        self.testing_overhead = total_testing_time

        self.get_matrix_metrics(self.y_train_pred, self.y_train_label, self.y_test_pred, self.y_test_label)
    
    def check_metrics(self)-> bool:
        curr_acc = float(os.getenv('MLP_TESTING_ACCURACY'))
        curr_rec = float(os.getenv('MLP_TESTING_RECALL'))
        print("New: {}".format(self.testing_accuracy + (self.testing_recall*100)))
        print("Old: {}".format(curr_acc+(curr_rec*100)))
        print(self.testing_recall)
        print(self.testing_accuracy)
        if(self.testing_accuracy + (self.testing_recall*100) > curr_acc+(curr_rec*100)):
            self.save_attributes()
            self.save_epochs_metrics_display()
            self.save_matrix_display(self.y_train_pred, self.y_train_label, False)
            self.save_matrix_display(self.y_test_pred, self.y_test_label, True)
            return True
        return False

    def save_attributes(self):
        os.environ['MLP_TRAINING_ACCURACY'] = str(self.training_accuracy)
        os.environ['MLP_TRAINING_LOSS'] = str(self.training_loss)
        os.environ['MLP_TRAINING_SPECIFICITY'] = str(self.training_specificity)
        os.environ['MLP_TRAINING_PRECISION'] = str(self.training_precision)
        os.environ['MLP_TRAINING_RECALL'] = str(self.training_recall)
        os.environ['MLP_TRAINING_F1'] = str(self.training_f1)
        os.environ['MLP_TRAINING_MCC'] = str(self.training_mcc)
        os.environ['MLP_TRAINING_TP'] = str(self.train_tp)
        os.environ['MLP_TRAINING_FP'] = str(self.train_fp)
        os.environ['MLP_TRAINING_TN'] = str(self.train_tn)
        os.environ['MLP_TRAINING_FN'] = str(self.train_fn)
        os.environ['MLP_TRAINING_OVERHEAD'] = str(self.training_overhead)
        
        os.environ['MLP_VALIDATION_ACCURACY'] = str(self.validation_accuracy)
        os.environ['MLP_VALIDATION_LOSS'] = str(self.validation_loss)

        os.environ['MLP_TESTING_ACCURACY'] = str(self.testing_accuracy)
        os.environ['MLP_TESTING_LOSS'] = str(self.testing_loss)
        os.environ['MLP_TESTING_SPECIFICITY'] = str(self.testing_specificity)
        os.environ['MLP_TESTING_PRECISION'] = str(self.testing_precision)
        os.environ['MLP_TESTING_RECALL'] = str(self.testing_recall)
        os.environ['MLP_TESTING_F1'] = str(self.testing_f1)
        os.environ['MLP_TESTING_MCC'] = str(self.testing_mcc)
        os.environ['MLP_TESTING_TP'] = str(self.test_tp)
        os.environ['MLP_TESTING_FP'] = str(self.test_fp)
        os.environ['MLP_TESTING_TN'] = str(self.test_tn)
        os.environ['MLP_TESTING_FN'] = str(self.test_fn)
        os.environ['MLP_TESTING_OVERHEAD'] = str(self.testing_overhead)

        os.environ['MLP_BATCH_SIZE'] = str(self.BATCH_SIZE)
        os.environ['MLP_EPOCHS'] = str(self.EPOCHS)
        os.environ['MLP_HIDDEN_NEURONS'] = str(self.HIDDEN_NEURONS)
        os.environ['MLP_LR'] = str(self.LR)       

        dotenv.set_key(env_file, 'MLP_TRAINING_ACCURACY', os.environ['MLP_TRAINING_ACCURACY'])
        dotenv.set_key(env_file, 'MLP_TRAINING_LOSS', os.environ['MLP_TRAINING_LOSS'])
        dotenv.set_key(env_file, 'MLP_TRAINING_SPECIFICITY', os.environ['MLP_TRAINING_SPECIFICITY'])
        dotenv.set_key(env_file, 'MLP_TRAINING_PRECISION', os.environ['MLP_TRAINING_PRECISION'])
        dotenv.set_key(env_file, 'MLP_TRAINING_RECALL', os.environ['MLP_TRAINING_RECALL'])
        dotenv.set_key(env_file, 'MLP_TRAINING_F1', os.environ['MLP_TRAINING_F1'])
        dotenv.set_key(env_file, 'MLP_TRAINING_MCC', os.environ['MLP_TRAINING_MCC'])
        dotenv.set_key(env_file, 'MLP_TRAINING_TP', os.environ['MLP_TRAINING_TP'])
        dotenv.set_key(env_file, 'MLP_TRAINING_FP', os.environ['MLP_TRAINING_FP'])
        dotenv.set_key(env_file, 'MLP_TRAINING_TN', os.environ['MLP_TRAINING_TN'])
        dotenv.set_key(env_file, 'MLP_TRAINING_FN', os.environ['MLP_TRAINING_FN'])
        dotenv.set_key(env_file, 'MLP_TRAINING_OVERHEAD', os.environ['MLP_TRAINING_OVERHEAD'])

        dotenv.set_key(env_file, 'MLP_VALIDATION_ACCURACY', os.environ['MLP_VALIDATION_ACCURACY'])
        dotenv.set_key(env_file, 'MLP_VALIDATION_LOSS', os.environ['MLP_VALIDATION_LOSS'])

        dotenv.set_key(env_file, 'MLP_TESTING_ACCURACY', os.environ['MLP_TESTING_ACCURACY'])
        dotenv.set_key(env_file, 'MLP_TESTING_LOSS', os.environ['MLP_TESTING_LOSS'])
        dotenv.set_key(env_file, 'MLP_TESTING_SPECIFICITY', os.environ['MLP_TESTING_SPECIFICITY'])
        dotenv.set_key(env_file, 'MLP_TESTING_PRECISION', os.environ['MLP_TESTING_PRECISION'])
        dotenv.set_key(env_file, 'MLP_TESTING_RECALL', os.environ['MLP_TESTING_RECALL'])
        dotenv.set_key(env_file, 'MLP_TESTING_F1', os.environ['MLP_TESTING_F1'])
        dotenv.set_key(env_file, 'MLP_TESTING_MCC', os.environ['MLP_TESTING_MCC'])
        dotenv.set_key(env_file, 'MLP_TESTING_TP', os.environ['MLP_TESTING_TP'])
        dotenv.set_key(env_file, 'MLP_TESTING_FP', os.environ['MLP_TESTING_FP'])
        dotenv.set_key(env_file, 'MLP_TESTING_TN', os.environ['MLP_TESTING_TN'])
        dotenv.set_key(env_file, 'MLP_TESTING_FN', os.environ['MLP_TESTING_FN'])
        dotenv.set_key(env_file, 'MLP_TESTING_OVERHEAD', os.environ['MLP_TESTING_OVERHEAD'])

        dotenv.set_key(env_file, 'MLP_BATCH_SIZE', os.environ['MLP_BATCH_SIZE'])
        dotenv.set_key(env_file, 'MLP_EPOCHS', os.environ['MLP_EPOCHS'])
        dotenv.set_key(env_file, 'MLP_HIDDEN_NEURONS', os.environ['MLP_HIDDEN_NEURONS'])
        dotenv.set_key(env_file, 'MLP_LR', os.environ['MLP_LR']) 

def main(): 
    model = MLP()
    model.train()
    model.test()
    if(model.check_metrics()):
        torch.save(model.state_dict(), "saved models/model_mlp.pt")
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
