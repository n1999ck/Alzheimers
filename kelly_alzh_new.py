import torch
import torch.nn as nn
nn.Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR, CyclicLR, OneCycleLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, log_loss, balanced_accuracy_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, VotingRegressor, RandomForestRegressor
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
    #def init(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) 
        # Linear function
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) 
        self.bn2 = nn.BatchNorm1d(hidden_dim) 
        self.fc3 = nn.Linear(hidden_dim, output_dim)  
        #self.reLU = nn.ReLU()  #Rectified Linear Unit
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        lstm_out, _=self.lstm(x)
        out = lstm_out[:, :]
        # Linear function  # LINEAR
        out = self.fc1(x)
        out = self.bn1(out)
        # Non-linearity  # NON-LINEAR
        out = self.sigmoid(out)
        #out = self.reLU(out)
        out = self.dropout(out)
        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.sigmoid(out)
        #out = self.reLU(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

class CSVDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
# Create test and train datasets    
dataset = pd.read_csv('Dataset.csv', encoding="ISO-8859-1")
dataset_labels = dataset.pop("Diagnosis")
dataset.pop("DoctorInCharge")
dataset.pop("PatientID")

x = dataset.values
y = dataset_labels
# Models
modelR = RandomForestClassifier(n_estimators=100, min_samples_split=5)
modelLR = LogisticRegression(solver='saga', max_iter=7600) #max_iter 7600
modelEN = ElasticNet(alpha=1.0, l1_ratio=0.5)
ensemble_model = VotingClassifier(estimators=[('rf', modelR), ('lr', modelLR), ('en', modelEN)], voting='hard')
# Fitting models

modelLR.fit(x, y)
modelEN.fit(x, y)
modelR.fit(x, y) 

param_grid = {
    'n_estimators': [100, 200, 400, 600],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}
#Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_rf = cross_val_score(modelR,x, y, cv=kf)
scores_ensemble = cross_val_score(modelEN, x, y, cv=kf)

print("Ensemble Model Cross validation scores:",(scores_ensemble))
print("Ensemble Model Mean Score: ", np.mean(scores_ensemble))
print("Random Forest Cross-validation scores:",(scores_rf))
print("Random Forest Mean score:", np.mean(scores_rf))
print("Random Forest Standard Deviation:", np.std(scores_rf))
scores_en = cross_val_score(modelEN, x, y, cv=kf)
print("Elastic Net Cross-validation scores: ",(scores_en))
print("Elastic Net Mean Score: ", np.mean(scores_en))

#grid_search = GridSearchCV(estimator=modelR, param_grid=param_grid, cv=kf, scoring='accuracy')
#grid_search.fit(x,y)  #source: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

#print("Best parameters found: ", grid_search.best_params_)
#print("Best cross-validation score: ", grid_search.best_score_)

test_portion_dataset = dataset[int((len(dataset) * (7/8))):]
test_portion_dataset_labels = dataset_labels[int((len(dataset) * (7/8))):]
train_portion_dataset = dataset[:int((len(dataset) * (7/8)))]
train_portion_dataset_labels = dataset_labels[:int((len(dataset) * (7/8)))]
print(len(test_portion_dataset))
print(len(test_portion_dataset_labels))
print(len(train_portion_dataset))
print(len(train_portion_dataset_labels))

scaler = StandardScaler()   
dataset_features = np.vstack(train_portion_dataset.values).astype(np.float32)   
test_dataset_features = np.vstack(test_portion_dataset.values).astype(np.float32)
print(dataset_features)

dataset_features_scaled = scaler.fit_transform(dataset_features)
test_dataset_features_scaled = scaler.transform(test_dataset_features)

#training parameters
batch_size = 64
n_iters = 6400 # was - changed from 3000
num_epochs = n_iters / (len(dataset_features) / batch_size)
num_epochs = int(num_epochs)
print(num_epochs)
input_dim = 32
output_dim = 2
hidden_dim = 21   #64    #512 hidden layer 2/3 of input + output
lstm_hidden_dim = 2
dropout_prob = 0.6
#Initialize model
train_dataset = CSVDataset(dataset_features, train_portion_dataset_labels)
test_dataset = CSVDataset(test_dataset_features, test_portion_dataset_labels)
dataset_loader =torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)

test_loader =torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
step_scheduler = StepLR(optimizer, step_size=5, gamma=0.1) 
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0) #T_max=20
one_cycle_scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=35000)  

val_split = 0.25    #.15
val_size = int(len(train_portion_dataset) * val_split)
train_size = int(len(train_portion_dataset) * (1- val_split))
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,[train_size, val_size])
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

actual = np.random.binomial(1,.9, size=1504) 
predicted = np.random.binomial(1,.9, size =1504)

confusion_matrix = metrics.confusion_matrix(actual, predicted)
FPR = confusion_matrix[0][1] / (confusion_matrix[0][1] + confusion_matrix[0][0])
FNR = confusion_matrix[1][0] / (confusion_matrix[1][0] + confusion_matrix[1][1])

#cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
#cm_display.plot()
#plt.show()

#Accuracy = metrics.accuracy_score(actual, predicted)
#Precision = metrics.precision_score(actual, predicted)
#Sensitivity_recall = metrics.recall_score(actual, predicted)
#Specificity = metrics.recall_score(actual, predicted, pos_label=0)
#F1_score = metrics.f1_score(actual, predicted)
#print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score})

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for i, (fields, labels) in enumerate(val_loader):
        fields, labels= fields.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(fields)
        loss = criterion(outputs, labels)
        L1_Lambda = .001 #regularization tech to prevent overfitting
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss += L1_Lambda 
        loss.backward()
        optimizer.step()
        one_cycle_scheduler.step()
        total_loss += loss.item()
    scheduler.step()
    step_scheduler.step()
    
    print(f'Epoch: {epoch+1}/{num_epochs} completed., Loss: {total_loss/ len(val_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.4f}')
        
    # Validation Phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        
        for fields, labels in val_loader:
            fields, labels = fields.to(device), labels.to(device)
                
                # Forward pass only to get logits/output
            outputs = model(fields)
                           
                # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
                
                # Total number of labels
            total += labels.size(0)
                
                # Total correct predictions
            correct += (predicted == labels).sum().item()
                # Calculate loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # Avg validation loss over all batches
        avg_loss = val_loss / len(test_loader)
        print(f'Validation Loss: {avg_loss:.4f}')    
        print(f'Training Accuracy: {100 * correct/ total:.2f}%')
        
    correct = 0
    total = 0
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        __, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct/total
print(f'Testing Accuracy: {100 * correct/ total:.2f}%')