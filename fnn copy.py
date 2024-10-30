import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


# Set the device to GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loading Dataset
dataset = pd.read_csv('Dataset.csv', encoding="Latin-1")  # Read the dataset
dataset.dropna(inplace=True)  # Remove rows with missing values
dataset.pop("DoctorInCharge")  # Remove the "DoctorInCharge" column
dataset.pop("PatientID")  # Remove the "PatientID" column

# Normalize each feature column using Mac-abs normalization
for column in dataset.columns:
    dataset[column] = dataset[column] / dataset[column].abs().max()

# Split features and labels
X = np.array(dataset.iloc[:, :-1])  # Features (all columns except the last one)
Y = np.array(dataset.iloc[:, -1])    # Labels (last column)

# Splitting the dataset into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

# Print dataset shapes to understand the split
print(f"Training set: {X_train.shape} rows, {round(X_train.shape[0] / dataset.shape[0], 4) * 100}%")
print(f"Validation set: {X_val.shape} rows, {round(X_val.shape[0] / dataset.shape[0], 4) * 100}%")
print(f"Testing set: {X_test.shape} rows, {round(X_test.shape[0] / dataset.shape[0], 4) * 100}%")

# Custom Dataset class for loading data
class CSVDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)  # Convert features to tensor and move to device
        self.Y = torch.tensor(Y, dtype=torch.float32).to(device)  # Convert labels to tensor and move to device

    def __len__(self):
        return len(self.X)  # Return the number of samples

    def __getitem__(self, index):
        return self.X[index], self.Y[index]  # Return a sample and its label

# Hyperparameters for the model
BATCH_SIZE = 4
EPOCHS = 50
HIDDEN_NEURONS = 8
LR = 1e-2

# Create DataLoader for each dataset
train_dataloader = DataLoader(CSVDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(CSVDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
testing_dataloader = DataLoader(CSVDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# Define the MLP Model Class
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Define layers of the neural network
        self.input_layer = nn.Linear(X.shape[1], HIDDEN_NEURONS)
        self.hidden_layer1 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.hidden_layer2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.output_layer = nn.Linear(HIDDEN_NEURONS, 1)
        self.relu = nn.ReLU()  # ReLU activation function
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        # Define forward pass
        x = self.relu(self.input_layer(x))  # Apply input layer and ReLU
        x = self.relu(self.hidden_layer1(x))  # Apply first hidden layer and ReLU
        x = self.relu(self.hidden_layer2(x))  # Apply second hidden layer and ReLU
        x = self.sigmoid(self.output_layer(x))  # Apply output layer and Sigmoid
        return x

# Instantiate the model and move it to the specified device
model = MLP().to(device)

# Print the model architecture and total parameters
print(model)
total_params = sum(p.numel() for p in model.parameters())  # Count total parameters
print(f"Total parameters: {total_params}")
total_params_size = abs(total_params * 4.0 / (1024 ** 2.))  # Estimate memory size in MB
print(f"Total parameters size in MB: {total_params_size:.2f} MB")

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = Adam(model.parameters(), lr=LR)  # Adam optimizer

# Training the Model
total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []

for epoch in range(EPOCHS):
    total_loss_train = 0
    total_acc_train = 0
    model.train()  # Set model to training mode
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()  # Zero the gradients
        predictions = model(inputs).squeeze(1)  # Forward pass
        batch_loss = criterion(predictions, labels)  # Compute loss
        total_loss_train += batch_loss.item()  # Accumulate loss
        acc = ((predictions.round() == labels).sum().item())  # Calculate accuracy
        total_acc_train += acc
        batch_loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

    # Validation phase
    total_loss_val = 0
    total_acc_val = 0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No gradient tracking
        for inputs, labels in validation_dataloader:
            predictions = model(inputs).squeeze(1)  # Forward pass
            batch_loss = criterion(predictions, labels)  # Compute loss
            total_loss_val += batch_loss.item()  # Accumulate validation loss
            acc = ((predictions.round() == labels).sum().item())  # Calculate accuracy
            total_acc_val += acc

    # Record metrics for plotting
    total_loss_train_plot.append(total_loss_train / len(train_dataloader))
    total_loss_validation_plot.append(total_loss_val / len(validation_dataloader))
    total_acc_train_plot.append(total_acc_train / len(X_train))
    total_acc_validation_plot.append(total_acc_val / len(X_val))

    # Print training and validation results for the epoch
    print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {total_loss_train/1000:.4f}, Train Accuracy: {total_acc_train / len(X_train) * 100:.2f}%")
    print(f"Val Loss: {total_loss_val:.4f}, Val Accuracy: {total_acc_val / len(X_val) * 100:.2f}%")
    print("=" * 60)

# Testing the Model
with torch.no_grad():  # No gradient tracking
    total_loss_test = 0
    total_acc_test = 0
    y_pred=[]
    y_label=[]
    acc=0
    for data in testing_dataloader:
        inputs, labels = data
        prediction = model(inputs).squeeze(1)
        batch_loss_test = criterion((prediction), labels)
        total_loss_test += batch_loss_test.item()
        acc = ((prediction).round() == labels).sum().item()
        total_acc_test += acc

        for item in prediction:
            y_pred.append(int(item.round()))
        for item in labels:
            y_label.append(int(item))

#print(f"Test Accuracy: {total_acc_test / len(X_test) * 100:.2f}%")

# Confusion matrix with true neg, false pos, false neg, true pos respectively
tn, fp, fn, tp = confusion_matrix(y_true=y_label, y_pred=y_pred).ravel()      

precision = tp /(tp+fp)     # Correctly predicted positives over all predicted positives
specificity = tn / (tn+fp)  # Correctly predicted negatives over all actual negatives
recall = tp / (fn+tp)       # Correctly predicted positives over all actual positives

f1 = 2 * ((precision*recall)/(precision+recall))                        # F1 Score 
mcc = ((tp*tn) - (fp*fn))/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))  # Matthews Correlation Coefficient

# Print all calculated metrics for test samples
print("Test Accuracy:\t\t{}%".format(round((total_acc_test/X_test.shape[0])*100, 4)))
print("Total correct:\t\t{}".format(total_acc_test))
print("Total predictions:\t{}".format(X_test.shape[0]))
print("-"*60)
print("Precision:\t{}".format(round(precision, 4)))
print("Specificity:\t{}".format(round(specificity, 4)))
print("Recall:\t\t{}".format(round(recall, 4)))
print("F1:\t\t{}".format(round(f1, 4)))
print("MCC:\t\t{}".format(round(mcc, 4)))

# Plotting Metrics
figs, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
axs[0].plot(total_loss_train_plot, label="Train Loss")  # Plot training loss
axs[0].plot(total_loss_validation_plot, label="Validation Loss")  # Plot validation loss
axs[0].set_title("Train and Validation Loss Over Epochs")  # Set title
axs[0].set_xlabel('Epochs')  # Set x-label
axs[0].set_ylabel('Loss')  # Set y-label
axs[0].set_ylim([0, 2])  # Set y-axis limits
axs[0].legend()  # Show legend

axs[1].plot(total_acc_train_plot, label="Train Accuracy")  # Plot training accuracy
axs[1].plot(total_acc_validation_plot, label="Validation Accuracy")  # Plot validation accuracy
axs[1].set_title("Train and Validation Accuracy Over Epochs")  # Set title
axs[1].set_xlabel('Epoch')  # Set x-label
axs[1].set_ylabel('Accuracy')  # Set y-label
axs[1].set_ylim([0, 100])  # Set y-axis limits
axs[1].legend()  # Show legend

plt.tight_layout()  # Adjust layout
plt.show()  # Display the plots
