# https://www.kaggle.com/c/digit-recognizer

# Import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import shutil

# Import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Split training data into training and validation sets
train, val = train_test_split(train, test_size=0.2, random_state=42)

# Separate features and labels
train_labels = train['label']
train_features = train.drop(['label'], axis=1)
val_labels = val['label']
val_features = val.drop(['label'], axis=1)

# Reshape features
train_features = train_features.values.reshape(-1, 1, 28, 28)
val_features = val_features.values.reshape(-1, 1, 28, 28)
test = test.values.reshape(-1, 1, 28, 28)

# Convert labels to categorical
train_labels = torch.tensor(train_labels.values)
val_labels = torch.tensor(val_labels.values)

# Normalize features
train_features = train_features / 255
val_features = val_features / 255

# Convert features and labels to tensors
train_features = torch.tensor(train_features, dtype=torch.float32)
val_features = torch.tensor(val_features, dtype=torch.float32)
test = torch.tensor(test, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
val_labels = torch.tensor(val_labels, dtype=torch.long)

# Create dataset
train_dataset = TensorDataset(train_features, train_labels)
val_dataset = TensorDataset(val_features, val_labels)

# Create dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define the CNN model
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.relu5 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, tensors):
        tensors = self.conv1(tensors)
        tensors = self.relu1(tensors)
        tensors = self.conv2(tensors)
        tensors = self.relu2(tensors)
        tensors = self.maxpool1(tensors)
        tensors = self.dropout1(tensors)
        tensors = self.conv3(tensors)
        tensors = self.relu3(tensors)
        tensors = self.conv4(tensors)
        tensors = self.relu4(tensors)
        tensors = self.maxpool2(tensors)
        tensors = self.dropout2(tensors)
        tensors = self.flatten(tensors)
        tensors = self.fc1(tensors)
        tensors = self.relu5(tensors)
        tensors = self.dropout3(tensors)
        tensors = self.fc2(tensors)
        return tensors

# Compile model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNIST_CNN().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 64

# Set early stopping monitor
best_val_loss = float('inf')
patience = 3
counter = 0

# Train model
start_time = time.time()

for epoch in range(30):
    model.train()
    total_loss = 0
    
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping: Validation loss did not improve for {} epochs.".format(patience))
            break

    print(f"Epoch [{epoch + 1}/30], Training Loss: {total_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")

end_time = time.time()
print("Training time: {}".format(end_time - start_time))


# save model
torch.save(model.state_dict(), 'model.pth')

# predict test data
model.eval()
with torch.no_grad():
    test = test.to(device)
    outputs = model(test)
    _, predicted = torch.max(outputs.data, 1)
predictons = predicted.cpu().numpy()


# Create submission file
submission = pd.DataFrame({'ImageId': range(1, len(predictons) + 1), 'Label': predictons})
submission.to_csv('digit_recognizer_submission.csv', index=False)

# Evaluate model and create confusion matrix
model.eval()
with torch.no_grad():
    val_features = val_features.to(device)
    val_predictions = model(val_features)
    val_predictions = torch.argmax(F.softmax(val_predictions, dim=1), dim=1)
    # val_labels = torch.argmax(val_labels, dim=1)

val_predictions = val_predictions.cpu().numpy()
val_labels = val_labels.cpu().numpy()

confusion_matrix = confusion_matrix(val_labels, val_predictions)
print(confusion_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix, annot=True, fmt='.3f', linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix') 
plt.show()
plt.savefig('confusion_matrix.png')
plt.close()
