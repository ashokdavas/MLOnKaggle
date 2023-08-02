# https://www.kaggle.com/c/digit-recognizer

# Import libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
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
train_features = train_features.values.reshape(-1, 28, 28, 1)
val_features = val_features.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# Convert labels to categorical
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)

# Normalize features
train_features = train_features / 255
val_features = val_features / 255

# Create model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=10, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)

# Train model
start_time = time.time()
model.fit(train_features, train_labels, epochs=30, validation_data=(val_features, val_labels), callbacks=[early_stopping_monitor])
end_time = time.time()
print("Training time: {}".format(end_time - start_time))

# Save model
model.save('digit_recognizer_model.h5')

# Make predictions
predictions = model.predict(test)
predictions = np.argmax(predictions, axis=1)

# Create submission file
submission = pd.DataFrame({'ImageId': range(1, len(predictions) + 1), 'Label': predictions})
submission.to_csv('digit_recognizer_submission.csv', index=False)

# Create confusion matrix
val_predictions = model.predict(val_features)
val_predictions = np.argmax(val_predictions, axis=1)
val_labels = np.argmax(val_labels, axis=1)
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

