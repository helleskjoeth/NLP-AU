### NEURAL NETWORK OR LOGISTIC REGRESSION CLASSIFIER ###
# system tools
import os
# pytorch
import torch
import torch.nn as nn
# data processing
import pandas as pd
import numpy as np
# huggingface datasets
from datasets import load_dataset
# scikit learn tools
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
# plotting tools
import matplotlib.pyplot as plt

## Load data
# load the sst2 dataset
dataset = load_dataset("rotten_tomatoes", "hate")
# select the train split
print(dataset) # Check out the dataset, see what splits are available.
# Downloading the rotten tomatoes dataset
train = dataset["train"]
val = dataset["validation"]
test = dataset["test"]

# create lists for each split
X = train["text"]
X_val = val["text"]
X_test = test["text"]
y_train = train["label"]
y_val = val["label"]
y_test = test["label"]

# create vectorizer, in order to vectorize input tweets. Feature extraction, in other words.
vectorizer = CountVectorizer()
# vectorized training data
X_train_vect = vectorizer.fit_transform(X)
# only transform val and test
X_val_vect = vectorizer.transform(X_val)
X_test_vect = vectorizer.transform(X_test)
# to tensors
X_train_vect = torch.tensor(X_train_vect.toarray(), dtype=torch.float)
X_val_vect = torch.tensor(X_val_vect.toarray(), dtype=torch.float)
X_test_vect = torch.tensor(X_test_vect.toarray(), dtype=torch.float)
# y_train
y_train = torch.tensor(list(y_train), dtype=torch.float)
# y_validation
y_val =torch.tensor(list(y_val), dtype=torch.float)
# y_test
y_test = torch.tensor(list(y_test), dtype = torch.float)

# Adding extra axes
y_train = y_train.view(y_train.shape[0], 1)
y_val = y_val.view(y_val.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

# X_train_vect.shape # 8530 instances of ~16.000 features.
# X_val_vect.shape and X_test_vect.shape show the same number of features.

# Define model class
class Model(nn.Module):
    def __init__(self, n_input_features=10):            # default input features, can be overridden
        super().__init__()                              # inherit from parent class
        # self.linear = nn.Linear(n_input_features, 1)    # one linear layer with single output
        # LAYERS GO HERE       
        self.linear1 = nn.Linear(n_input_features, 30)
        self.linear2 = nn.Linear(30, 30)
        self.linear3 = nn.Linear(30, 1)

    def forward(self, x): # Function used for generating a prediction vector, given an input vector of features.
        a1 = self.linear1(x) # Activation in the first layer is created from input
        a1_sig = torch.sigmoid(a1) # Squish the activation values into a sigmoid function
        a2 = self.linear2(a1_sig) # Activation in the second layer determined from the first
        a2_sig = torch.sigmoid(a2) # Sigmoid again
        a3 = self.linear3(a2_sig) # Activation for third layer
        y_pred = torch.sigmoid(a3) # Predicted values are the sigmoid-transformed activation values in layer 3.
        return y_pred

# initialize model
n_samples, n_features = X_train_vect.shape # We needn't do this for the
model = Model(n_input_features=n_features) # Call the model class

# define loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters())

# train
epochs = 100
print("[INFO:] Training classifier...")
loss_history = []
loss_history_val = []
for epoch in range(epochs):
    # forward
    y_hat = model(X_train_vect) # Use model object on design matrix
    # backward
    loss = criterion(y_hat, y_train)
    loss_history.append(loss)
    # Validation time
    y_hat_val = model(X_val_vect) 
    loss_val = criterion(y_hat_val, y_val)
    loss_history_val.append(loss_val)
    # backpropagation
    loss.backward()
    # take step, reset
    optimizer.step()
    optimizer.zero_grad()
    # Validation after last epoch
    if epoch == epochs-1:
        y_hat_val = model(X_val_vect) 
        loss_val = criterion(y_hat_val, y_val)
        loss_history_val.append(loss_val)
    # some print to see that it is running
    if (epoch + 1) % 10 == 0:
        print(f"epoch: {epoch+1}, loss = {loss.item():.4f}, validation loss = {loss_val.item():.4f}")

print("[INFO:] Finished traning!")

loss_H = [val.item() for val in loss_history]
loss_val_H = [val.item() for val in loss_history_val]
fig, ax = plt.subplots()
ax.plot(loss_H)
ax.plot(loss_val_H)
ax.legend(["Training loss", "Validation loss"])

plt.savefig("out/loss.png")

# Plot
predicted = model(X_test_vect).detach().numpy()
report = classification_report(y_test, 
                            np.where(predicted > 0.5, 1, 0),
                            target_names = ["Negative", "Positive"])
print(report)
print(type(report))

# Saving the report as a txt file (Its a string. There are probably ways to save it in a better format given that it is
# very clearly delimited in some sense by whitespace or tab)
text_file = open(r'out/classification_report.txt', 'w')
my_string = report
text_file.write(my_string)
text_file.close()

# Setting up the ability to call from terminal