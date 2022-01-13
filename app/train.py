import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import bag_of_words, normalize, tokenize
from utils import stem, stop_words, pickle_read, pickle_save
from utils import Dataset, DataLoader
from models import NeuralNetwork

PICKLE_FILE = "data.pickle"
PYTORCH_FILE = "save.ptc"

with open("intents.json", "r") as f:
    intents = json.load(f)

try: 
    data = pickle_read(PICKLE_FILE)
except Exception as e: 
    print(e)
    data = {"all_tokens": set(), "tags": set(), "token_tag": list()} 

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            normalized_pattern = normalize(pattern)
            tokens = tokenize(normalized_pattern)
            # token_list = [stem(token) for token in tokens if token not in stop_words]
            token_list = [stem(token) for token in tokens]
            
            data["all_tokens"].update(token_list)
            data["tags"].add(intent["tag"])
            data["token_tag"].append((token_list, intent["tag"]))
            
    pickle_save(file_name=PICKLE_FILE, data=data)

data["all_tokens"] = sorted(data["all_tokens"])
data["tags"] = sorted(data["tags"])
X_train = []
y_train = []

for token_list, tag in data["token_tag"]:
    X_train.append([val for val in bag_of_words(token_list, data["all_tokens"]).values()])
    y_train.append(sorted(data["tags"]).index(tag))

print(data["token_tag"])
print("###############################")
print(X_train, "###", y_train)


n_inputs = len(data["all_tokens"])
n_hidden = 20
n_outputs = len(data["tags"])
epoch_size = 1000
batch_size = 10
learning_rate = 0.001

dataset = Dataset(np.array(X_train), np.array(y_train))
loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

model = NeuralNetwork(n_inputs, n_hidden, n_outputs)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)


def train():
    for epoch in tqdm(range(epoch_size)):
        for X, y in loader:

            y_pred = model.forward(X)
            loss = criterion(y_pred, torch.tensor(y, dtype=torch.long))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if not (epoch+1) % 100:
            print (f'Epoch: {epoch+1}/{epoch_size}, Loss: {loss.item():.4f}')
    print(f'final loss: {loss.item():.4f}')
    torch.save({"model_state": model.state_dict(), "n_inputs": n_inputs,
                "n_hidden": n_hidden, "n_outputs": n_outputs}, 
                PYTORCH_FILE)

if __name__ == "__main__":
    train()
