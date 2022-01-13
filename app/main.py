import random
import json

import torch

from models import NeuralNetwork
from utils import bag_of_words, normalize, tokenize, stem, pickle_read
from utils import stop_words

PICKLE_FILE = "data.pickle"
PYTORCH_FILE = "save.ptc"

model_data = torch.load(PYTORCH_FILE)
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

token_data = pickle_read(PICKLE_FILE)

model_state = model_data["model_state"]
n_inputs = model_data["n_inputs"]
n_hidden = model_data["n_hidden"]
n_outputs = model_data["n_outputs"]


model = NeuralNetwork(n_inputs, n_hidden, n_outputs)
model.load_state_dict(model_state)
model.eval()

print("Type q to quit...")
done = False
while not done:
    # sentence = "do you use credit cards?"
    sentence = input("you: ")
    if sentence == "q":
        done = True

    tokens = tokenize(normalize(sentence))
    token_list = [stem(token) for token in tokens]
    
    X = torch.tensor([val for val in bag_of_words(token_list, sorted(token_data["all_tokens"])).values()], dtype=torch.float)
    print(X)
    y = model(X)
    _, indices = torch.max(y, -1)
    prob, index = torch.max(torch.softmax(y, -1), -1)
    tag = sorted(token_data["tags"])[index.item()]
    print("####", tag)
    if prob.item() > 0.7:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"Bot: {random.choice(intent['responses'])}")
    else:
        print("bot: I do not understand...")