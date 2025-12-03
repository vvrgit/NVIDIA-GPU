import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

import pandas as pd

data = {
    "text": [
        "I love this movie",
        "This is a bad product",
        "Amazing experience",
        "Very disappointing",
        "I am happy",
        "I hate this"
    ],
    "label": [1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

import nltk
import re
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)
df["tokens"] = df["text"].apply(word_tokenize)

# Build Vocabulary
all_words = [word for tokens in df["tokens"] for word in tokens]
vocab = Counter(all_words)
vocab_size = len(vocab)

word2idx = {word: idx+1 for idx, word in enumerate(vocab)}

# Convert text to numbers
def encode_sentence(tokens):
    return [word2idx[word] for word in tokens]

df["encoded"] = df["tokens"].apply(encode_sentence)

#Padding sequences
from torch.nn.utils.rnn import pad_sequence
X = [torch.tensor(seq) for seq in df["encoded"]]
X = pad_sequence(X, batch_first=True)
y = torch.tensor(df["label"].values)

X = X.to(device)
y = y.to(device)


import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super(SentimentLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size+1, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return self.sigmoid(out)



model = SentimentLSTM(vocab_size).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 100

for epoch in range(epochs):
    model.train()
    
    optimizer.zero_grad()
    outputs = model(X).squeeze()
    
    loss = criterion(outputs, y.float())
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


def predict_sentiment(text):
    model.eval()
    
    text = clean_text(text)
    tokens = word_tokenize(text)
    encoded = torch.tensor([word2idx.get(word, 0) for word in tokens]).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(encoded)
        return "Positive" if output.item() > 0.5 else "Negative"

# Test
print(predict_sentiment("I really love this product"))
print(predict_sentiment("This is very bad"))


torch.save(model.state_dict(), "sentiment_lstm_gpu.pth")
model.load_state_dict(torch.load("sentiment_lstm_gpu.pth", map_location=device))
model.eval()
