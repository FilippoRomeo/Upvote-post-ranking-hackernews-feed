# src/train_regression_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from regression_model import RegressionModel
from preprocess_data import preprocess_data, load_hacker_news_data
import psycopg2

# Set the connection to the PostgreSQL database
conn = psycopg2.connect(
    dbname="hd64m1ki", 
    user="sy91dhb", 
    password="g5t49ao", 
    host="178.156.142.230", 
    port="5432"
)

# Load the Word2Vec model (you can load the model you trained for CBOW)
from word2vec_model import CBOWModel

# Initialize the CBOWModel
vocab_size = 71292  # Set the appropriate vocabulary size (adjust according to your model)
embedding_dim = 500  # Adjust according to your embedding size
word2vec_model = CBOWModel(vocab_size, embedding_dim)

# Load the saved state_dict
checkpoint = torch.load('data/text8_cbow_model.pt')

# Extract only the model weights (excluding the extra information like optimizer_state_dict)
model_state_dict = checkpoint['model_state_dict']  # Assuming the model state is under this key
word2vec_model.load_state_dict(model_state_dict, strict=False)

# Load data and preprocess it
df = load_hacker_news_data(conn)
X, y = preprocess_data(df, word2vec_model)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create DataLoader for batching
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize the regression model
input_size = X.shape[1]  # Number of features (word embeddings)
model = RegressionModel(input_size=input_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.view(-1, 1))  # Reshaping target to match output
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'data/hacker_news_regression_model.pt')
