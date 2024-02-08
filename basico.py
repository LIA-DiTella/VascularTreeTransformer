import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from generateTrees import generate_random_tree, serialize

# Define the Transformer model
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(DecoderOnlyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_layers = nn.TransformerDecoderLayer(d_model, nhead)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layers, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        print("x", x)
        print("shape x", x.shape)
        x = self.embedding(x)
        #print("embedding", x.shape)
        memory = torch.zeros_like(x)
        output = self.transformer_decoder(x, memory)
        output = self.fc(output)
        return output

# Function for sequence generation
def generate_sequence(model, start_token, stop_token, max_length=10):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        current_token = torch.tensor([start_token])

        generated_sequence = [start_token]

        # Generate sequences until the stop token is encountered or reach max length
        for _ in range(max_length):
            logits = model(current_token.unsqueeze(0))  # Add batch dimension
            

            # Sample the next token using argmax
            next_token = torch.argmax(logits[:, -1, :]).item()
            # Append the next token to the generated sequence
            generated_sequence.append(next_token)

            # If the stop token is encountered, break the loop
            if next_token == stop_token:
                break

            # Update the current token for the next iteration
            current_token = torch.tensor([next_token])

        return generated_sequence

    
# Create a dummy dataset
def read_tree(filename, dir):
    #with open('./prof6/' +filename, "r") as f:
    #with open('./trees/' +filename, "r") as f:
    #with open('./' +dir +'/' +filename, "r") as f:
    with open(dir +'/' +filename, "r") as f:
        byte = f.read() 
        return byte

seq_length = 5
num_sequences = 100 
batch_size = 1
vocab_size = 50
dummy_dataset = torch.randint(0, vocab_size, (num_sequences, seq_length))

#print("dataset", dummy_dataset)
#print("shape", dummy_dataset.shape)
dummy_dataset = dummy_dataset.reshape(batch_size, -1, seq_length)  # Reshape for batching
#print("reshapeado", dummy_dataset)
#print("reshapeado", dummy_dataset.shape)

# Instantiate the model
#model = DecoderOnlyTransformer(vocab_size=vocab_size, d_model=768, nhead=16, num_layers=24)
model = DecoderOnlyTransformer(vocab_size=vocab_size, d_model=512, nhead=8, num_layers=4)


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with batches
num_epochs = 1
losses = []

for epoch in range(num_epochs):
    loss_batch = []
    for batch in dummy_dataset:
        
        optimizer.zero_grad()
        print("batch", batch)
        input_sequence = batch[:, :-1]
        target_sequence = batch[:, 1:]

        # Forward pass
        outputs = model(input_sequence)
        #print("input", input_sequence)
        #print("target", target_sequence)
        #print("output", outputs)
        # Calculate the loss using the shifted target sequence
        #breakpoint()
        loss = criterion(outputs.view(-1, vocab_size), target_sequence.reshape(-1))
        loss.backward()
        optimizer.step()
        loss_batch.append(loss.item())
    losses.append(np.average(loss_batch))
    if (epoch) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {np.average(loss_batch)}')

# Plotting the loss curve
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss Curve')
plt.show()

# Choose a starting token and stop token for generation
start_token = torch.randint(0, vocab_size, (1,))
stop_token = torch.randint(0, vocab_size, (1,))

# Generate a sequence using autoregressive sampling
generated_sequence = generate_sequence(model, start_token.item(), stop_token.item(), max_length=10)


# Print the generated sequence
print("Generated Sequence:")
print(generated_sequence)