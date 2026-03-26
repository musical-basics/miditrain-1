import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_pipeline import MidiCorrectionDataset, pad_collate_fn
from model import MidiCorrector

# 1. Setup Data, Model, and Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

dataset = MidiCorrectionDataset("raw_midi_grieg_waltz.mid", "cleaned_midi_grieg_waltz.mid", seconds_per_chunk=30)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=pad_collate_fn, shuffle=True)

# Note: You get vocab_size from your tokenizer (e.g., len(dataset.tokenizer))
vocab_size = len(dataset.tokenizer) 
model = MidiCorrector(vocab_size=vocab_size).to(device)

# 2. Setup Optimizer and Loss Function
optimizer = Adam(model.parameters(), lr=0.0001)

# We tell the loss function to ignore the "PAD" tokens (usually 0) 
# so it doesn't waste time learning how to predict empty space.
pad_token_id = dataset.pad_token_id 
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

# 3. The Training Loop
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        messy = batch["messy"].to(device)
        clean = batch["clean"].to(device)
        
        # The Seq2Seq Trick (Teacher Forcing):
        # The model's decoder gets the clean sequence minus the last token as input.
        # It tries to predict the clean sequence minus the first token as output.
        tgt_input = clean[:, :-1]
        tgt_expected = clean[:, 1:]
        
        # Clear old gradients
        optimizer.zero_grad()
        
        # Forward Pass: Make a prediction
        logits = model(messy, tgt_input)
        
        # Reshape the outputs so PyTorch's loss function can read them
        logits = logits.reshape(-1, logits.shape[-1])
        tgt_expected = tgt_expected.reshape(-1)
        
        # Calculate how wrong the prediction was
        loss = criterion(logits, tgt_expected)
        
        # Backward Pass: Calculate the adjustments needed
        loss.backward()
        
        # Optimizer Step: Actually update the model's brain
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")

# 4. Save the trained model!
torch.save(model.state_dict(), "midi_corrector_weights.pth")
print("\nTraining complete. Model saved as 'midi_corrector_weights.pth'!")
