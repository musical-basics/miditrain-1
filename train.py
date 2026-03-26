import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from dotenv import load_dotenv

import wandb
from supabase import create_client, Client

from data_pipeline import MidiCorrectionDataset, pad_collate_fn
from model import MidiCorrector

# Load environment variables (Supabase keys)
load_dotenv()

# --- Configurations ---
config = {
    "run_name": f"midi_corrector_{int(time.time())}",
    "d_model": 256,
    "nhead": 8,
    "num_layers": 4,
    "batch_size": 4,
    "learning_rate": 0.0001,
    "epochs": 10,
    "seconds_per_chunk": 30
}

# 1. Initialize Logging & Database
print("Initializing Weights & Biases...")
wandb.init(
    project="midicorrector",
    name=config["run_name"],
    config=config
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Connected to Supabase.")
else:
    supabase = None
    print("WARNING: Supabase credentials not found in .env. Database logging disabled.")

# 2. Setup Data, Model, and Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

dataset = MidiCorrectionDataset("raw_midi_grieg_waltz.mid", "cleaned_midi_grieg_waltz.mid", seconds_per_chunk=config["seconds_per_chunk"])
dataloader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=pad_collate_fn, shuffle=True)

vocab_size = len(dataset.tokenizer) 
model = MidiCorrector(
    vocab_size=vocab_size,
    d_model=config["d_model"],
    nhead=config["nhead"],
    num_layers=config["num_layers"]
).to(device)

# 3. Setup Optimizer and Loss Function
optimizer = Adam(model.parameters(), lr=config["learning_rate"])
pad_token_id = dataset.pad_token_id 
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

# 4. The Training Loop
print("Starting training loop...")
for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        messy = batch["messy"].to(device)
        clean = batch["clean"].to(device)
        
        tgt_input = clean[:, :-1]
        tgt_expected = clean[:, 1:]
        
        optimizer.zero_grad()
        logits = model(messy, tgt_input)
        
        logits = logits.reshape(-1, logits.shape[-1])
        tgt_expected = tgt_expected.reshape(-1)
        
        loss = criterion(logits, tgt_expected)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{config['epochs']} | Average Loss: {avg_loss:.4f}")
    
    # Log metrics to W&B
    wandb.log({
        "epoch": epoch + 1,
        "loss": avg_loss
    })

# Finish W&B run
wandb.finish()

# 5. Save the trained model and log to Supabase!
weights_filename = f"{config['run_name']}.pth"
torch.save(model.state_dict(), weights_filename)
print(f"\nTraining complete. Model saved as '{weights_filename}'!")

if supabase:
    print("Logging final run metadata to Supabase...")
    try:
        data, count = supabase.table('training_runs').insert({
            "run_name": config["run_name"],
            "model_architecture": "Seq2Seq Transformer",
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "final_loss": avg_loss,
            "weights_path": weights_filename
        }).execute()
        print("✅ Successfully logged metrics to Supabase!")
    except Exception as e:
        print(f"❌ Failed to log to Supabase: {e}")
