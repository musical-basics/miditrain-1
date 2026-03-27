import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from dotenv import load_dotenv

import wandb
from supabase import create_client, Client, ClientOptions

# Import the NEW dataset and collate function we built
from data_pipelinev2 import MeasureAlignedMidiDataset, pad_collate_fn
from model import MidiCorrector

# ==========================================
# 1. THE RIGHT-BRAIN LOSS FUNCTION
# ==========================================
class AutonomousMusicalLoss(nn.Module):
    def __init__(self, pad_token_id, ce_weight=0.8, harmony_weight=0.2):
        super().__init__()
        self.pad_token_id = pad_token_id
        
        # How much we care about exact matching (Syntax) vs. overall vibe (Gestalt)
        self.ce_weight = ce_weight
        self.harmony_weight = harmony_weight

    def forward(self, logits, targets, dissonance_mask):
        # --- THE LEFT BRAIN (Syntax & Precision) ---
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), 
            targets.reshape(-1), 
            ignore_index=self.pad_token_id
        )
        
        # --- THE RIGHT BRAIN (Context & Gestalt) ---
        # Convert raw guesses into smooth percentages (probabilities)
        probs = F.softmax(logits, dim=-1)
        
        # dissonance_mask shape is (Batch, Vocab_Size). 
        # We add a dimension so it broadcasts across the entire sequence length: (Batch, 1, Vocab_Size)
        # Any "good" note gets multiplied by 0. Any "bad" out-of-key note gets multiplied by 1.
        penalized_probs = probs * dissonance_mask.unsqueeze(1) 
        
        # Sum up all that "bad vibe" probability and average it out
        harmony_loss = torch.mean(torch.sum(penalized_probs, dim=-1))
        
        # Combine the strict spelling test with the vibe check
        return (self.ce_weight * ce_loss) + (self.harmony_weight * harmony_loss)


# ==========================================
# 2. CONFIGURATION & SETUP
# ==========================================
# Load environment variables (Supabase keys)
load_dotenv(".env.local")

config = {
    "run_name": f"midi_corrector_gestalt_{int(time.time())}",
    "d_model": 256,
    "nhead": 8,
    "num_layers": 4,
    "batch_size": 4,
    "learning_rate": 0.0001,
    "epochs": 10,
    "measures_per_chunk": 4 # Upgraded from 'seconds_per_chunk'
}

print("Initializing Weights & Biases...")
wandb.init(project="midicorrector", name=config["run_name"], config=config)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=ClientOptions(schema="miditrain"))
    print("Connected to Supabase.")
else:
    supabase = None
    print("WARNING: Supabase credentials not found in .env. Database logging disabled.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# ==========================================
# 3. DATA & MODEL INITIALIZATION
# ==========================================
# Load our newly structured data
dataset = MeasureAlignedMidiDataset(
    "raw_midi_grieg_waltz.mid", 
    "cleaned_midi_grieg_waltz.mid", 
    measures_per_chunk=config["measures_per_chunk"]
)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=pad_collate_fn, shuffle=True)

vocab_size = len(dataset.tokenizer)
model = MidiCorrector(
    vocab_size=vocab_size,
    d_model=config["d_model"],
    nhead=config["nhead"],
    num_layers=config["num_layers"]
).to(device)

optimizer = Adam(model.parameters(), lr=config["learning_rate"])

# Initialize our new Gestalt-aware loss function!
criterion = AutonomousMusicalLoss(
    pad_token_id=dataset.pad_token_id, 
    ce_weight=0.8, 
    harmony_weight=0.2
)

# ==========================================
# 4. THE GESTALT TRAINING LOOP
# ==========================================
print("Starting holistic training loop...")
for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to GPU, including our new right-brain mask
        messy = batch["messy"].to(device)
        clean = batch["clean"].to(device)
        dissonance_mask = batch["mask"].to(device) 
        
        tgt_input = clean[:, :-1]
        tgt_expected = clean[:, 1:]
        
        optimizer.zero_grad()
        
        # The model makes its guesses (Batch, Seq_Len, Vocab_Size)
        logits = model(messy, tgt_input)
        
        # We grade the model not just on exact matches, but on its holistic 
        # understanding of the harmony, dictated by the dynamically generated mask.
        loss = criterion(logits, tgt_expected, dissonance_mask)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{config['epochs']} | Average Holistic Loss: {avg_loss:.4f}")
    
    wandb.log({"epoch": epoch + 1, "loss": avg_loss})

wandb.finish()

# ==========================================
# 5. SAVE & LOG
# ==========================================
weights_filename = f"{config['run_name']}.pth"
torch.save(model.state_dict(), weights_filename)
print(f"\nTraining complete. Gestalt-aware model saved as '{weights_filename}'!")

if supabase:
    print("Logging final run metadata to Supabase...")
    try:
        data, count = supabase.table('training_runs').insert({
            "run_name": config["run_name"],
            "model_architecture": "Gestalt-Aware Seq2Seq Transformer",
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "final_loss": avg_loss,
            "weights_path": weights_filename
        }).execute()
        print("✅ Successfully logged metrics to Supabase!")
    except Exception as e:
        print(f"❌ Failed to log to Supabase: {e}")