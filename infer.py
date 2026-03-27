"""
Inference Script: Feed the raw MIDI through the trained model and produce a "fixed" output.

Note: The model was only trained for 10 epochs on a single pair of files,
so the output will be a rough proof-of-concept, not a polished correction.
"""
import torch
from pathlib import Path
from miditok import REMI, TokenizerConfig
from symusic import Score

from model import MidiCorrector

# --- Config ---
INPUT_PATH = "raw_midi_grieg_waltz.mid"
OUTPUT_PATH = "fixed_grieg_waltz.mid"
WEIGHTS_PATH = "midi_corrector_weights.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running inference on: {device}")

# 1. Load Tokenizer
tokenizer = REMI(TokenizerConfig(num_velocities=16, use_chords=False, use_programs=False))
vocab_size = len(tokenizer)

# 2. Load Model
model = MidiCorrector(vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()
print("Model loaded.")

# 3. Tokenize the raw input
messy_midi = Score(INPUT_PATH)
messy_tokens = tokenizer(messy_midi)[0].ids
print(f"Input token count: {len(messy_tokens)}")

# 4. Run autoregressive generation
messy_tensor = torch.tensor(messy_tokens, dtype=torch.long).unsqueeze(0).to(device)

print("Generating corrected sequence (this may take a moment)...")
with torch.no_grad():
    output_tokens = model.generate(messy_tensor, max_length=len(messy_tokens))

# Strip the batch dimension and convert to list
output_ids = output_tokens.squeeze(0).cpu().tolist()
print(f"Output token count: {len(output_ids)}")

# 5. Detokenize back to MIDI
output_midi = tokenizer.decode(output_ids)
output_midi.dump_midi(OUTPUT_PATH)

print(f"\n✅ Fixed MIDI saved to: {OUTPUT_PATH}")
print(f"   Open it in your DAW or MIDI player to listen!")
