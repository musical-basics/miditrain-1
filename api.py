from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
import time
from pathlib import Path
from miditok import REMI, TokenizerConfig
from symusic import Score

# Import your model class
from model import MidiCorrector

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the Tokenizer and Model on startup
print("Loading model into memory...")
tokenizer = REMI(TokenizerConfig(num_velocities=16, use_chords=False, use_programs=False))
vocab_size = len(tokenizer)

model = MidiCorrector(vocab_size=vocab_size).to(device)
# Load the weights we saved in train.py!
model.load_state_dict(torch.load("midi_corrector_weights.pth", map_location=device))
model.eval() # Set model to inference mode (turns off training-specific layers)
print("Model ready!")

@app.post("/clean-midi")
async def clean_midi(file: UploadFile = File(...)):
    # Save the incoming messy file temporarily
    input_path = f"temp_{file.filename}"
    with open(input_path, "wb") as f:
        f.write(await file.read())
        
    # 1. Tokenize the incoming file
    messy_midi = Score(input_path)
    messy_tokens = torch.tensor(tokenizer(messy_midi)[0].ids).unsqueeze(0).to(device)
    
    # 2. Run Inference (The AI prediction)
    # Note: In a real Transformer, you generate this token-by-token (autoregressively).
    # For this microapp API skeleton, we are wrapping that logic in a conceptual generate method.
    with torch.no_grad():
        # Let's assume you've added a .generate() method to your model class
        # that handles the token-by-token loop.
        clean_tokens = model.generate(messy_tokens, max_length=len(messy_tokens[0]) + 50)
    
    # 3. Detokenize back to audio
    output_path = f"cleaned_{file.filename}"
    
    # We must construct a TokSequence conceptually or just pass the ids.
    # MidiTok usually takes a list of integers here, so .tolist() on 1D tensor works.
    tokens_list = clean_tokens.cpu().numpy().tolist()
    
    # MidiTok __call__ supports directly translating tokens to Score in typical versions
    recovered_midi = tokenizer(tokens_list)
    recovered_midi.dump_midi(output_path)
    
    # Clean up the temp input file
    Path(input_path).unlink()
    
    # Send the cleaned MIDI file back to the user!
    return FileResponse(output_path, media_type="audio/midi", filename=output_path)
