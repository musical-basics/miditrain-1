import torch
from torch.utils.data import Dataset, DataLoader
from miditok import REMI, TokenizerConfig
from symusic import Score

class MidiCorrectionDataset(Dataset):
    def __init__(self, messy_path, clean_path, measures_per_chunk=4):
        # 1. Initialize Tokenizer
        self.tokenizer = REMI(TokenizerConfig(num_velocities=16, use_chords=False, use_programs=False))
        self.pad_token_id = self.tokenizer["PAD_None"]
        
        # 2. Load the MIDI files
        self.messy_midi = Score(messy_path)
        self.clean_midi = Score(clean_path)
        
        # 3. Calculate time windows (Ticks)
        # Assuming 4/4 time. Ticks per quarter note * 4 beats * number of measures
        tpq = self.messy_midi.ticks_per_quarter
        self.window_size_ticks = tpq * 4 * measures_per_chunk
        
        # 4. Chop and align the data!
        self.data_pairs = self._chop_and_tokenize()

    def _chop_and_tokenize(self):
        pairs = []
        max_ticks = max(self._get_max_tick(self.messy_midi), self._get_max_tick(self.clean_midi))
        
        # Slide a window across the song by our designated chunk size
        for start_tick in range(0, max_ticks, self.window_size_ticks):
            end_tick = start_tick + self.window_size_ticks
            
            # Clip both MIDIs to the exact same time window
            messy_chunk = self.messy_midi.clip(start_tick, end_tick, clip_end=True)
            clean_chunk = self.clean_midi.clip(start_tick, end_tick, clip_end=True)
            
            # Skip empty chunks (e.g., long rests)
            if len(messy_chunk.notes) == 0 and len(clean_chunk.notes) == 0:
                continue
                
            # Tokenize the synchronized chunks
            messy_tokens = self.tokenizer(messy_chunk).ids[0]
            clean_tokens = self.tokenizer(clean_chunk).ids[0]
            
            pairs.append({
                "messy": torch.tensor(messy_tokens, dtype=torch.long),
                "clean": torch.tensor(clean_tokens, dtype=torch.long)
            })
            
        return pairs

    def _get_max_tick(self, score):
        # Helper to find the end of the song
        return max([note.end for note in score.notes]) if score.notes else 0

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        return self.data_pairs[idx]

# --- Let's test the microapp! ---
if __name__ == "__main__":
    # Point this to the files you dropped in your folder
    dataset = MidiCorrectionDataset("mistakes.mid", "clean.mid", measures_per_chunk=4)
    
    print(f"Successfully chopped track into {len(dataset)} synchronized chunks.")
    
    # Grab the first chunk to see what the model will get
    first_chunk = dataset[0]
    print(f"\nChunk 1 Messy Token Length: {len(first_chunk['messy'])}")
    print(f"Chunk 1 Clean Token Length: {len(first_chunk['clean'])}")
