import torch
from torch.utils.data import Dataset, DataLoader
from miditok import REMI, TokenizerConfig
from symusic import Score

class MidiCorrectionDataset(Dataset):
    def __init__(self, messy_path, clean_path, seconds_per_chunk=30):
        # 1. Initialize Tokenizer
        self.tokenizer = REMI(TokenizerConfig(num_velocities=16, use_chords=False, use_programs=False))
        self.pad_token_id = self.tokenizer["PAD_None"]
        
        # 2. Load the MIDI files
        self.messy_midi = Score(messy_path)
        self.clean_midi = Score(clean_path)
        
        # 3. Calculate time windows (Ticks)
        # Assuming roughly 120 BPM mapping: 1 beat = 0.5s => 1 second = 2 quarter notes.
        tpq = self.messy_midi.ticks_per_quarter
        self.window_size_ticks = int(tpq * 2 * seconds_per_chunk)
        
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
            num_messy_notes = sum(len(t.notes) for t in messy_chunk.tracks)
            num_clean_notes = sum(len(t.notes) for t in clean_chunk.tracks)
            if num_messy_notes == 0 and num_clean_notes == 0:
                continue
                
            # Tokenize the synchronized chunks
            messy_tokens = self.tokenizer(messy_chunk)[0].ids
            clean_tokens = self.tokenizer(clean_chunk)[0].ids
            
            pairs.append({
                "messy": torch.tensor(messy_tokens, dtype=torch.long),
                "clean": torch.tensor(clean_tokens, dtype=torch.long)
            })
            
        return pairs

    def _get_max_tick(self, score):
        # Helper to find the end of the song
        if not score.tracks:
            return 0
        return max([note.end for track in score.tracks for note in track.notes]) if any(track.notes for track in score.tracks) else 0

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        return self.data_pairs[idx]

from torch.nn.utils.rnn import pad_sequence

def pad_collate_fn(batch):
    # Extract the 'messy' and 'clean' sequences from the batch
    messy_seqs = [item['messy'] for item in batch]
    clean_seqs = [item['clean'] for item in batch]
    
    # Pad the sequences with a designated padding token (usually 0)
    # batch_first=True makes the tensor shape (batch_size, sequence_length)
    pad_token_id = 0 # In MidiTok, PAD is usually 0, check your vocab to be sure
    
    messy_padded = pad_sequence(messy_seqs, batch_first=True, padding_value=pad_token_id)
    clean_padded = pad_sequence(clean_seqs, batch_first=True, padding_value=pad_token_id)
    
    return {
        "messy": messy_padded,
        "clean": clean_padded
    }

# --- Let's test the microapp! ---
if __name__ == "__main__":
    # Point this to the files you dropped in your folder
    dataset = MidiCorrectionDataset("raw_midi_grieg_waltz.mid", "cleaned_midi_grieg_waltz.mid", seconds_per_chunk=30)
    
    # dataloader = DataLoader(dataset, batch_size=8, collate_fn=pad_collate_fn)
    print(f"Successfully chopped track into {len(dataset)} synchronized chunks.")
    
    # Grab the first chunk to see what the model will get
    first_chunk = dataset[0]
    print(f"\nChunk 1 Messy Token Length: {len(first_chunk['messy'])}")
    print(f"Chunk 1 Clean Token Length: {len(first_chunk['clean'])}")
