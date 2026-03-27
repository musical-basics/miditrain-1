import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from miditok import REMI, TokenizerConfig
from symusic import Score

# Import our new cognitive mapping engine (Single Time Signature version)
from STS_bootstrapper import run_full_pipeline

class MeasureAlignedMidiDataset(Dataset):
    def __init__(self, messy_path, clean_path, measures_per_chunk=4):
        # 1. Initialize Tokenizer (Left-Brain Grammar)
        self.tokenizer = REMI(TokenizerConfig(num_velocities=16, use_chords=False, use_programs=False))
        self.pad_token_id = self.tokenizer["PAD_None"]
        self.measures_per_chunk = measures_per_chunk
        
        # 2. Load the MIDI files
        self.messy_midi = Score(messy_path)
        self.clean_midi = Score(clean_path)
        
        # 3. Extract keyframes from the messy MIDI for the Bootstrapper
        # Convert symusic notes into our expected format: (Time_ms, [(pitch, octave, velocity)])
        print("Extracting physical performance data for cognitive mapping...")
        messy_keyframes = self._extract_keyframes(self.messy_midi)
        
        # 4. Run the Reverse Echolocation Bootstrapper!
        print("Bootstrapping metrical grid...")
        self.grid_anchors = run_full_pipeline(messy_keyframes, initial_tempo=500.0)
        
        # 5. Chop and align the data based on the extracted Measure Grid
        print(f"Chopping data into {measures_per_chunk}-measure chunks...")
        self.data_pairs = self._chop_and_tokenize()

    def _extract_keyframes(self, score):
        """Converts symusic Score into a timeline of active notes for the Bootstrapper."""
        # Note: symusic uses ticks. We convert ticks to milliseconds based on the score's tempo.
        # Assuming a default 120bpm (500ms per quarter) for raw unquantized MIDI.
        tpq = score.ticks_per_quarter
        tick_to_ms = 500.0 / tpq 
        
        # Map MIDI pitch class (0-11) to the interval names used by unified_trackerv2
        PC_TO_INTERVAL = {
            0: "1", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4",
            6: "#4", 7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7"
        }
        
        keyframes = []
        # We group notes by their onset time
        time_map = {}
        for track in score.tracks:
            for note in track.notes:
                time_ms = int(note.start * tick_to_ms)
                # Map pitch to the interval name the vector engine expects
                pitch_class = PC_TO_INTERVAL[note.pitch % 12]
                octave = note.pitch // 12
                
                if time_ms not in time_map:
                    time_map[time_ms] = []
                time_map[time_ms].append((pitch_class, octave, note.velocity))
                
        for t in sorted(time_map.keys()):
            keyframes.append((t, time_map[t]))
            
        return keyframes

    def _chop_and_tokenize(self):
        pairs = []
        tpq = self.messy_midi.ticks_per_quarter
        tick_to_ms = 500.0 / tpq 
        ms_to_tick = tpq / 500.0

        # Group our anchors by Measure
        measure_boundaries = {}
        for anchor in self.grid_anchors:
            if anchor.beat == 1: # We only care about Downbeats for chunking
                measure_boundaries[anchor.measure] = int(anchor.time_ms * ms_to_tick)

        measure_numbers = sorted(measure_boundaries.keys())
        
        # Slide through the song, grabbing X measures at a time
        for i in range(0, len(measure_numbers) - self.measures_per_chunk, self.measures_per_chunk):
            start_measure = measure_numbers[i]
            end_measure = measure_numbers[i + self.measures_per_chunk]
            
            start_tick = measure_boundaries[start_measure]
            end_tick = measure_boundaries[end_measure]
            
            # 1. Clip the chunks precisely on the mathematical measure boundaries
            messy_chunk = self.messy_midi.clip(start_tick, end_tick, clip_end=True)
            clean_chunk = self.clean_midi.clip(start_tick, end_tick, clip_end=True)
            
            # Skip empty chunks
            if not messy_chunk.tracks[0].notes and not clean_chunk.tracks[0].notes:
                continue
                
            # 2. Build the Right-Brain Context Mask (Duration * Velocity Heuristic)
            dissonance_mask = self._create_dissonance_mask(clean_chunk)
            
            # 3. Tokenize
            messy_tokens = self.tokenizer(messy_chunk)[0].ids
            clean_tokens = self.tokenizer(clean_chunk)[0].ids
            
            pairs.append({
                "messy": torch.tensor(messy_tokens, dtype=torch.long),
                "clean": torch.tensor(clean_tokens, dtype=torch.long),
                "mask": dissonance_mask # The context now travels with the sequence!
            })
            
        return pairs

    def _create_dissonance_mask(self, clean_chunk):
        """Right-brain heuristic: Determines harmony based on Note Duration & Velocity."""
        pitch_weights = {i: 0.0 for i in range(12)}
        
        for track in clean_chunk.tracks:
            for note in track.notes:
                # Structural Weight = Duration * Velocity
                weight = note.duration * (note.velocity / 127.0)
                pitch_weights[note.pitch % 12] += weight
                
        # Find the 5 pitches with the least structural weight (passing/out-of-key notes)
        sorted_pitches = sorted(pitch_weights.keys(), key=lambda p: pitch_weights[p])
        dissonant_pitch_classes = sorted_pitches[:5]
        
        vocab_size = len(self.tokenizer)
        mask = torch.zeros(vocab_size, dtype=torch.float32)
        
        for midi_pitch in range(128):
            if midi_pitch % 12 in dissonant_pitch_classes:
                event_string = f"Pitch_{midi_pitch}"
                if event_string in self.tokenizer.vocab:
                    token_id = self.tokenizer[event_string]
                    mask[token_id] = 1.0 # 1.0 = Apply harmony penalty here
                    
        return mask

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        return self.data_pairs[idx]


# --- Updated Collate Function ---
def pad_collate_fn(batch):
    messy_seqs = [item['messy'] for item in batch]
    clean_seqs = [item['clean'] for item in batch]
    masks = [item['mask'] for item in batch] # Grab the masks
    
    pad_token_id = 0 
    
    messy_padded = pad_sequence(messy_seqs, batch_first=True, padding_value=pad_token_id)
    clean_padded = pad_sequence(clean_seqs, batch_first=True, padding_value=pad_token_id)
    masks_stacked = torch.stack(masks) # Stack masks into a batch tensor
    
    return {
        "messy": messy_padded,
        "clean": clean_padded,
        "mask": masks_stacked # Pass to the GPU
    }

if __name__ == "__main__":
    # Test the new pipeline!
    dataset = MeasureAlignedMidiDataset("raw_midi_grieg_waltz.mid", "cleaned_midi_grieg_waltz.mid", measures_per_chunk=4)
    print(f"\nSuccessfully chopped track into {len(dataset)} metrically-aligned chunks.")