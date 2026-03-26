import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    # Standard PyTorch boilerplate to give the model a sense of time/order
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MidiCorrector(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, sos_token_id=1, eos_token_id=2):
        super().__init__()
        self.d_model = d_model
        
        # We need to know these special tokens for the generate() loop
        self.sos_token_id = sos_token_id # Start of Sequence
        self.eos_token_id = eos_token_id # End of Sequence
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model) 
        
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, 
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    # METHOD 1: Training (Solely based on user-provided paired data)
    def forward(self, messy_src, clean_tgt):
        src_emb = self.pos_encoder(self.embedding(messy_src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(clean_tgt) * math.sqrt(self.d_model))
        
        # Prevent the model from looking ahead at future user-provided data
        tgt_mask = self.transformer.generate_square_subsequent_mask(clean_tgt.size(1)).to(clean_tgt.device)
        
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc_out(out)

    # METHOD 2: Inference (Generating completely new data token-by-token)
    def generate(self, messy_src, max_length=1000):
        device = messy_src.device
        batch_size = messy_src.size(0)
        
        # We must encode the messy input once to give the decoder its context
        src_emb = self.pos_encoder(self.embedding(messy_src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb)
        
        # Start the clean sequence with the [SOS] token
        generated_tokens = torch.full((batch_size, 1), self.sos_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_length):
            # Embed the tokens we have generated *so far*
            tgt_emb = self.pos_encoder(self.embedding(generated_tokens) * math.sqrt(self.d_model))
            tgt_mask = self.transformer.generate_square_subsequent_mask(generated_tokens.size(1)).to(device)
            
            # Pass through the decoder using the encoder's memory
            out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            
            # Get the logits for the very last token generated
            next_token_logits = self.fc_out(out[:, -1, :])
            
            # Pick the most likely next token (Greedy decoding)
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            # Append it to our running sequence
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            
            # If the model outputs an [EOS] token, it thinks the song segment is done
            if (next_token == self.eos_token_id).all():
                break
                
        return generated_tokens
