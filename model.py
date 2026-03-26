import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # pe shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # unsqueeze to match batch_size dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MidiCorrector(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        
        # 1. Embedding layer: Turns token IDs into vectors of size 'd_model'
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # We need positional encoding so the model knows the order of notes
        # (A standard component in Transformers, usually implemented as a separate small class)
        self.pos_encoder = PositionalEncoding(d_model) 
        
        # 2. The core Transformer
        # This handles both the Encoder (messy input) and Decoder (clean output)
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        
        # 3. Final Output Layer: Maps the Transformer's output back to our vocabulary size
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, messy_src, clean_tgt):
        # Embed and add positional information
        src_emb = self.pos_encoder(self.embedding(messy_src))
        tgt_emb = self.pos_encoder(self.embedding(clean_tgt))
        
        # Create a mask for the target sequence so the model can't "look ahead" and cheat
        tgt_mask = self.transformer.generate_square_subsequent_mask(clean_tgt.size(1)).to(clean_tgt.device)
        
        # Pass through the Transformer
        out = self.transformer(
            src_emb, 
            tgt_emb, 
            tgt_mask=tgt_mask
        )
        
        # Project back to vocabulary token probabilities
        logits = self.fc_out(out)
        return logits
