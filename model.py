import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.pos_encoding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # Add positional encoding to the input embeddings
        pos = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)  # (batch_size, seq_len)
        embedded = self.pos_encoding(pos)  # (batch_size, seq_len, d_model)
        return x + embedded


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, dropout=0.1):
        super(TransformerLM, self).__init__()

        # Embedding layer with positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = LearnedPositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layer)

        # Output projection layer
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, padding_mask=None):
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)

        # Generate causal mask for autoregressive behavior
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device).bool()
        if padding_mask is not None:
            padding_mask = padding_mask.bool()

        # Transformer forward pass with masks
        x = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)  # (batch_size, seq_len, d_model)
        x = self.fc(x)  # (batch_size, seq_len, vocab_size)
        return x
