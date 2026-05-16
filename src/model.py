import torch
import torch.nn as nn

from config import (
    VOCAB_SIZE, MAX_SEQ_LEN, HIDDEN_DIM, NUM_LAYERS, NHEAD, DROPOUT, DEVICE,
)


class LayoutTransformer(nn.Module):

    def __init__(self, vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN,
                 d_model=HIDDEN_DIM, nhead=NHEAD, num_layers=NUM_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, x, padding_mask=None):
        batch_size, seq_len = x.shape
        device = x.device

        positions = torch.arange(seq_len, device=device)
        positions = positions.unsqueeze(0)
        positions = positions.expand(batch_size, -1)

        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        h = tok_emb + pos_emb
        h = self.drop(h)

        causal_mask = self._generate_causal_mask(seq_len, device)

        memory = torch.zeros(batch_size, 1, self.d_model, device=device)

        h = self.transformer(
            h, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )

        h = self.ln_f(h)
        logits = self.head(h)
        return logits


def create_model():
    model = LayoutTransformer().to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] LayoutTransformer: {num_params / 1e6:.1f}M parametre")
    return model
