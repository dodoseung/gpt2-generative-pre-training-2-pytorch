# Radford, Alec, et al. "Language models are unsupervised multitask learners." OpenAI blog 1.8 (2019): 9.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Transformer
class GPT2(nn.Module):
    def __init__(self, num_decoder_layer=12, d_model=768, num_heads=12, d_ff=3072, dropout=0.1, 
                 trg_pad_idx=0, vocab_size=50257, max_seq_len=1024, device="cpu"):
        super(GPT2, self).__init__()
        # Device
        self.device = device

        # Token masks
        self.trg_pad_idx = trg_pad_idx

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)

        #decoder
        self.decoder = Decoder(num_decoder_layer, d_model, num_heads, d_ff, dropout)
        
        # Output layer
        self.out_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, trg):
        # Get mask
        trg_mask = self.look_ahead_mask(trg)

        # Token embedding
        pos = torch.arange(0, trg.size(-1)).unsqueeze(0)
        trg_emb = self.token_embedding(trg) + self.positional_embedding(pos)
        trg_emb = self.dropout(trg_emb)

        #Decoder
        decoder_out = self.decoder(trg_emb, trg_mask)
        
        # Transform to character
        out = self.out_layer(decoder_out)
        
        return out
    
    # Set the look ahead mask
    # seq: (batch, seq_len)
    # mask: (batch, 1, seq_len, seq_len)
    # Pad -> True
    def look_ahead_mask(self, seq):
        # Set the look ahead mask
        # (batch, seq_len, seq_len)
        seq_len = seq.shape[1]
        mask = torch.ones(seq_len, seq_len)
        mask = torch.tril(mask)
        mask = mask.bool().to(self.device)

        # Set the padding mask
        # (batch, 1, 1, seq_len)
        pad_mask = (seq != self.trg_pad_idx)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)
        
        # Merge the masks
        mask = mask & pad_mask

        return mask

    # Decoder with the lasf fc layer
    # Prediction
    def predict(self, trg, trg_mask):
        # Decoder
        decoder_out = self.decoder(trg, trg_mask)
        
        # Transform to character
        out = self.out_layer(decoder_out)

        return out

# Decoder
class Decoder(nn.Module):
    def __init__(self, num_layer, d_model, num_heads, d_ff, dropout):
        super(Decoder, self).__init__()
        # Decoder layers
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layer)]) 

    def forward(self, trg, trg_mask):
        # Encoder layers
        for layer in self.layers:
            trg = layer(trg, trg_mask)
            
        return trg

# Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.position_wise_feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-5)

        self.dropout = dropout

    def forward(self, trg, trg_mask):
        # Masked multi head attention
        out = self.layer_norm1(trg)
        out = self.masked_multi_head_attention(out, out, out, trg_mask)
        residual = out
        
        # Position wise feed foward
        out = self.layer_norm2(out)
        out = self.position_wise_feed_forward(out, self.dropout)
        out = residual + out

        return out

# Multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_model // self.num_heads
        
        # Define w_q, w_k, w_v, w_o
        self.weight_q = nn.Linear(self.d_model, self.d_model)
        self.weight_k = nn.Linear(self.d_model, self.d_model)
        self.weight_v = nn.Linear(self.d_model, self.d_model)
        self.weight_o = nn.Linear(self.d_model, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        # Batch size
        batch_size = query.shape[0]
        
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        query = self.weight_q(query)
        key = self.weight_k(key)
        value = self.weight_v(value)

        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k)
        query = query.view(batch_size, -1, self.num_heads, self.d_k)
        key = key.view(batch_size, -1, self.num_heads, self.d_k)
        value = value.view(batch_size, -1, self.num_heads, self.d_k)
        
        # (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = torch.transpose(query, 1, 2)
        key = torch.transpose(key, 1, 2)
        value = torch.transpose(value, 1, 2)
        
        # Get the scaled attention
        # (batch, h, query_len, d_k) -> (batch, query_len, h, d_k)
        scaled_attention = self.scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = torch.transpose(scaled_attention, 1, 2).contiguous()

        # Concat the splitted attentions
        # (batch, query_len, h, d_k) -> (batch, query_len, d_model)
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)
        
        # Get the multi head attention
        # (batch, query_len, d_model) -> (batch, query_len, d_model)
        multihead_attention = self.weight_o(concat_attention)
        
        return multihead_attention
    
    # Query, key, and value size: (batch, num_heads, seq_len, d_k)
    # Mask size(optional): (batch, 1, seq_len, seq_len)   
    def scaled_dot_product_attention(self, query, key, value, mask):
        # Get the q matmul k_t
        # (batch, h, query_len, d_k) dot (batch, h, d_k, key_len)
        # -> (batch, h, query_len, key_len)
        attention_score = torch.matmul(query, torch.transpose(key, -2, -1))

        # Get the attention score
        d_k = query.size(-1)
        attention_score = attention_score / math.sqrt(d_k)

        # Get the attention wights
        attention_score = attention_score.masked_fill(mask==0, -1e10) if mask is not None else attention_score
        attention_weights = F.softmax(attention_score, dim=-1, dtype=torch.float)

        # Get the attention value
        # (batch, h, query_len, key_len) -> (batch, h, query_len, d_k)
        attention_value = torch.matmul(attention_weights, value)
        
        return attention_value

# Position wise feed forward
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = F.gelu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        
        return out