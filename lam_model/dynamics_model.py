import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class DynamicsTransformer(nn.Module):
    def __init__(self, n_tokens, action_dim, d_model=512, n_heads=8, n_layers=6, dropout=0.1, max_seq_len=256):
        super().__init__()
        
        # Embeddings
        self.token_embedding = nn.Embedding(n_tokens, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        
        # Frame feature projection
        self.frame_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.out = nn.Linear(d_model, n_tokens)
        
        self.dropout = nn.Dropout(dropout)
        self.n_tokens = n_tokens
        
    def _create_pos_embedding(self, max_len, d_model):
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        return pe
    
    def _create_causal_mask(self, size):
        """Create causal attention mask"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def forward(self, frame_tokens, action_embedding, mask=None):
        """
        Args:
            frame_tokens: Frame token indices [B, H*W] or [B, T, H*W]
            action_embedding: Action embedding from VQVAE [B, action_dim]
            mask: Optional attention mask [B, H*W]
        """
        # Handle both single frame and sequence inputs
        if frame_tokens.dim() == 2:
            frame_tokens = frame_tokens.unsqueeze(1)  # [B, 1, H*W]
        B, T, L = frame_tokens.shape
        
        # Embed frame tokens
        x = self.token_embedding(frame_tokens)  # [B, T, H*W, D]
        x = self.frame_proj(x)  # Project frame features
        
        # Project and add action embedding
        action_cond = self.action_proj(action_embedding)  # [B, D]
        action_cond = action_cond.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, D]
        action_cond = action_cond.expand(-1, T, L, -1)  # [B, T, H*W, D]
        x = x + action_cond
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :x.size(1)]
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
            
        # Project to token probabilities for next frame
        logits = self.out(x[:, -1])  # Use last timestep [B, H*W, n_tokens]
        
        return logits
    
    @torch.no_grad()
    def generate(self, frame_tokens, action_embedding, num_steps=10, temperature=1.0):
        """
        Generate next frame tokens using iterative refinement
        Args:
            frame_tokens: Frame token indices [B, H*W] or [B, T, H*W]
            action_embedding: Action embedding from VQVAE [B, action_dim]
            num_steps: Number of refinement steps
            temperature: Sampling temperature
        """
        # Handle both single frame and sequence inputs
        if frame_tokens.dim() == 2:
            frame_tokens = frame_tokens.unsqueeze(1)  # [B, 1, H*W]
        B, T, L = frame_tokens.shape
        
        # Initialize prediction with last frame tokens
        curr_tokens = frame_tokens[:, -1].clone()  # [B, H*W]
        
        for step in range(num_steps):
            # Calculate ratio of tokens to unmask
            ratio = 1.0 - (step + 1) / num_steps
            num_mask = int(L * ratio)
            
            # Create random mask
            mask = torch.ones((B, L), dtype=torch.bool, device=curr_tokens.device)
            for i in range(B):
                perm = torch.randperm(L, device=curr_tokens.device)
                mask[i, perm[:num_mask]] = 0
            
            # Get predictions
            logits = self(curr_tokens, action_embedding, mask)
            
            # Sample new tokens for masked positions
            probs = F.softmax(logits[~mask].view(-1, self.n_tokens) / temperature, dim=-1)
            new_tokens = torch.multinomial(probs, 1).view(-1)
            
            # Update masked positions
            curr_tokens[~mask] = new_tokens
            
        return curr_tokens

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with causal masking
        attended = self.attention(x, x, x, attn_mask=mask)[0]
        x = self.norm1(x + self.dropout(attended))
        
        # Feed-forward
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x 