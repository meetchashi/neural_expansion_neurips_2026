import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
    
    def forward(self, x):
        """
        Args:
            x: Shape `(batch_size, in_channels, img_size, img_size)`
        Returns:
            patch_embeddings: Shape `(batch_size, n_patches, embed_dim)`
        """
        x = self.proj(x)  # (batch_size, embed_dim, n_patches^0.5, n_patches^0.5)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x

class AttentionBlock(nn.Module):
    """Simplified multi-head self-attention block."""
    def __init__(self, embed_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Self-attention with skip connections and layer norm
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP with skip connection and layer norm
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    """Simplified Vision Transformer for small images."""
    def __init__(
        self,
        img_size=28,
        patch_size=7,
        in_channels=1,
        num_classes=10,
        embed_dim=64,
        depth=6,
        num_heads=4,
        dropout=0.1
    ):
        super().__init__()
        
        # Image patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(depth)
        ])
        
        # Layer norm and classifier head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize patch projections and position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize attention and MLP weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: Shape `(batch_size, channels, img_size, img_size)`
        Returns:
            x: Shape `(batch_size, num_classes)`
        """
        # Get patch embeddings
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)
        
        # Add class token and position embeddings
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, 1 + n_patches, embed_dim)
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply layer norm and take class token for classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Take only the class token
        
        # Classification head
        x = self.head(cls_token_final)
        
        return x
    
    def extract_features(self, x):
        """
        Extracts features from each stage of the Vision Transformer.
        Returns a list of tensors containing:
        - Original input 
        - Patch embeddings
        - Embeddings after position embedding
        - Outputs from each transformer block
        - Normalized class token (penultimate representation)
        - Final classification output
        """
        features = []
        
        # Store original input
        features.append(x)
        
        # Patch embedding
        x = self.patch_embed(x)
        features.append(x)
        
        # Add class token and position embeddings
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        features.append(x)
        
        # Apply transformer blocks and collect intermediate features
        for block in self.blocks:
            x = block(x)
            features.append(x)
        
        # Apply layer norm
        x = self.norm(x)
        features.append(x)
        
        # Extract class token (this is typically used as the image representation)
        cls_token_final = x[:, 0]
        features.append(cls_token_final)  # This is the penultimate layer representation
        
        # Final classification head
        output = self.head(cls_token_final)
        features.append(output)
        
        return features
