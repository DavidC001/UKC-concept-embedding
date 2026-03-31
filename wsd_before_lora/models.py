"""Model architectures for WSD classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """Projects sentence embeddings (768-dim) to concept space (500-dim)."""
    # 768 = mmbert
    # 4096 = qwen
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=500, dropout=0.1, hidden_dim2 = 1536):
        super().__init__()

        #best
        self.net = nn.Sequential(
        #     # nn.Linear(input_dim, hidden_dim),
        #     # # nn.GroupNorm(num_groups=32, num_channels=hidden_dim),
        #     # nn.LayerNorm(hidden_dim),
        #     # nn.GELU(),
        #     # nn.Dropout(dropout),
        #     # nn.Linear(hidden_dim, output_dim)


            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Linear(hidden_dim, output_dim)
        )

        # 20260228_120726_lr5e-05_wd0.001_bt32_t1_m0.9_AdamW_mmbert
        # self.net = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     # nn.GroupNorm(num_groups=32, num_channels=hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, output_dim)
        # )

        # 20260228_112407_lr5e-05_wd0.001_bt32_t1_m0.9_AdamW_mmbert
        # self.net = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, hidden_dim2),
        #     nn.LayerNorm(hidden_dim2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim2, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, output_dim)
        # )
    
    def forward(self, x):
        """Args: x (batch_size, input_dim). Returns: (batch_size, output_dim)"""
        return self.net(x)


class ConceptClassifier(nn.Module):
    """WSD classifier with MLP projection and concept embedding matching."""
    
    # 768 = mmbert
    # 4096 = qwen
    def __init__(self, concept_embeddings, input_dim=768, hidden_dim=1024, 
                 output_dim=500, dropout=0.1, temperature=0.1, concept_mask=None, baseline=False, baseline_type='linear'):
        """
        Args:
            concept_embeddings: (num_concepts, 500) torch tensor
            input_dim: dimension of input embeddings
            hidden_dim: hidden dimension of MLP
            output_dim: projection output dimension
            dropout: dropout probability
            temperature: temperature for scaling logits
            concept_mask: (num_concepts,) binary mask, 1 for concepts in training set, 0 otherwise
            baseline: if True, concept embeddings are not used and the model is a simple classifier
            baseline_type: if baseline is True, specifies the type of baseline ('linear' or 'random')
        """
        super().__init__()
        self.projector = Projector(input_dim, hidden_dim, output_dim, dropout)
        # self.register_buffer('concept_embeddings', F.normalize(concept_embeddings, dim=-1))
        self.baseline = baseline
        self.baseline_type = baseline_type
        if not baseline:
            self.concept_embeddings = nn.Parameter(concept_embeddings, requires_grad=False)  # (num_concepts, output_dim)
        else:
            num_concepts = concept_embeddings.shape[0]
            if baseline_type == 'linear':
                # baseline with learnable classifier layer (no concept embedding)
                self.concept_embeddings = nn.Linear(output_dim, num_concepts, bias=False)  # learnable classifier layer
            elif baseline_type == 'random':
                # baseline with fixed random classifier layer (no concept embedding)
                self.concept_embeddings = nn.Parameter(torch.randn(num_concepts, output_dim), requires_grad=False)  # fixed random classifier layer

        self.temperature = float(temperature)
        # self.scale = nn.Parameter(torch.tensor(5.0))

        
        # Register concept mask as buffer (not trainable, moves with model to device)
        if concept_mask is not None:
            self.register_buffer('concept_mask', concept_mask)
        else:
            self.concept_mask = None
    
    # DOT proeduct version
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)  embeddings
        Returns:
            logits: (batch_size, num_concepts)
        """
        projected = self.projector(x)  # (batch_size, output_dim)

        # projected = F.normalize(projected, dim=-1)    # unit norm
        
        # projected = F.normalize(self.projector(x), dim=-1)

        # concept_emb = F.normalize(self.concept_embeddings, dim=-1)
        
        # if it's a linear classifier baseline, use the linear layer; otherwise use the concept embeddings
        if self.baseline and isinstance(self.concept_embeddings, nn.Linear):
            logits = self.concept_embeddings(projected)  # (batch_size, num_concepts)
        else:
            # use dot product between projected embeddings and concept embeddings as logits
            # logits = torch.matmul(projected, self.concept_embeddings.t()) 
            # use cosine similarity (dot product of normalized vectors) as logits
            norm_concept_emb = F.normalize(self.concept_embeddings, dim=-1)
            norm_projected = F.normalize(projected, dim=-1)
            logits = torch.matmul(norm_projected, norm_concept_emb.t())  # (batch_size, num_concepts)
        
        logits = logits / self.temperature
        
        # self.scale = nn.Parameter(torch.tensor(10.0))

        # Apply concept mask: set logits to very negative for concepts not in training set
        # if self.concept_mask is not None:
        #     logits = logits * self.concept_mask.unsqueeze(0) + (1.0 - self.concept_mask.unsqueeze(0)) * (-1e9)
        
        return logits


    # euclidean distance
    # def forward(self, x):
    #     projected = self.projector(x)

    #     dist = torch.cdist(projected, self.concept_embeddings)

    #     logits = -dist / self.temperature

    #     return logits

    # Square distance
    # def forward(self, x):
        
    #     projected = self.projector(x)

       
    #     diff = projected.unsqueeze(1) - self.concept_embeddings.unsqueeze(0)
    #     dist = torch.sum(diff ** 2, dim=-1)

    #     logits = -dist / self.temperature

    #     return logits