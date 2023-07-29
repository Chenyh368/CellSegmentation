"""Pytorch Implementations of the layers used in the SCS model
"""

import torch
from torch import nn

class PatchEncoder(nn.Module):
    def __init__(self, input_shape, input_position_shape, projection_dim):
        super(PatchEncoder, self).__init__()
        self.projection = nn.Linear(input_shape, projection_dim)
        self.position_embedding = nn.Linear(input_position_shape, projection_dim)

    def forward(self, patch, position):
        return (self.projection(patch) + self.position_embedding(position))


class SCSTransformer(nn.Module):
    def __init__(self,class_num, input_shape, input_position_shape, projection_dim, num_heads, transformer_units, transformer_layers, mlp_head_units):
        super(SCSTransformer, self).__init__()
        self.patch_encoder = PatchEncoder(input_shape, input_position_shape, projection_dim)

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=projection_dim, nhead=num_heads,
                                                            dim_feedforward = transformer_units, dropout=0.1, layer_norm_eps=1e-6)

        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, transformer_layers)

        # Create a [batch_size, projection_dim] tensor.
        self.norm_cls = nn.LayerNorm(projection_dim, eps=1e-6)
        self.feature_cls = nn.Sequential(
            nn.Linear(projection_dim, mlp_head_units[0]),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_head_units[0], mlp_head_units[1]),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        # Classifier
        self.pos = nn.Linear(mlp_head_units[1], class_num)
        self.binary = nn.Linear(mlp_head_units[1], 1)

    def forward(self, inputs, inputs_positions):
        x = self.patch_encoder(inputs, inputs_positions)
        x = self.transformer_encoder(x)
        x = self.norm_cls(x)
        feature = self.feature_cls(self.norm_cls(x)[:,0,:])
        return {'pos': self.pos(feature), 'binary':self.binary(feature)}

