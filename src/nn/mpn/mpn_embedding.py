from typing import Dict, Optional, Any

import torch
from torch import nn
from torch.nn import Parameter, init

from src.nn import MLP


class EmptyLinearLayer(nn.Module):
    """
    "Linear" layer that expects an input_size of 0.
    This layer is meant to be called with an empty input of shape (batch_size, 0) and will return
    batch_size repeats of its (trained) bias as an array of shape (batch_size, out_features)
    """
    def __init__(self, out_features: int) -> None:
        super(EmptyLinearLayer, self).__init__()
        self.bias = Parameter(torch.empty(out_features))
        init.uniform_(self.bias, -1, 1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor: An empty graph of size (batch_size, 0)

        Returns: An output graph of shape (batch_size, self.out_features) containing #batch_size copies of the bias,
          or 0 if no bias is provided
        """
        return self.bias.unsqueeze(0).repeat(tensor.shape[0], 1)


class MPNEmbedding(nn.Module):
    """
    Linear Embedding module. Essentially a learned matrix multiplication (and a bias) to make input dimension of tokens
    compatible with the "main" architecture that uses them.
    """
    def __init__(self, in_features: int, latent_dimension: int, embedding_config: Optional[Dict[str, Any]]):
        super().__init__()
        if in_features == 0:
            # If there are no input features to embed, we instead learn constant initial values.
            # In this case, we also do not make use of the embedding configuration
            self.embedding = EmptyLinearLayer(out_features=latent_dimension)
            self.is_constant_embedding = True
        else:
            if embedding_config is None:
                # Linearly embed if not embedding config is provided
                self.embedding = nn.Linear(in_features=in_features, out_features=latent_dimension)
            else:
                # Otherwise, we embed the features with an MLP
                mlp_config = embedding_config.get("mlp")
                self.embedding = MLP(in_features=in_features, latent_dimension=latent_dimension,
                                     config=mlp_config)
            self.is_constant_embedding = False

        self.out_features = latent_dimension
        self.in_features = in_features

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Embed the input graph using an MLP.
        If the input dimension for this embedding is 0, we instead use a learned constant placeholder value.
        """
        return self.embedding(tensor)
