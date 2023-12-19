import torch
from typing import List


class MLP(torch.nn.Module):
    """Standard multilayer perceptron."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        structure: List[int] = [],
        adn_fn: torch.nn.Module = torch.nn.Identity,
    ):
        """
        Args:
            input_dim (int): input dimension.
            output_dim (int): output dimension.
            structure (List[int], optional): hidden layer structure. Should
                be a list of ints. Defaults to [].
            adn_fn (torch.nn.Module, optional): function that returns a
                torch.nn.Module that does activation/dropout/normalization.
                Should take as arguments the number of channels in a given
                layer. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.structure = structure
        self.adn_fn = adn_fn

        self.init_layers()

    def init_layers(self):
        """Initialises layers."""
        curr_in = self.input_dim
        ops = torch.nn.ModuleList([])
        if len(self.structure) > 0:
            curr_out = self.structure[0]
            for i in range(1, len(self.structure)):
                ops.append(torch.nn.Linear(curr_in, curr_out))
                ops.append(self.adn_fn(curr_out))
                curr_in = curr_out
                curr_out = self.structure[i]
            ops.append(torch.nn.Linear(curr_in, curr_out))
        else:
            curr_out = curr_in
        ops.append(self.adn_fn(curr_out))
        ops.append(torch.nn.Linear(curr_out, self.output_dim))
        self.op = torch.nn.Sequential(*ops)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects the input to have two or more dimensions.

        Args:
            X (torch.Tensor): tensor with shape [...,self.input_dim]

        Returns:
            torch.Tensor: tensor with shape [...,self.output_dim]
        """
        return self.op(X)