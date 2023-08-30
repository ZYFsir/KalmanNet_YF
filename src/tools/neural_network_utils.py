import torch.nn as nn


def create_fully_connected(input_dim, output_dim, num_hidden_layers, hidden_dim=100, activation=nn.ReLU()):
    layers = []
    if num_hidden_layers <=0:
        hidden_dim = output_dim
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(activation)

    # Hidden layers
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation)

    # Output layer
    layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers)

if __name__ == "__main__":
    # Example usage
    input_dim = 64
    output_dim = 32
    num_hidden_layers = 2
    hidden_dim = 128

    fully_connected_net = create_fully_connected(input_dim, output_dim, num_hidden_layers, hidden_dim)
