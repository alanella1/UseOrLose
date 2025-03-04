import torch
import torch.nn as nn

INPUT_DIM = 561
OUTPUT_DIM = 6


class UseOrLose(nn.Module):
    def __init__(self, hidden_dim=256, kill_threshold=1e-2):
        super(UseOrLose, self).__init__()
        self.kill_threshold = kill_threshold

        self.linear1 = nn.Linear(INPUT_DIM, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.linear3 = nn.Linear(hidden_dim, OUTPUT_DIM)

        self.relu = nn.ReLU()

        # Register masks for each Linear layer
        self.masks = {}
        for name, layer in self.named_children():
            if isinstance(layer, nn.Linear):
                mask = torch.ones_like(layer.weight, device='cuda')
                self.register_buffer(f"mask_{name}", mask)
                self.masks[name] = mask

    def forward(self, x):
        # masks
        self.linear1.weight.data *= self.masks["linear1"]
        self.linear2.weight.data *= self.masks["linear2"]
        self.linear3.weight.data *= self.masks["linear3"]

        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        return x

    def kill_small_weights(self, writer=None, global_step=None):
        with torch.no_grad():
            for name, layer in self.named_children():
                killed_weights = {}
                total_weights = {}
                if isinstance(layer, nn.Linear):
                    mask = self.masks[name]
                    small_weights = layer.weight.abs() < self.kill_threshold

                    killed_count = small_weights.sum().item()
                    total_count = layer.weight.numel()

                    mask[small_weights] = 0
                    layer.weight[small_weights] = 0

                    killed_weights[name] = killed_count
                    total_weights[name] = total_count

                    # Log sparsity to TensorBoard if writer is available
                    if writer is not None and global_step is not None:
                        writer.add_scalar(f"Sparsity/{name}_killed", killed_count, global_step)
                        writer.add_scalar(f"Sparsity/{name}_percent", (killed_count / total_count) * 100, global_step)
