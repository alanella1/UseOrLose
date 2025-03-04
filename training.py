import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from regular_model import SimpleFeedForward
from use_or_lose_model import UseOrLose
from dataloader import get_dataloader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
base_name = 'debug'
batch_size = 128
learning_rate = 1e-3
num_epochs = 20
decay_value = 0.999
use_or_lose = True
kill_threshold = 1e-4
# tensorboard
logger_folder = 'loggin_stuff'
writer = SummaryWriter(log_dir=os.path.join(logger_folder, base_name), flush_secs=1)
# Load data
train_loader = get_dataloader(batch_size=batch_size, train=True)
test_loader = get_dataloader(batch_size=batch_size, train=False)

if use_or_lose:
    model = UseOrLose(kill_threshold=kill_threshold).to(device)
else:
    model = SimpleFeedForward().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def evaluate(model, test_loader, criterion, epoch, batch_idx):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    writer.add_scalar("Test Loss", avg_loss, epoch * len(train_loader) + batch_idx)
    return avg_loss


# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward
        loss.backward()

        # 0 grads
        with torch.no_grad():
            for name, layer in model.named_children():
                if isinstance(layer, nn.Linear):
                    mask = model.masks[name]
                    if layer.weight.grad is not None:
                        layer.weight.grad *= mask
        # step
        optimizer.step()
        global_step = epoch * len(train_loader) + batch_idx

        # Decay weights
        with torch.no_grad():
            model.linear1.weight.data *= decay_value
            model.linear2.weight.data *= decay_value
            model.linear3.weight.data *= decay_value

        # Kill dead neurons
        if use_or_lose:
            model.kill_small_weights(writer, global_step)

        writer.add_scalar("Train Loss", loss.item(), global_step)

        # Log test loss
        if batch_idx % 10 == 0:
            test_loss = evaluate(model, test_loader, criterion, epoch, batch_idx)

        # Log layer-wise absolute weights and gradient updates
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in ["linear1", "linear2", "linear3"]) and "weight" in name:
                layer_name = name.replace(".weight", "")

                # absolute  weight value
                avg_weight = param.data.abs().mean().item()
                writer.add_scalar(f"Weights/{layer_name}", avg_weight, global_step)

                #  absolute average gradient update
                if param.grad is not None:
                    lr = optimizer.param_groups[0]["lr"]
                    avg_grad_update = (param.grad.abs() * lr).mean().item()
                    # writer.add_scalar(f"GradUpdates/{layer_name}", avg_grad_update, global_step)
                    # writer.add_scalar(f"GradRatio/{layer_name}", (avg_grad_update / avg_weight), global_step)

    print(f"Epoch [{epoch + 1}/{num_epochs}] completed.")

writer.close()
