import torch
from torch.utils.data import DataLoader, TensorDataset

from training.utils_training import get_loss_function, get_optimizer


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    problem_type,
    epochs=100,
    batch_size=32,
    lr=0.001,
    patience=10
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Convert to tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val.values, dtype=torch.float32).to(device)

    if problem_type == "multi_class_classification":
        y_train = torch.tensor(y_train.values, dtype=torch.long).to(device)
        y_val = torch.tensor(y_val.values, dtype=torch.long).to(device)
    else:
        y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
        y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    criterion = get_loss_function(problem_type)
    optimizer = get_optimizer(model, lr)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):

        # -------- TRAIN --------
        model.train()
        running_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # -------- VALIDATION --------
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()

        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        # -------- EARLY STOPPING --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses
