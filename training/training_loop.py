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
    patience=5,
    optimizer_name="adam"
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ======================================================
    # Convert Data to Tensors (NaN-SAFE)
    # ======================================================
    X_train_tensor = torch.tensor(
        X_train.values, dtype=torch.float32, device=device
    )

    X_val_tensor = torch.tensor(
        X_val.values, dtype=torch.float32, device=device
    )

    # ----------- Target conversion -----------
    if problem_type == "multi_class_classification":
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long, device=device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.long, device=device)

    else:
        y_train_tensor = torch.tensor(
            y_train.values, dtype=torch.float32, device=device
        ).view(-1, 1)

        y_val_tensor = torch.tensor(
            y_val.values, dtype=torch.float32, device=device
        ).view(-1, 1)

    # ======================================================
    # DataLoader
    # ======================================================
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    # ======================================================
    # Loss + Optimizer
    # ======================================================
    criterion = get_loss_function(problem_type)
    optimizer = get_optimizer(model, lr=lr, optimizer_name=optimizer_name)

    # Gradient clipping threshold
    max_grad_norm = 5.0

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    # ======================================================
    # TRAINING LOOP
    # ======================================================
    for epoch in range(epochs):

        model.train()
        total_loss = 0.0

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()

            preds = model(batch_X)

            # ----------- NaN / Inf detector -----------
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                print("\n❌ NaN detected in predictions! Stopping batch...")
                return model

            # ----------- Compute loss safely -----------
            try:
                loss = criterion(preds, batch_y)
            except Exception as e:
                print("\n❌ Loss computation error:", e)
                print("Pred shape:", preds.shape)
                print("Target shape:", batch_y.shape)
                raise e

            # ----------- Loss explosion fix -----------
            if torch.isnan(loss) or torch.isinf(loss):
                print("\n❌ NaN/Inf loss encountered — skipping batch")
                continue

            # ----------- Backprop -----------
            loss.backward()

            # ----------- Gradient clipping -----------
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)

        # ======================================================
        # VALIDATION
        # ======================================================
        model.eval()
        with torch.no_grad():

            val_preds = model(X_val_tensor)

            # guard
            if torch.isnan(val_preds).any() or torch.isinf(val_preds).any():
                print("❌ NaN in validation predictions")
                break

            val_loss = criterion(val_preds, y_val_tensor).item()

        # ----------- Logging -----------
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # ======================================================
        # EARLY STOPPING
        # ======================================================
        min_delta = 1e-4

        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"⏹ Early stopping triggered at epoch {epoch+1}")
            break

    # ======================================================
    # Restore BEST model
    # ======================================================
    if best_state is not None:
        model.load_state_dict(best_state)

    return model