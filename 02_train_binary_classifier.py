#!/usr/bin/env python3
"""
Train a simple CPU-only PyTorch Lightning classifier that maps binary inputs
(input_1..input_N) to the integer class in `target`.

Assumes two CSVs made by your generator:
    training_data/train.csv
    training_data/val.csv
"""

import argparse
from pathlib import Path
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
# ExecuTorch export + runtime
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime, Verification


SEED = 112


class BinaryCSVDataset(Dataset):
    """
    Minimal dataset that reads a CSV with columns:
        input_1..input_N, target
    """
    
    def __init__(self, csv_path: Path):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)
        
        # Find all columns that start with "input_" and sort them by number
        input_columns = []
        for column in df.columns:
            if column.startswith("input_"):
                input_columns.append(column)
        
        # Sort columns by the number after "input_" (e.g., input_1, input_2, etc.)
        self.feature_cols = sorted(
            input_columns, 
            key=lambda x: int(x.split("_")[1])
        )
        
        # Check that the CSV has a 'target' column
        if "target" not in df.columns:
            raise ValueError("CSV must contain a 'target' column.")
        
        # Extract features (X) and convert to numpy array
        self.X = df[self.feature_cols].to_numpy(dtype=np.float32)
        
        # Extract targets (y) and convert to numpy array
        self.y = df["target"].to_numpy(dtype=np.int64)
        
        # Store dataset properties
        self.num_features = self.X.shape[1]
        # binary includes 0, so we need to add 1 more to the class count
        self.num_classes = int(self.y.max()) + 1
        self.dataset_length = len(df)
        return
    
    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, idx):
        # Convert numpy array to PyTorch tensor
        x = torch.from_numpy(self.X[idx])
        
        # Convert target to PyTorch tensor
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

class BinaryClassifier(pl.LightningModule):
    """
    Tiny MLP classifier for binary-to-integer mapping.
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_sizes=(32, 32), lr=1e-3, weight_decay=0.0):
        super().__init__()
        # automatically save hyperparameters to a self.hparams namespace
        self.save_hyperparameters()
        
        layers = []
        in_dim = input_dim
        for layer_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, layer_size))
            layers.append(nn.ReLU())
            in_dim = layer_size  # roll forward the input dimension
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)
        
        self.loss_fn = nn.CrossEntropyLoss()
        return
    
    def forward(self, x):
        return self.net(x)
    
    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "val")
        return loss
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(),
                            lr=self.hparams.lr,
                            weight_decay=self.hparams.weight_decay)
        return opt


def parse_hidden_layer_sizes(s: str):
    """
    Parse a comma-separated string like '32,32' into a tuple of ints.
    """
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    sizes = tuple(int(p) for p in parts)
    return sizes


def main():
    parser = argparse.ArgumentParser(description="Train a simple Lightning classifier on binary data.")
    parser.add_argument("--data_dir", type=str, default="training_data", help="Directory containing train.csv and val.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_layer_sizes", type=str, default="32,32", 
                        help="Comma-separated hidden sizes, e.g. '32,32' This determines the MLP architecture.")
    parser.add_argument("--num_workers", type=int, default=0, help="Use 0 for maximum portability across OSes.")
    args = parser.parse_args()
    
    pl.seed_everything(SEED, workers=True)
    
    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Expected {train_csv} and {val_csv} to exist.")
    
    train_ds = BinaryCSVDataset(train_csv)
    val_ds = BinaryCSVDataset(val_csv)
    
    # Validate consistency between splits
    if train_ds.num_features != val_ds.num_features:
        raise ValueError("Train and val feature dimensions differ.")
    if train_ds.num_classes != val_ds.num_classes:
        raise ValueError("Train and val num_classes differ.")
    
    input_dim = train_ds.num_features
    num_classes = train_ds.num_classes
    
    model = BinaryClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_sizes=parse_hidden_layer_sizes(args.hidden_layer_sizes),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    
    logger = CSVLogger(save_dir="lightning_logs", name="binary_clf")
    
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:02d}-{val_loss:.3f}",
    )
    
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
        enable_checkpointing=True,
        deterministic=True,
        logger=logger,
        callbacks=[ckpt_cb],
    )
    
    start_time = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    end_time = time.time()
    time_hours = int((end_time - start_time) // 3600)
    time_minutes = int(((end_time - start_time) % 3600) // 60)
    time_seconds = ((end_time - start_time) % 3600) % 60
    print(f"\nTotal time (HH:MM:SS.ss): {time_hours:02d}:{time_minutes:02d}:{time_seconds:05.2f}")
    
    
    
    ################################ METRICS PLOTTING ################################
    # Plot train/val loss from the CSVLogger
    metrics_csv = Path(logger.log_dir) / "metrics.csv"
    dfm = pd.read_csv(metrics_csv)
    
    # Keep only epoch-level entries for clean curves
    # (we logged with on_epoch=True, on_step=False)
    dfm = dfm.sort_values(["epoch"]).reset_index(drop=True)
    
    train_curve = dfm[~dfm["train_loss"].isna()][["epoch", "train_loss"]].drop_duplicates(subset=["epoch"])
    val_curve   = dfm[~dfm["val_loss"].isna()][["epoch", "val_loss"]].drop_duplicates(subset=["epoch"])
    
    plt.figure()
    if not train_curve.empty:
        plt.plot(train_curve["epoch"], train_curve["train_loss"], label="train_loss")
    if not val_curve.empty:
        plt.plot(val_curve["epoch"], val_curve["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_png = Path(logger.log_dir) / "loss_curves.png"
    plt.savefig(out_png, dpi=150)
    plt.savefig("loss_curves.png", dpi=150)
    print("Saved loss plot to: loss_curves.png")
    plt.close()
    
    # plot train/val accuracy
    train_acc_curve = dfm[~dfm["train_acc"].isna()][["epoch", "train_acc"]].drop_duplicates(subset=["epoch"])
    val_acc_curve   = dfm[~dfm["val_acc"].isna()][["epoch", "val_acc"]].drop_duplicates(subset=["epoch"])
    
    plt.figure()
    if not train_acc_curve.empty:
        plt.plot(train_acc_curve["epoch"], train_acc_curve["train_acc"], label="train_acc")
    if not val_acc_curve.empty:
        plt.plot(val_acc_curve["epoch"], val_acc_curve["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    acc_png = Path(logger.log_dir) / "acc_curves.png"
    plt.savefig(acc_png, dpi=150)
    plt.savefig("acc_curves.png", dpi=150)
    print("Saved accuracy plot to: acc_curves.png")
    plt.close()
    ################################ METRICS PLOTTING ################################
    
    
    ################################ LOAD BEST MODEL ################################
    # Reload the best model checkpoint found during training
    best_checkpoint_path = ckpt_cb.best_model_path
    print(f"\nBest model checkpoint path: {best_checkpoint_path}")
    best_model = BinaryClassifier.load_from_checkpoint(best_checkpoint_path)
    best_model.eval()  # put the model in eval mode to disable dropout, etc.
    # also make a copy of the best model in the main directory for easy access
    best_model_path = f"best_model_{best_checkpoint_path.split('/')[-1]}"  # keep the metrics in the filename
    shutil.copy(best_checkpoint_path, best_model_path)
    print(f"Copied best model checkpoint to: {best_model_path}")
    ################################ LOAD BEST MODEL ################################
    
    
    ################################ MODEL EVALUATION ################################
    # Quick evaluation of the model by showing a few predictions from the val split
    best_model.eval()
    with torch.no_grad():
        print("\nEval predictions on first 10 validation rows:")
        x_demo = torch.from_numpy(val_ds.X[:10])
        y_true = val_ds.y[:10]
        logits = best_model(x_demo)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        for i in range(len(y_true)):
            bits = " ".join(str(int(b)) for b in x_demo[i].numpy().tolist())
            correct = "✓" if y_true[i] == y_pred[i] else "✗"
            print(f"({i})  x=[{bits}]  |  true --> {int(y_true[i]):>2}  {int(y_pred[i]):<2} <-- pred  |  {correct}")
    ################################ MODEL EVALUATION ################################
    
    
    ################################ SAVE→RELOAD INTEGRITY CHECK ################################
    # --- Save→Reload integrity check on a deterministic subset of val data ---
    # Use the first up-to-16 rows for a stable subset
    subset_idx = np.arange(0, min(16, len(val_ds)))
    x_eval = torch.from_numpy(val_ds.X[subset_idx])
    
    # Get original model predictions for comparison
    best_model.eval()
    with torch.no_grad():
        logits_a = best_model(x_eval)
        pred_a = torch.argmax(logits_a, dim=1).cpu().numpy()
    
    # Reload model from checkpoint (restores architecture + weights via saved hparams)
    # get the path to the best model checkpoint in the main directory. It is the only one that starts with "best_model" and ends with ".ckpt"
    best_model_path = list(Path.cwd().glob("best_model_*.ckpt"))[0]
    reloaded_model = BinaryClassifier.load_from_checkpoint(best_model_path)
    reloaded_model.eval()
    with torch.no_grad():
        logits_b = reloaded_model(x_eval)
        pred_b = torch.argmax(logits_b, dim=1).cpu().numpy()
    
    # Compare class predictions exactly; also report numeric diff in logits
    same_preds = np.array_equal(pred_a, pred_b)
    max_abs_diff = float(torch.max(torch.abs(logits_a - logits_b)).cpu().numpy())
    
    print(f"\nIntegrity check for pytorch lightning model save on {len(subset_idx)} samples:")
    print(f"  predicted class arrays equal: {same_preds}")
    print(f"  max absolute diff in logits (raw model output): {max_abs_diff:.9f}")
    if not same_preds:
        mismatch = np.where(pred_a != pred_b)[0].tolist()
        print(f"  differing indices: {mismatch}")
    ################################ SAVE→RELOAD INTEGRITY CHECK ################################
    
    
    ################################ EXECUTORCH EXPORT ################################
    # Export best_model to ExecuTorch .pte (targeting XNNPACK CPU backend)
    best_model.eval()
    
    pte_path = Path("binary_mlp_xnnpack.pte")
    example_inputs = (torch.randn(1, input_dim),)
    
    # no batch size specified as we will run 1 sample at a time on a Raspberry Pi Zero w2
    exported = export(
        best_model,
        example_inputs,
    )
    
    et_program = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()
    
    with open(pte_path, "wb") as f:
        f.write(et_program.buffer)
    
    print(f"Saved ExecuTorch program to: {pte_path}")
    ################################ EXECUTORCH EXPORT ################################
    
    
    ################################ EXECUTORCH RUNTIME CHECK ################################
    # Load .pte and run it with the Python ExecuTorch runtime
    rt = Runtime.get()
    program = rt.load_program(pte_path, verification=Verification.Minimal)
    forward = program.load_method("forward")
    
    # Ensure CPU float32 and contiguous memory
    x_eval_f32 = x_eval.to(dtype=torch.float32, device="cpu").contiguous()
    
    def run_executorch_batched(forward, x_batch: torch.Tensor) -> torch.Tensor:
        """
        Works with static-batch=1 exports by looping per sample.
        If you ever re-export with a dynamic batch, this still works.
        """
        outs = []
        for i in range(x_batch.shape[0]):
            out_i = forward.execute((x_batch[i:i+1, :],))[0]  # (1, num_classes)
            outs.append(out_i)
        return torch.vstack(outs)  # (N, num_classes)
    
    logits_et = run_executorch_batched(forward, x_eval_f32)
    pred_et = torch.argmax(logits_et, dim=1).cpu().numpy()
    
    same_preds_et = np.array_equal(pred_a, pred_et)
    max_abs_diff_et = float(torch.max(torch.abs(logits_a - logits_et)).cpu().numpy())
    
    print(f"\nExecuTorch runtime check on {len(x_eval_f32)} samples:")
    print(f"  predicted class arrays equal: {same_preds_et}")
    print(f"  max absolute diff in logits (PyTorch vs ExecuTorch): {max_abs_diff_et:.9f}")
    ################################ EXECUTORCH RUNTIME CHECK ################################
    print("\nDone.")
    return


if __name__ == "__main__":
    main()
