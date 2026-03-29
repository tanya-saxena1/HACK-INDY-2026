import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, classification_report, confusion_matrix
)
import joblib

FEATURE_COLS = [
    "lat", "lng",
    "restaurant", "cafe", "gym", "grocery", "bar",
    "transit_station", "park", "pharmacy", "hospital",
    "convenience", "stadium", "library", "movie_theater", "shopping_mall",
    "avg_temp_c", "avg_precip_mm", "avg_sun_hrs",
]

DIM_WEIGHTS = {
    "walkability":    0.25,
    "food_amenities": 0.20,
    "campus_spirit":  0.20,
    "weather":        0.20,
    "safety":         0.15,
}


def compute_sub_scores(row: pd.Series) -> dict:
    def norm_poi(val, cap=20):
        return min((val or 0) / cap, 1.0)

    walkability = (
        norm_poi(row.get("transit_station"), 20) * 0.40 +
        norm_poi(row.get("grocery"), 20)          * 0.25 +
        norm_poi(row.get("convenience"), 20)       * 0.15 +
        norm_poi(row.get("pharmacy"), 20)          * 0.20
    ) * 100

    food_amenities = (
        norm_poi(row.get("restaurant"), 20) * 0.40 +
        norm_poi(row.get("cafe"), 20)        * 0.25 +
        norm_poi(row.get("bar"), 20)         * 0.15 +
        norm_poi(row.get("gym"), 20)         * 0.10 +
        norm_poi(row.get("shopping_mall"), 5)* 0.10
    ) * 100

    campus_spirit = (
        norm_poi(row.get("stadium"), 3)        * 0.35 +
        norm_poi(row.get("movie_theater"), 10) * 0.25 +
        norm_poi(row.get("bar"), 20)           * 0.20 +
        norm_poi(row.get("library"), 10)       * 0.20
    ) * 100

    temp_c     = row.get("avg_temp_c") or 15.0
    precip_mm  = row.get("avg_precip_mm") or 3.0
    sun_hrs    = row.get("avg_sun_hrs") or 6.0
    temp_score = max(0, 1 - abs(temp_c - 18) / 25)
    rain_score = max(0, 1 - precip_mm / 15)
    sun_score  = min(sun_hrs / 10, 1.0)
    weather    = (temp_score * 0.40 + rain_score * 0.30 + sun_score * 0.30) * 100

    safety = (
        norm_poi(row.get("hospital"), 10) * 0.60 +
        norm_poi(row.get("pharmacy"), 20) * 0.40
    ) * 100

    return {
        "walkability":    round(walkability, 1),
        "food_amenities": round(food_amenities, 1),
        "campus_spirit":  round(campus_spirit, 1),
        "weather":        round(weather, 1),
        "safety":         round(safety, 1),
    }


def compute_overall(sub_scores: dict) -> float:
    return round(sum(sub_scores[dim] * w for dim, w in DIM_WEIGHTS.items()), 1)


class CampusDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LivabilityModel(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return torch.clamp(self.net(x), 0, 100)


def bucketize(scores: np.ndarray) -> np.ndarray:
    """Low=0 (<40), Mid=1 (40-70), High=2 (>70)"""
    buckets = np.zeros(len(scores), dtype=int)
    buckets[scores >= 40] = 1
    buckets[scores >= 70] = 2
    return buckets


def evaluate(model, test_loader):
    model.eval()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            all_preds.extend(outputs.squeeze().tolist())
            all_targets.extend(targets.squeeze().tolist())

    preds   = np.array(all_preds)
    targets = np.array(all_targets)

    # Regression metrics
    mse  = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(targets, preds)
    r2   = r2_score(targets, preds)

    print("\n── Regression Metrics ──────────────────────────")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}  (model is off by ~±{rmse:.1f} pts on average)")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}   (1.0 = perfect fit)")

    pred_classes   = bucketize(preds)
    target_classes = bucketize(targets)

    f1_macro    = f1_score(target_classes, pred_classes, average="macro",    zero_division=0)
    f1_weighted = f1_score(target_classes, pred_classes, average="weighted", zero_division=0)
    f1_per_cls  = f1_score(target_classes, pred_classes, average=None,       zero_division=0)

    print("\n── Adapted F1 (Low / Mid / High buckets) ───────")
    print(f"  F1 macro    : {f1_macro:.4f}")
    print(f"  F1 weighted : {f1_weighted:.4f}")
    for label, score in zip(["Low (<40)", "Mid (40-70)", "High (>70)"], f1_per_cls):
        print(f"    {label:15s}: {score:.4f}")

    print("\n── Classification Report ───────────────────────")
    print(classification_report(
    target_classes,
    pred_classes,
    labels=[0, 1, 2],  # force all 3 classes
    target_names=["Low (<40)", "Mid (40-70)", "High (>70)"],
    zero_division=0
    ))

    print("── Confusion Matrix (rows=actual, cols=predicted)")
    print(confusion_matrix(target_classes, pred_classes))
    print("  order: [Low, Mid, High]\n")

    return {
        "mse": mse, "rmse": rmse, "mae": mae, "r2": r2,
        "f1_macro": f1_macro, "f1_weighted": f1_weighted,
        "f1_per_class": f1_per_cls.tolist(),
    }


def train(csv_path="campus_features.csv"):
    df = pd.read_csv(csv_path)

    sub        = df.apply(compute_sub_scores, axis=1).apply(pd.Series)
    df["score"] = sub.apply(lambda row: compute_overall(row.to_dict()), axis=1)

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["score"].values.astype(np.float32)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42
    )

    train_dl = DataLoader(CampusDataset(X_tr, y_tr), batch_size=16, shuffle=True)
    test_dl  = DataLoader(CampusDataset(X_te, y_te), batch_size=16)

    model     = LivabilityModel(input_size=len(FEATURE_COLS))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(200):
        model.train()
        ep_loss = 0
        for inputs, targets in train_dl:
            optimizer.zero_grad()
            outputs  = model(inputs)
            loss     = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 40 == 0:
            print(f"Epoch {epoch+1}/200  loss={ep_loss/len(train_dl):.4f}")

    print("\nEvaluating on test set…")
    metrics = evaluate(model, test_dl)

    torch.save(model.state_dict(), "model_weights.pth")
    print("Saved model_weights.pth + scaler.pkl")
    return model, scaler, metrics


if __name__ == "__main__":
    train()