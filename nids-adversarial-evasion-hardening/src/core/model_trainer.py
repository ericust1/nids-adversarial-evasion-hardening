import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json


class IDSModel(nn.Module):
    def __init__(self, input_dim):
        super(IDSModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        logits = self.network(x)
        return logits

    def train_model(self, train_loader, epochs, lr, device):
        self.to(device)
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        history = {'loss': [], 'accuracy': []}

        for epoch in range(epochs):
            total_loss = 0.0
            all_preds = []
            all_labels = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_x.size(0)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())

            epoch_loss = total_loss / len(train_loader.dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)

            print("Epoch {}/{} - Loss: {:.4f} - Accuracy: {:.4f}".format(
                epoch + 1, epochs, epoch_loss, epoch_acc))

        return history

    def evaluate(self, test_loader, device):
        self.to(device)
        self.eval()
        criterion = nn.CrossEntropyLoss()

        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)

                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(test_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        cm = _compute_confusion_matrix(all_labels, all_preds)

        metrics = {
            'loss': avg_loss,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'probabilities': all_probs,
            'true_labels': all_labels,
        }

        print("Eval - Loss: {:.4f} - Acc: {:.4f} - Prec: {:.4f} - Rec: {:.4f} - F1: {:.4f}".format(
            avg_loss, acc, prec, rec, f1))

        return metrics

    def save_model(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'architecture': {
                'input_dim': self.network[0].in_features,
            }
        }, path)
        print("Model saved to {}".format(path))

    def load_model(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded from {}".format(path))
        return self


def _compute_confusion_matrix(y_true, y_pred):
    cm = [[0, 0], [0, 0]]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def train_baseline(data_dir=None, synthetic=False, n_samples=10000,
                   epochs=30, batch_size=128, lr=0.001, output_dir='models/'):
    from src.core.dataset_loader import DatasetManager

    dm = DatasetManager()

    if synthetic:
        X, y = dm.generate_synthetic_traffic(n_samples=n_samples)
        print("Using {} synthetic samples".format(n_samples))
    else:
        X_train, y_train, X_test, y_test = dm.load_nsl_kdd(data_dir)
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])
        print("Using NSL-KDD dataset")

    X_train, X_test, y_train, y_test = dm.split_data(X, y, test_size=0.2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on device: {}".format(device))

    n_features = X_train.shape[1]
    model = IDSModel(input_dim=n_features)
    model = model.to(device)

    train_tensor = torch.FloatTensor(X_train)
    label_tensor = torch.LongTensor(y_train)
    train_ds = TensorDataset(train_tensor, label_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_tensor = torch.FloatTensor(X_test)
    test_label_tensor = torch.LongTensor(y_test)
    test_ds = TensorDataset(test_tensor, test_label_tensor)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    history = model.train_model(train_loader, epochs=epochs, lr=lr, device=device)

    metrics = model.evaluate(test_loader, device=device)

    model_path = os.path.join(output_dir, 'ids_baseline.pt')
    model.save_model(model_path)

    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    return model, metrics, history


def main():
    parser = argparse.ArgumentParser(description='Train IDS Baseline Model')
    parser.add_argument('--data-dir', type=str, default='data/',
                        help='Path to NSL-KDD dataset')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data')
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of synthetic samples')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='models/',
                        help='Output directory for model')
    args = parser.parse_args()

    model, metrics, history = train_baseline(
        data_dir=args.data_dir,
        synthetic=args.synthetic,
        n_samples=args.n_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir
    )

    print("\nBaseline Training Complete")
    print("Accuracy: {:.4f}".format(metrics['accuracy']))
    print("F1 Score: {:.4f}".format(metrics['f1']))


if __name__ == '__main__':
    main()
