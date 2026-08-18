import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score


class AdversarialTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)

    def train_step(self, batch, labels, epsilon, attack_fn):
        batch_adv = attack_fn(batch, labels, epsilon=epsilon)
        mix_mask = np.random.random(batch.size(0)) < 0.5
        mixed_batch = torch.where(
            torch.tensor(mix_mask).unsqueeze(1).to(self.device),
            batch_adv, batch
        )
        return mixed_batch

    def adversarial_train(self, train_loader, epochs, lr, epsilon, mix_ratio=0.5):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        from src.core.adversarial_generator import AdversarialAttackEngine

        attack_engine = AdversarialAttackEngine(self.model, self.device)

        history = {'loss': [], 'clean_acc': [], 'adv_acc': []}

        for epoch in range(epochs):
            total_loss = 0.0
            all_preds = []
            all_labels = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                n_adv = int(len(batch_x) * mix_ratio)

                if n_adv > 0:
                    adv_indices = np.random.choice(len(batch_x), n_adv, replace=False)
                    adv_batch = batch_x[adv_indices]
                    adv_labels = batch_y[adv_indices]

                    self.model.eval()
                    adv_data = attack_engine.fgsm_attack(
                        adv_batch, adv_labels, epsilon=epsilon
                    )
                    self.model.train()

                clean_indices = list(set(range(len(batch_x))) - set(adv_indices)) \
                    if n_adv > 0 else list(range(len(batch_x)))
                clean_batch = batch_x[clean_indices]
                clean_labels = batch_y[clean_indices]

                if n_adv > 0:
                    mixed_x = torch.cat([clean_batch, adv_data], dim=0)
                    mixed_y = torch.cat([clean_labels, adv_labels], dim=0)
                else:
                    mixed_x = clean_batch
                    mixed_y = clean_labels

                optimizer.zero_grad()
                outputs = self.model(mixed_x)
                loss = criterion(outputs, mixed_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * mixed_x.size(0)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(mixed_y.cpu().numpy())

            epoch_loss = total_loss / len(train_loader.dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)
            history['loss'].append(epoch_loss)
            history['clean_acc'].append(epoch_acc)

            print("Adversarial Training Epoch {}/{} - Loss: {:.4f} - Acc: {:.4f}".format(
                epoch + 1, epochs, epoch_loss, epoch_acc))

        return history

    def evaluate_hardened_model(self, test_loader_clean, test_loader_adv, device=None):
        if device is None:
            device = self.device

        self.model.eval()

        all_clean_preds = []
        all_clean_labels = []
        all_adv_preds = []
        all_adv_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader_clean:
                batch_x = batch_x.to(device)
                outputs = self.model(batch_x)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_clean_preds.extend(preds)
                all_clean_labels.extend(batch_y.numpy())

            for batch_x, batch_y in test_loader_adv:
                batch_x = batch_x.to(device)
                outputs = self.model(batch_x)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_adv_preds.extend(preds)
                all_adv_labels.extend(batch_y.numpy())

        clean_acc = accuracy_score(all_clean_labels, all_clean_preds)
        adv_acc = accuracy_score(all_adv_labels, all_adv_preds)

        return {
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'robustness_gap': clean_acc - adv_acc,
            'clean_predictions': all_clean_preds,
            'adversarial_predictions': all_adv_preds,
            'clean_labels': all_clean_labels,
            'adversarial_labels': all_adv_labels
        }

    def compare_models(self, baseline_path, hardened_path, test_data, test_labels):
        from src.core.model_trainer import IDSModel

        n_features = test_data.shape[1]

        baseline_model = IDSModel(input_dim=n_features)
        baseline_model.load_model(baseline_path)
        baseline_model = baseline_model.to(self.device)
        baseline_model.eval()

        hardened_model = IDSModel(input_dim=n_features)
        hardened_model.load_model(hardened_path)
        hardened_model = hardened_model.to(self.device)
        hardened_model.eval()

        from src.core.adversarial_generator import AdversarialAttackEngine

        test_tensor = torch.FloatTensor(test_data).to(self.device)
        test_label_tensor = torch.LongTensor(test_labels).to(self.device)

        baseline_engine = AdversarialAttackEngine(baseline_model, self.device)
        hardened_engine = AdversarialAttackEngine(hardened_model, self.device)

        baseline_results = baseline_engine.evaluate_robustness(
            test_tensor, test_label_tensor,
            lambda d, lbl: baseline_engine.fgsm_attack(d, lbl, epsilon=0.1)
        )

        hardened_results = hardened_engine.evaluate_robustness(
            test_tensor, test_label_tensor,
            lambda d, lbl: hardened_engine.fgsm_attack(d, lbl, epsilon=0.1)
        )

        comparison = {
            'baseline_clean_accuracy': baseline_results['clean_accuracy'],
            'baseline_adv_accuracy': baseline_results['adversarial_accuracy'],
            'hardened_clean_accuracy': hardened_results['clean_accuracy'],
            'hardened_adv_accuracy': hardened_results['adversarial_accuracy'],
            'baseline_evasion_rate': baseline_results['evasion_rate'],
            'hardened_evasion_rate': hardened_results['evasion_rate'],
            'robustness_improvement': (
                hardened_results['adversarial_accuracy'] - baseline_results['adversarial_accuracy']
            ),
            'evasion_reduction': (
                baseline_results['evasion_rate'] - hardened_results['evasion_rate']
            )
        }

        print("\n=== Model Comparison ===")
        print("Baseline - Clean: {:.4f} | Adv: {:.4f} | Evasion: {:.2f}%".format(
            comparison['baseline_clean_accuracy'],
            comparison['baseline_adv_accuracy'],
            comparison['baseline_evasion_rate'] * 100))
        print("Hardened - Clean: {:.4f} | Adv: {:.4f} | Evasion: {:.2f}%".format(
            comparison['hardened_clean_accuracy'],
            comparison['hardened_adv_accuracy'],
            comparison['hardened_evasion_rate'] * 100))
        print("Robustness Improvement: +{:.4f}".format(
            comparison['robustness_improvement']))
        print("Evasion Reduction: {:.2f}%".format(
            comparison['evasion_reduction'] * 100))

        return comparison


def main():
    parser = argparse.ArgumentParser(
        description='Adversarially train the IDS model'
    )
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to baseline model')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data')
    parser.add_argument('--n-samples', type=int, default=5000,
                        help='Number of samples')
    parser.add_argument('--data-dir', type=str, default='data/',
                        help='Path to NSL-KDD dataset')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Adversarial training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Perturbation magnitude')
    parser.add_argument('--mix-ratio', type=float, default=0.5,
                        help='Ratio of adversarial examples in training')
    parser.add_argument('--attack', type=str, default='fgsm',
                        choices=['fgsm', 'pgd'],
                        help='Attack type for adversarial training')
    parser.add_argument('--output-dir', type=str, default='models/',
                        help='Output directory')
    args = parser.parse_args()

    from src.core.dataset_loader import DatasetManager
    from src.core.model_trainer import IDSModel

    dm = DatasetManager()

    if args.synthetic:
        X, y = dm.generate_synthetic_traffic(n_samples=args.n_samples)
        X_train, X_test, y_train, y_test = dm.split_data(X, y, test_size=0.2)
    else:
        dm.load_nsl_kdd(args.data_dir)
        X_train = dm.X_train
        y_train = dm.y_train
        X_test = dm.X_test
        y_test = dm.y_test

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = IDSModel(input_dim=X_train.shape[1])
    model.load_model(args.model_path)
    model = model.to(device)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    trainer = AdversarialTrainer(model, device)
    trainer.adversarial_train(
        train_loader, epochs=args.epochs,
        lr=args.lr, epsilon=args.epsilon, mix_ratio=args.mix_ratio
    )

    hardened_path = os.path.join(args.output_dir, 'ids_hardened.pt')
    model.save_model(hardened_path)

    test_ds = TensorDataset(
        torch.FloatTensor(X_test), torch.LongTensor(y_test)
    )
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    results = trainer.evaluate_hardened_model(test_loader, test_loader, device)

    print("\n=== Hardened Model Evaluation ===")
    print("Clean Accuracy: {:.4f}".format(results['clean_accuracy']))
    print("Adversarial Accuracy: {:.4f}".format(results['adversarial_accuracy']))
    print("Robustness Gap: {:.4f}".format(results['robustness_gap']))


if __name__ == '__main__':
    main()
