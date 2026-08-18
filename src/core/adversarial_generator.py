import argparse
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


class AdversarialAttackEngine:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def fgsm_attack(self, data, labels, epsilon=0.1):
        data_adv = data.clone().detach().to(self.device)
        data_adv.requires_grad = True

        self.model.zero_grad()
        outputs = self.model(data_adv)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels.to(self.device))

        loss.backward()
        grad_sign = data_adv.grad.data.sign()
        perturbed = data_adv + epsilon * grad_sign

        return perturbed.detach()

    def pgd_attack(self, data, labels, epsilon=0.1, alpha=0.01, iterations=40):
        original_data = data.clone().detach().to(self.device)
        perturbed = data.clone().detach().to(self.device)
        perturbed.requires_grad = True

        criterion = nn.CrossEntropyLoss()

        for _ in range(iterations):
            self.model.zero_grad()
            outputs = self.model(perturbed)
            loss = criterion(outputs, labels.to(self.device))

            loss.backward()
            grad = perturbed.grad.data

            perturbed = perturbed.detach() + alpha * grad.sign()

            delta = perturbed - original_data
            delta = torch.clamp(delta, -epsilon, epsilon)
            perturbed = torch.clamp(original_data + delta, 0.0, 1.0)

            perturbed.requires_grad = True

        return perturbed.detach()

    def cw_attack(self, data, labels, targeted=False, cw_confidence=0, max_iter=100):
        original_data = data.clone().detach().to(self.device)
        perturbed = data.clone().detach().to(self.device)

        batch_size = data.size(0)
        labels_onehot = torch.zeros(batch_size, 2).to(self.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)

        if not targeted:
            target_labels = 1 - labels.clone().detach().to(self.device)
        else:
            target_labels = labels.clone().detach().to(self.device)

        target_onehot = torch.zeros(batch_size, 2).to(self.device)
        target_onehot.scatter_(1, target_labels.unsqueeze(1), 1)

        kappa = max(cw_confidence, 1.0)
        c_init = 1.0
        c = c_init

        perturbed.requires_grad = True

        optimizer = torch.optim.Adam([perturbed], lr=0.01)

        for iteration in range(max_iter):
            optimizer.zero_grad()
            outputs = self.model(perturbed)
            probs = torch.softmax(outputs, dim=1)

            real_prob = (probs * labels_onehot).sum(dim=1)
            other_prob = (probs * (1 - labels_onehot)).sum(dim=1)

            loss_attack = torch.clamp(other_prob - real_prob + kappa, min=0)
            loss_perturbation = torch.norm(perturbed - original_data, p=2)

            total_loss = loss_perturbation + c * loss_attack.mean()

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta = perturbed - original_data
                norms = torch.norm(delta, p=2, dim=1, keepdim=True)
                mask = (norms > 1.0).squeeze(1)
                if mask.any():
                    delta[mask] = delta[mask] / norms[mask].clamp(min=1e-12)
                perturbed.data = torch.clamp(original_data + delta, 0.0, 1.0)
                perturbed.requires_grad = True

        return perturbed.detach()

    def evaluate_robustness(self, test_data, test_labels, attack_fn, batch_size=64):
        self.model.eval()
        all_preds_clean = []
        all_preds_adv = []
        evasions = []

        n_samples = test_data.size(0) if isinstance(test_data, torch.Tensor) else len(test_data)

        if not isinstance(test_data, torch.Tensor):
            test_data = torch.FloatTensor(test_data)
        if not isinstance(test_labels, torch.Tensor):
            test_labels = torch.LongTensor(test_labels)

        for i in range(0, n_samples, batch_size):
            batch_data = test_data[i:i + batch_size].to(self.device)
            batch_labels = test_labels[i:i + batch_size].to(self.device)

            with torch.no_grad():
                clean_outputs = self.model(batch_data)
                clean_preds = clean_outputs.argmax(dim=1)

            adv_data = attack_fn(batch_data, batch_labels)

            with torch.no_grad():
                adv_outputs = self.model(adv_data)
                adv_preds = adv_outputs.argmax(dim=1)

            for j in range(len(batch_labels)):
                clean_pred = clean_preds[j].item()
                adv_pred = adv_preds[j].item()
                true_label = batch_labels[j].item()

                all_preds_clean.append(clean_pred)
                all_preds_adv.append(adv_pred)

                if clean_pred == true_label and adv_pred != true_label:
                    evasions.append({
                        'sample_idx': i + j,
                        'true_label': true_label,
                        'clean_prediction': clean_pred,
                        'adversarial_prediction': adv_pred
                    })

        clean_acc = accuracy_score(
            test_labels.cpu().numpy(), all_preds_clean
        )
        adv_acc = accuracy_score(
            test_labels.cpu().numpy(), all_preds_adv
        )

        return {
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'accuracy_drop': clean_acc - adv_acc,
            'total_samples': n_samples,
            'successful_evasions': len(evasions),
            'evasion_rate': len(evasions) / n_samples,
            'evasion_details': evasions
        }

    def generate_evasion_report(self, baseline_acc, robust_acc, attack_name):
        return {
            'attack_name': attack_name,
            'baseline_accuracy': round(baseline_acc, 4),
            'adversarial_accuracy': round(robust_acc, 4),
            'accuracy_degradation': round(baseline_acc - robust_acc, 4),
            'degradation_percentage': round(
                ((baseline_acc - robust_acc) / max(baseline_acc, 1e-8)) * 100, 2
            ),
            'risk_level': self._assess_risk(baseline_acc, robust_acc),
            'recommendation': self._get_recommendation(baseline_acc, robust_acc)
        }

    def _assess_risk(self, baseline_acc, robust_acc):
        drop = baseline_acc - robust_acc
        if drop >= 0.3:
            return 'CRITICAL'
        elif drop >= 0.15:
            return 'HIGH'
        elif drop >= 0.05:
            return 'MEDIUM'
        return 'LOW'

    def _get_recommendation(self, baseline_acc, robust_acc):
        drop = baseline_acc - robust_acc
        if drop >= 0.15:
            return ('Model is highly vulnerable to adversarial evasion. '
                    'Immediate adversarial training recommended with epsilon=0.1 and '
                    'mix_ratio=0.5 for at least 20 epochs.')
        elif drop >= 0.05:
            return ('Model shows moderate vulnerability. Adversarial training '
                    'with PGD-based augmentation should improve robustness.')
        return ('Model shows acceptable robustness. Periodic adversarial '
                'testing recommended to maintain security posture.')


def main():
    parser = argparse.ArgumentParser(
        description='Generate adversarial evasion attacks against IDS model'
    )
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--attack', type=str, default='fgsm',
                        choices=['fgsm', 'pgd', 'cw'],
                        help='Attack type')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Perturbation magnitude')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='Step size for PGD')
    parser.add_argument('--iterations', type=int, default=40,
                        help='Iterations for PGD/CW')
    parser.add_argument('--cw-confidence', type=float, default=0,
                        help='Confidence for CW attack')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data')
    parser.add_argument('--n-samples', type=int, default=2000,
                        help='Number of samples')
    parser.add_argument('--data-dir', type=str, default='data/',
                        help='Path to NSL-KDD dataset')
    args = parser.parse_args()

    from src.core.dataset_loader import DatasetManager
    from src.core.model_trainer import IDSModel

    dm = DatasetManager()

    if args.synthetic:
        X, y = dm.generate_synthetic_traffic(n_samples=args.n_samples)
        X_train, X_test, y_train, y_test = dm.split_data(X, y, test_size=0.3)
    else:
        dm.load_nsl_kdd(args.data_dir)
        X_test = dm.X_test
        y_test = dm.y_test

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_features = X_test.shape[1]

    model = IDSModel(input_dim=n_features)
    model.load_model(args.model_path)
    model = model.to(device)

    engine = AdversarialAttackEngine(model, device)

    test_data = torch.FloatTensor(X_test).to(device)
    test_labels = torch.LongTensor(y_test).to(device)

    print("Running {} attack with epsilon={}".format(args.attack, args.epsilon))

    if args.attack == 'fgsm':
        results = engine.evaluate_robustness(
            test_data, test_labels,
            lambda d, lbl: engine.fgsm_attack(d, lbl, epsilon=args.epsilon)
        )
    elif args.attack == 'pgd':
        results = engine.evaluate_robustness(
            test_data, test_labels,
            lambda d, lbl: engine.pgd_attack(
                d, lbl, epsilon=args.epsilon,
                alpha=args.alpha, iterations=args.iterations
            )
        )
    elif args.attack == 'cw':
        results = engine.evaluate_robustness(
            test_data, test_labels,
            lambda d, lbl: engine.cw_attack(
                d, lbl, targeted=False,
                cw_confidence=args.cw_confidence,
                max_iter=args.iterations
            )
        )

    report = engine.generate_evasion_report(
        results['clean_accuracy'],
        results['adversarial_accuracy'],
        args.attack.upper()
    )

    print("\n=== Adversarial Evasion Report ===")
    print("Attack: {}".format(report['attack_name']))
    print("Clean Accuracy: {:.4f}".format(report['baseline_accuracy']))
    print("Adversarial Accuracy: {:.4f}".format(report['adversarial_accuracy']))
    print("Accuracy Drop: {:.4f}".format(report['accuracy_degradation']))
    print("Risk Level: {}".format(report['risk_level']))
    print("Successful Evasions: {}/{}".format(
        results['successful_evasions'], results['total_samples']))
    print("Evasion Rate: {:.2f}%".format(report['degradation_percentage']))
    print("Recommendation: {}".format(report['recommendation']))


if __name__ == '__main__':
    main()
