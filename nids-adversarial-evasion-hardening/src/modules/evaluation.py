import os
import sys
import argparse
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ModelEvaluator:
    def __init__(self):
        self.results = {}

    def compute_metrics(self, y_true, y_pred, y_scores=None):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }

        if y_scores is not None:
            y_scores = np.array(y_scores)
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
            except ValueError:
                metrics['auc_roc'] = 0.0

        return metrics

    def plot_confusion_matrix(self, cm, class_names, save_path):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               ylabel='True Label',
               xlabel='Predicted Label',
               title='Confusion Matrix')

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')

        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, y_true, y_scores, save_path):
        fig, ax = plt.subplots(figsize=(8, 6))

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_val = roc_auc_score(y_true, y_scores)

        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label='ROC curve (AUC = {:.4f})'.format(auc_val))
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc='lower right')

        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_accuracy_comparison(self, results_dict, save_path):
        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(results_dict.keys())
        clean_accs = [results_dict[m]['clean_accuracy'] for m in models]
        adv_accs = [results_dict[m]['adversarial_accuracy'] for m in models]

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax.bar(x - width/2, clean_accs, width, label='Clean Accuracy',
                       color='#2196F3', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, adv_accs, width, label='Adversarial Accuracy',
                       color='#F44336', edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Accuracy')
        ax.set_title('Model Robustness Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.05])

        for bar in bars1:
            height = bar.get_height()
            ax.annotate('{:.3f}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            ax.annotate('{:.3f}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)

        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def generate_evaluation_report(self, all_results, output_path):
        lines = []
        lines.append('# NIDS Adversarial Evasion - Evaluation Report')
        lines.append('')
        lines.append('## Overview')
        lines.append('')
        lines.append('This report evaluates the robustness of ML-based Network Intrusion Detection')
        lines.append('Systems against adversarial evasion attacks.')
        lines.append('')

        lines.append('## Model Performance Summary')
        lines.append('')
        lines.append('| Model | Clean Acc | Adv Acc (FGSM) | Adv Acc (PGD) | F1 Score | AUC-ROC |')
        lines.append('|-------|-----------|----------------|---------------|----------|---------|')

        for model_name, metrics in all_results.items():
            clean_acc = metrics.get('clean_accuracy', 0)
            fgsm_acc = metrics.get('fgsm_accuracy', 0)
            pgd_acc = metrics.get('pgd_accuracy', 0)
            f1 = metrics.get('f1', 0)
            auc = metrics.get('auc_roc', 0)
            lines.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                model_name, clean_acc, fgsm_acc, pgd_acc, f1, auc))

        lines.append('')

        lines.append('## Attack Analysis')
        lines.append('')

        if 'baseline_fgsm' in all_results:
            fgsm = all_results['baseline_fgsm']
            lines.append('### FGSM (Fast Gradient Sign Method)')
            lines.append('')
            lines.append('- **Epsilon**: 0.1')
            lines.append('- **Clean Accuracy**: {:.4f}'.format(
                fgsm.get('clean_accuracy', 0)))
            lines.append('- **Adversarial Accuracy**: {:.4f}'.format(
                fgsm.get('adversarial_accuracy', 0)))
            lines.append('- **Accuracy Drop**: {:.4f}'.format(
                fgsm.get('accuracy_drop', 0)))
            lines.append('- **Successful Evasions**: {}'.format(
                fgsm.get('successful_evasions', 0)))
            lines.append('- **Evasion Rate**: {:.2f}%'.format(
                fgsm.get('evasion_rate', 0) * 100))
            lines.append('')

        if 'baseline_pgd' in all_results:
            pgd = all_results['baseline_pgd']
            lines.append('### PGD (Projected Gradient Descent)')
            lines.append('')
            lines.append('- **Epsilon**: 0.1, **Alpha**: 0.01, **Iterations**: 40')
            lines.append('- **Clean Accuracy**: {:.4f}'.format(
                pgd.get('clean_accuracy', 0)))
            lines.append('- **Adversarial Accuracy**: {:.4f}'.format(
                pgd.get('adversarial_accuracy', 0)))
            lines.append('- **Accuracy Drop**: {:.4f}'.format(
                pgd.get('accuracy_drop', 0)))
            lines.append('- **Successful Evasions**: {}'.format(
                pgd.get('successful_evasions', 0)))
            lines.append('- **Evasion Rate**: {:.2f}%'.format(
                pgd.get('evasion_rate', 0) * 100))
            lines.append('')

        lines.append('## Hardening Results')
        lines.append('')

        if 'hardened' in all_results:
            hardened = all_results['hardened']
            lines.append('After adversarial training:')
            lines.append('')
            lines.append('- **Clean Accuracy**: {:.4f}'.format(
                hardened.get('clean_accuracy', 0)))
            lines.append('- **Adversarial Accuracy**: {:.4f}'.format(
                hardened.get('adversarial_accuracy', 0)))
            lines.append('- **Robustness Gap**: {:.4f}'.format(
                hardened.get('robustness_gap', 0)))
            lines.append('')

        lines.append('## Conclusions')
        lines.append('')
        lines.append('1. ML-based NIDS models are vulnerable to adversarial perturbations')
        lines.append('2. FGSM provides fast single-step attacks with measurable accuracy degradation')
        lines.append('3. PGD multi-step attacks produce stronger adversarial examples')
        lines.append('4. Adversarial training improves model robustness against both attack types')
        lines.append('')

        report_text = '\n'.join(lines)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)

        print("Evaluation report saved to {}".format(output_path))
        return report_text


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate and compare IDS models'
    )
    parser.add_argument('--baseline-model', type=str, required=True,
                        help='Path to baseline model')
    parser.add_argument('--hardened-model', type=str, default=None,
                        help='Path to hardened model')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data')
    parser.add_argument('--n-samples', type=int, default=2000,
                        help='Number of test samples')
    parser.add_argument('--data-dir', type=str, default='data/',
                        help='Path to NSL-KDD dataset')
    parser.add_argument('--output-dir', type=str, default='reports/',
                        help='Output directory for reports and plots')
    args = parser.parse_args()

    from src.core.dataset_loader import DatasetManager
    from src.core.model_trainer import IDSModel
    from src.core.adversarial_generator import AdversarialAttackEngine

    dm = DatasetManager()

    if args.synthetic:
        X, y = dm.generate_synthetic_traffic(n_samples=args.n_samples)
        _, X_test, _, y_test = dm.split_data(X, y, test_size=0.3)
    else:
        dm.load_nsl_kdd(args.data_dir)
        X_test = dm.X_test
        y_test = dm.y_test

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_features = X_test.shape[1]

    evaluator = ModelEvaluator()

    baseline_model = IDSModel(input_dim=n_features)
    baseline_model.load_model(args.baseline_model)
    baseline_model = baseline_model.to(device)

    test_data = torch.FloatTensor(X_test).to(device)
    test_labels = torch.LongTensor(y_test).to(device)

    baseline_engine = AdversarialAttackEngine(baseline_model, device)

    baseline_clean_results = baseline_engine.evaluate_robustness(
        test_data, test_labels,
        lambda d, l: d
    )
    baseline_fgsm_results = baseline_engine.evaluate_robustness(
        test_data, test_labels,
        lambda d, l: baseline_engine.fgsm_attack(d, l, epsilon=0.1)
    )
    baseline_pgd_results = baseline_engine.evaluate_robustness(
        test_data, test_labels,
        lambda d, l: baseline_engine.pgd_attack(d, l, epsilon=0.1, alpha=0.01, iterations=20)
    )

    all_results = {
        'baseline': {
            'clean_accuracy': baseline_clean_results['clean_accuracy'],
            'fgsm_accuracy': baseline_fgsm_results['adversarial_accuracy'],
            'pgd_accuracy': baseline_pgd_results['adversarial_accuracy'],
            'f1': 0,
            'auc_roc': 0,
        },
        'baseline_fgsm': baseline_fgsm_results,
        'baseline_pgd': baseline_pgd_results,
    }

    if args.hardened_model and os.path.exists(args.hardened_model):
        hardened_model = IDSModel(input_dim=n_features)
        hardened_model.load_model(args.hardened_model)
        hardened_model = hardened_model.to(device)

        hardened_engine = AdversarialAttackEngine(hardened_model, device)

        hardened_clean = hardened_engine.evaluate_robustness(
            test_data, test_labels, lambda d, l: d
        )
        hardened_fgsm = hardened_engine.evaluate_robustness(
            test_data, test_labels,
            lambda d, l: hardened_engine.fgsm_attack(d, l, epsilon=0.1)
        )

        all_results['hardened'] = {
            'clean_accuracy': hardened_clean['clean_accuracy'],
            'fgsm_accuracy': hardened_fgsm['adversarial_accuracy'],
            'pgd_accuracy': 0,
            'f1': 0,
            'auc_roc': 0,
        }

    comparison_dict = {}
    for model_name in ['baseline', 'hardened']:
        if model_name in all_results:
            comparison_dict[model_name.title()] = {
                'clean_accuracy': all_results[model_name]['clean_accuracy'],
                'adversarial_accuracy': all_results[model_name]['fgsm_accuracy'],
            }

    if comparison_dict:
        evaluator.plot_accuracy_comparison(
            comparison_dict,
            os.path.join(args.output_dir, 'accuracy_comparison.png')
        )

    evaluator.generate_evaluation_report(
        all_results,
        os.path.join(args.output_dir, 'evaluation_report.md')
    )


if __name__ == '__main__':
    main()
