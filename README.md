# NIDS Adversarial Evasion Hardening

Adversarial ML security research project demonstrating evasion attacks against ML-based Network Intrusion Detection Systems and adversarial training as a defense mechanism.

## Architecture

```
                          +-------------------+
                          |  NSL-KDD Dataset  |
                          |  (or Synthetic)   |
                          +--------+----------+
                                   |
                                   v
                    +--------------+-------------+
                    |    DatasetManager         |
                    |  - Feature preprocessing  |
                    |  - Train/test split       |
                    |  - Normalization          |
                    +--------------+-------------+
                                   |
                     +-------------+-------------+
                     |                           |
                     v                           v
          +----------+--------+      +---------+----------+
          |  IDSModel (PyTorch) |      |  Baseline Metrics  |
          |  - Dense(64, ReLU)  |      |  - Accuracy        |
          |  - Dropout(0.3)     |      |  - F1 Score        |
          |  - Dense(32, ReLU)  |      |  - Confusion Matrix |
          |  - Dropout(0.3)     |      +--------------------+
          |  - Dense(16, ReLU)  |
          |  - Dense(2)         |
          +----------+----------+
                     |
                     v
          +----------+------------------------------------+
          |          |                                    |
          v          v                                    v
   +-----+----+ +---+-----------+              +----------+----------+
   |   FGSM   | |     PGD      |              | Carlini-Wagner (L2) |
   | Single   | | Multi-step   |              | PGD-approximated    |
   | Step     | | Iterative    |              | Minimal perturb.    |
   +-----+----+ +-------+-------+              +----------+----------+
         |               |                                  |
         +-------+-------+----------------------------------+
                 |
                 v
   +-------------+------------------+
   |  Accuracy Drop Detected       |
   |  - Clean:    92%             |
   |  - FGSM:     45%  (DROP!)     |
   |  - PGD:      28%  (DROP!)     |
   +-------------+------------------+
                 |
                 v
   +-------------+------------------+
   |  Adversarial Training          |
   |  - Mix clean + adversarial    |
   |  - Train on perturbed data    |
   |  - FGSM on-the-fly augment    |
   +-------------+------------------+
                 |
                 v
   +-------------+------------------+
   |  Hardened IDS Model            |
   |  - Clean:    90%              |
   |  - FGSM:     75%  (IMPROVED!) |
   |  - PGD:      65%  (IMPROVED!) |
   +--------------------------------+
```

## Features

- **ML-Based Intrusion Detection**: Neural network classifier trained on network traffic features
- **Adversarial Evasion Attacks**: FGSM, PGD, and Carlini-Wagner style attacks that bypass detection
- **Adversarial Training Defense**: On-the-fly adversarial example generation during training
- **Comprehensive Evaluation**: Metrics, confusion matrices, ROC curves, and comparison reports
- **Synthetic Data Generation**: No external dataset required for testing
- **Docker Lab**: Jupyter notebook + Flask API for interactive experimentation
- **AWS Deployment**: Terraform configs for SageMaker notebook instances

## Attack Descriptions

### FGSM (Fast Gradient Sign Method)
Single-step attack that computes the gradient of the loss with respect to input features and perturbs in the direction of the sign of the gradient. Fast but produces relatively weak adversarial examples.

### PGD (Projected Gradient Descent)
Multi-step iterative attack. Takes small steps in the gradient direction and projects the result back onto an epsilon-bounded ball around the original input. Produces the strongest adversarial examples among first-order methods.

### Carlini-Wagner L2 (PGD-Approximated)
Optimization-based attack that seeks minimal L2 perturbations. This implementation uses a PGD-style optimizer with adaptive perturbation scaling and confidence parameters to approximate CW behavior.

## Setup

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh

python -m pytest tests/ -v
```

Or manually:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn numpy matplotlib pandas pytest flask
```

## Quick Start

```bash
python -m src.core.model_trainer --synthetic --epochs 20 --output-dir models/
python -m src.core.adversarial_generator --model-path models/ids_baseline.pt --attack fgsm --epsilon 0.1 --synthetic
python -m src.core.adversarial_generator --model-path models/ids_baseline.pt --attack pgd --epsilon 0.1 --iterations 40 --synthetic
python -m src.modules.adversarial_training --model-path models/ids_baseline.pt --epochs 15 --epsilon 0.1 --synthetic --output-dir models/
python -m src.modules.evaluation --baseline-model models/ids_baseline.pt --hardened-model models/ids_hardened.pt --synthetic --output-dir reports/
```

## Expected Results Format

```
=== Baseline Model ===
Clean Accuracy:       0.9200
F1 Score:             0.9100

=== FGSM Attack (epsilon=0.1) ===
Adversarial Accuracy: 0.4500
Accuracy Drop:        0.4700
Evasion Rate:         47.00%
Risk Level:           CRITICAL

=== PGD Attack (epsilon=0.1, iterations=40) ===
Adversarial Accuracy: 0.2800
Accuracy Drop:        0.6400
Evasion Rate:         64.00%
Risk Level:           CRITICAL

=== After Adversarial Training ===
Hardened Clean Acc:   0.9000
Hardened FGSM Acc:    0.7500
Hardened PGD Acc:     0.6500
Robustness Gap:       0.1500 (down from 0.47)
```

## Project Structure

```
nids-adversarial-evasion-hardening/
+-- src/
|   +-- core/
|   |   +-- dataset_loader.py       # NSL-KDD loading & synthetic data
|   |   +-- model_trainer.py        # PyTorch IDS neural network
|   |   +-- adversarial_generator.py # FGSM, PGD, CW attack engine
|   +-- modules/
|       +-- adversarial_training.py  # Adversarial training defense
|       +-- evaluation.py            # Metrics, plots, reports
+-- tests/
|   +-- unit/                       # Unit tests for each module
|   +-- integration/                # End-to-end pipeline test
+-- lab/
|   +-- docker-compose.yml          # Jupyter + Flask API
|   +-- terraform/main.tf           # AWS SageMaker deployment
+-- docs/
|   +-- setup-guide.md              # Detailed setup instructions
+-- scripts/
    +-- setup.sh                    # Dependency installation
    +-- package_project.py          # Archive script
```

## Docker Lab

```bash
cd lab && docker compose up --build
```

## License

MIT License. See [LICENSE](LICENSE) for details.
