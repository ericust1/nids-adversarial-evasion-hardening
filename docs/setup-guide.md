# NIDS Adversarial Evasion Hardening - Setup Guide

## Prerequisites

- Python 3.9 or higher
- pip package manager
- 8 GB RAM minimum (16 GB recommended for adversarial training)
- 2 GB free disk space

## Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/nids-adversarial-evasion-hardening.git
cd nids-adversarial-evasion-hardening
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Core Dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn numpy matplotlib pandas pytest flask
```

### Step 4: Install Adversarial Robustness Toolbox (Optional)

```bash
pip install adversarial-robustness-toolbox
```

ART is optional because this project implements FGSM and PGD attacks manually using PyTorch autograd. ART is only needed if you want to use its built-in attack library for comparison experiments.

### Step 5: Download NSL-KDD Dataset (Optional)

The project includes synthetic traffic generation so you can run everything without the real dataset. If you want to train on real network intrusion data:

```bash
mkdir -p data
cd data
wget https://www.unb.ca/cic/datasets/nsl/kdd/NSL-KDD.zip
unzip NSL-KDD.zip
```

Place the CSV files (KDDTrain+.txt, KDDTest+.txt) inside the `data/` directory.

## Training the Baseline Model

Train a standard neural network intrusion detection system on the dataset:

```bash
python -m src.core.model_trainer --data-dir data/ --epochs 30 --batch-size 128 --lr 0.001 --output-dir models/
```

If you do not have the NSL-KDD dataset, use synthetic data:

```bash
python -m src.core.model_trainer --synthetic --n-samples 10000 --epochs 20 --output-dir models/
```

The baseline model will be saved to `models/ids_baseline.pt`. Training logs will show per-epoch loss, accuracy, precision, recall, and F1 score.

## Generating Adversarial Evasion Payloads

Once you have a trained baseline model, generate adversarial examples to test its robustness:

### FGSM Attack (Fast Gradient Sign Method)

```bash
python -m src.core.adversarial_generator --model-path models/ids_baseline.pt --attack fgsm --epsilon 0.1 --data-dir data/
```

FGSM computes the gradient of the loss with respect to the input features and perturbs each feature by `epsilon` in the direction that maximizes the loss. It is a single-step, fast attack.

### PGD Attack (Projected Gradient Descent)

```bash
python -m src.core.adversarial_generator --model-path models/ids_baseline.pt --attack pgd --epsilon 0.1 --alpha 0.01 --iterations 40 --data-dir data/
```

PGD iteratively applies small perturbations and projects back onto the epsilon ball. It is a multi-step attack that produces stronger adversarial examples than FGSM.

### Carlini-Wagner Style Attack

```bash
python -m src.core.adversarial_generator --model-path models/ids_baseline.pt --attack cw --cw-confidence 0 --max-iter 100 --data-dir data/
```

This is a PGD-based approximation of the Carlini-Wagner L2 attack, optimized for finding minimal perturbations that fool the classifier.

## Adversarial Training (Model Hardening)

Harden the baseline model by training it on a mixture of clean and adversarial examples:

```bash
python -m src.modules.adversarial_training --model-path models/ids_baseline.pt --epochs 20 --epsilon 0.1 --mix-ratio 0.5 --attack fgsm --output-dir models/
```

Parameters:
- `--mix-ratio`: Fraction of each training batch replaced with adversarial examples (0.5 means half clean, half adversarial)
- `--epsilon`: Perturbation magnitude used to generate adversarial examples during training
- `--attack`: Which adversarial attack to use during training (fgsm or pgd)

The hardened model is saved to `models/ids_hardened.pt`.

## Evaluation and Reporting

Generate a full evaluation report comparing baseline and hardened models:

```bash
python -m src.modules.evaluation --baseline-model models/ids_baseline.pt --hardened-model models/ids_hardened.pt --data-dir data/ --output-dir reports/
```

This produces:
- Confusion matrices for both models under clean and adversarial conditions
- ROC curves
- Accuracy comparison bar charts
- A comprehensive Markdown report with all metrics

## Docker Lab Environment

Start the full lab environment with Jupyter notebook and inference API:

```bash
cd lab
docker compose up --build
```

- Jupyter Notebook: http://localhost:8888
- Flask Model API: http://localhost:5000

## Terraform AWS Deployment

To deploy on AWS SageMaker:

```bash
cd lab/terraform
terraform init
terraform plan
terraform apply
```

## Troubleshooting

**Out of Memory**: Reduce batch size or use fewer synthetic samples. PGD attacks with many iterations are the most memory-intensive.

**ART Install Fails**: The manual FGSM/PGD implementations in this project do not require ART. Skip the ART installation and use the built-in attacks.

**Low Baseline Accuracy**: With synthetic data, accuracy depends on class separability. Increase `--n-samples` or adjust the synthetic data distributions in `dataset_loader.py`.
