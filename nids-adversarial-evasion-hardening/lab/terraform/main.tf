provider "aws" {
  region = "us-east-1"
}

resource "aws_s3_bucket" "model_artifacts" {
  bucket = "nids-evasion-model-artifacts-${random_id.bucket_suffix.hex}"

  lifecycle_rule {
    enabled = true
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
  }
}

resource "aws_s3_bucket_versioning" "model_versioning" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 8
}

resource "aws_iam_role" "sagemaker_execution_role" {
  name = "nids-sagemaker-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_basic" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy" "s3_access" {
  name = "nids-s3-model-access"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.model_artifacts.arn,
          "${aws_s3_bucket.model_artifacts.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_sagemaker_notebook_instance" "nids_notebook" {
  name          = "nids-adversarial-evasion-lab"
  role_arn      = aws_iam_role.sagemaker_execution_role.arn
  instance_type = "ml.t3.medium"

  lifecycle_config_name = aws_sagemaker_notebook_instance_lifecycle_config.nids_config.name

  tags = {
    Project = "nids-adversarial-evasion"
    Purpose = "ML Security Research"
  }
}

resource "aws_sagemaker_notebook_instance_lifecycle_config" "nids_config" {
  name = "nids-lifecycle-config"

  on_start = base64encode(<<-EOF
    #!/bin/bash
    set -e
    sudo -u ec2-user -i <<'SCRIPT'
    source /home/ec2-user/anaconda3/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install scikit-learn numpy matplotlib pandas flask pytest
    pip install adversarial-robustness-toolbox || true
    cd /home/ec2-user/SageMaker
    git clone https://github.com/yourusername/nids-adversarial-evasion-hardening.git || true
    SCRIPT
    EOF
  )
}

output "notebook_instance_url" {
  value = "https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/notebook-instances/openNotebook/nids-adversarial-evasion-lab?view=classic"
}

output "s3_bucket_name" {
  value = aws_s3_bucket.model_artifacts.bucket
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker_execution_role.arn
}
