# AWS execution guide

AWS Skill Builder and AWS Builder ID are learning and identity products. Run this project
in the regular AWS Management Console for the AWS account you created. The initial path is
S3 plus one EC2 GPU instance; the same Python package and YAML configuration used by Kaggle
run on AWS.

## Architecture

```text
VS Code / GitHub ───────► EC2 G5 Deep Learning AMI
       │                         │
       │                         ├─ temporary Parquet / embeddings on EBS
       │                         ├─ dual or multi GPU E5 encoding
       │                         └─ GPU XGBoost + validated TSV outputs
       │                                      │
       └──────── dataset/results ◄─────────────┘
                         S3 private bucket
```

S3 stores the immutable input and durable run artifacts. EBS stores large intermediate
files and is deleted after the run. GitHub stores source code. Never commit the dataset or
AWS credentials.

## One-time account setup

1. Sign in to the normal **AWS Management Console** with the new AWS account.
2. Enable MFA on the root user, create an administrative IAM identity for daily work, and
   create an AWS Budget alert before launching a GPU.
3. Open **CloudFormation**, choose **Create stack**, upload `storage-and-role.yaml`, and
   create the stack in one region. `ap-south-1` is convenient from India, but first confirm
   G5 availability and quota in that region.
4. Record the stack outputs: private bucket name and EC2 instance profile name.
5. In **Service Quotas → Amazon EC2**, verify the quota for G and VT On-Demand instances.
   Request an increase before the training day if the quota is zero.

The template blocks public access, enables S3 encryption and versioning, retains the bucket
if the stack is removed, and grants the EC2 role access only to that bucket.

## Upload the dataset from the VS Code terminal

Install AWS CLI v2, then run `aws configure sso` when using IAM Identity Center, or
`aws configure` for a dedicated IAM user. Do not use root access keys.

```powershell
cd D:\Bunker\BaseCamp\AmazonML
powershell -ExecutionPolicy Bypass -File .\code\business_entity_resolution\infra\aws\upload_dataset.ps1 `
  -Bucket YOUR_STACK_BUCKET `
  -ProjectRoot D:\Bunker\BaseCamp\AmazonML `
  -Region ap-south-1
```

This uploads the original TSV files. There is no CSV conversion step.

## Launch the GPU machine

1. Open **EC2 → Launch instance**.
2. Select the current AWS Deep Learning AMI for Ubuntu with PyTorch and CUDA.
3. For the first complete run choose **g5.12xlarge** (4 A10G GPUs) if quota and budget
   allow it. Use **g5.2xlarge** for a cheaper one-GPU engineering run. The code detects the
   number of GPUs automatically.
4. Attach the CloudFormation EC2 instance profile.
5. Allocate at least **350 GiB gp3** EBS for the original data, normalized Parquet,
   candidates, features, and temporary model files.
6. Restrict SSH in the security group to your current public IP. A Session Manager setup is
   preferable when available.
7. Use On-Demand for the first full run. The current stage runner does not resume an
   interrupted candidate stage, so an interrupted Spot instance can waste hours.

Connect through VS Code Remote SSH or the EC2 terminal and run:

```bash
export BER_BUCKET="YOUR_STACK_BUCKET"
export BER_REGION="ap-south-1"
export BER_GIT_REF="main"                 # replace with a tested commit hash for the final run
export BER_RUN_ID="aws-final-001"
curl -fsSL https://raw.githubusercontent.com/saltypal/ASR_AmazonML/main/code/business_entity_resolution/infra/aws/run_on_ec2.sh -o run_on_ec2.sh
bash run_on_ec2.sh
```

For an auditable final run, replace `main` with the exact tested Git commit. The script
downloads the dataset from S3, pins the E5 model revision, runs all stages, invokes the
official validator, and uploads outputs, model metadata, run metadata, and logs to:

```text
s3://YOUR_STACK_BUCKET/runs/YOUR_RUN_ID/
```

## Download and verify the results

```powershell
aws s3 sync s3://YOUR_STACK_BUCKET/runs/aws-final-001/ .\aws-results\aws-final-001\
python .\utils\validate_submission.py `
  --matching .\aws-results\aws-final-001\output\matching_results.tsv `
  --candidate .\aws-results\aws-final-001\output\candidate_pairs.tsv `
  --test-dir .\dataset\test
```

Stop the EC2 instance while inspecting results. Terminate it after all required artifacts
are in S3. EBS and idle GPU instances continue to cost money until removed.

## Team workflow

- Share code through GitHub branches and pull requests.
- Give teammates their own AWS identities or roles; never share passwords or access keys.
- Give the team access to only the project bucket and required EC2 actions.
- Give every experiment a unique run ID and record its Git commit, YAML config, hardware,
  validation score, holdout score, and output paths.
- Compare Kaggle and AWS runs by commit and config hash. Hardware may change runtime, but
  the data transformations and evaluation definition remain identical.
