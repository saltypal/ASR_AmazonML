param(
    [Parameter(Mandatory = $true)]
    [string]$Bucket,

    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [string]$Region = "ap-south-1"
)

$ErrorActionPreference = "Stop"
$resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$dataset = Join-Path $resolvedRoot "dataset"
if (-not (Test-Path -LiteralPath (Join-Path $dataset "train\train_source1.tsv"))) {
    throw "The supplied project root does not contain dataset\train\train_source1.tsv"
}

aws sts get-caller-identity | Out-Host
aws s3 sync $dataset "s3://$Bucket/dataset/" --region $Region --only-show-errors
if ($LASTEXITCODE -ne 0) {
    throw "Dataset upload failed with exit code $LASTEXITCODE"
}
Write-Host "Uploaded the challenge dataset to s3://$Bucket/dataset/"
