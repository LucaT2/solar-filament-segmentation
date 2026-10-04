# Full unattended pipeline for one experiment (Windows PowerShell):
#   train folds -> OOF maps with TTA -> tune post-processing -> fit rescorer -> test maps with TTA -> submission
#
# Run detached so it survives closing VS Code / the terminal (keep the laptop awake and plugged in):
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy','Bypass','-File','scripts\run_pipeline.ps1','-Folds','1,2,3,4'
#
# Progress: outputs\<exp>\pipeline.log (and log_fold<k>.csv per fold). Each step stops the pipeline if it fails.

param(
    [string]$Config = "configs\unet_r34.yaml",
    [string]$Exp = "unet_r34",
    [string]$Folds = "0,1,2,3,4",   # folds to train; "" = skip training (reuse existing checkpoints)
    [switch]$NoTTA
)

# Native programs write progress bars to stderr; Windows PowerShell 5.1 would turn that into errors under "Stop",
# so success is judged by exit codes only, and cmd.exe does the redirection (keeps the log UTF-8, not UTF-16).
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force "outputs\$Exp" | Out-Null
$Log = Join-Path $Root "outputs\$Exp\pipeline.log"
$Tta = if ($NoTTA) { @() } else { @("--tta") }

function Step([string]$Name, [string[]]$PyArgs) {
    "`n==== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Name ====" | Out-File $Log -Append -Encoding utf8
    $cmdline = "`"$Py`" -W ignore " + (($PyArgs | ForEach-Object { "`"$_`"" }) -join " ") + " >> `"$Log`" 2>&1"
    cmd.exe /c $cmdline
    if ($LASTEXITCODE -ne 0) {
        "!!!! $Name failed with exit code $LASTEXITCODE" | Out-File $Log -Append -Encoding utf8
        exit $LASTEXITCODE
    }
}

"pipeline started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  config=$Config exp=$Exp folds=$Folds tta=$(-not $NoTTA)" |
    Out-File $Log -Append -Encoding utf8

if ($Folds) {
    $foldList = $Folds.Split(",") | ForEach-Object { $_.Trim() }
    Step "train folds $Folds" (@("scripts\train.py", "--config", $Config, "--fold") + $foldList)
}
Step "OOF maps" (@("scripts\predict.py", "--exp", $Exp, "--split", "oof") + $Tta)
Step "tune post-processing" @("scripts\tune_postprocess.py", "--exp", $Exp, "--thresholds", "0.4", "0.5", "0.6",
                              "--min-areas", "100", "200", "300", "400", "500")
Step "fit rescorer" @("scripts\fit_rescorer.py", "--exp", $Exp)
Step "test maps" (@("scripts\predict.py", "--exp", $Exp, "--split", "test") + $Tta)
Step "submission" @("scripts\make_submission.py", "--exp", $Exp)
# The final rescorer was fitted on these same OOF components, so scoring OOF *with* it would be in-sample (leaky).
# The CSV check below therefore runs without it; the honest PQ with rescoring is "oof_pq_nested" in rescorer.json.
Step "OOF submission (no rescorer)" @("scripts\make_submission.py", "--exp", $Exp, "--split", "oof", "--no-rescore")
$trained = Get-ChildItem "outputs\$Exp\fold*.pt" | ForEach-Object { $_.BaseName.Substring(4) }
Step "evaluate OOF without rescorer (folds $($trained -join ','))" (@("scripts\evaluate.py", "--pred",
                                                   "outputs\$Exp\oof_submission.csv", "--fold") + $trained)
"`nhonest OOF PQ with rescorer (nested): see outputs\$Exp\rescorer.json -> oof_pq_nested" | Out-File $Log -Append -Encoding utf8

"`npipeline finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') -> outputs\$Exp\submission.csv" | Out-File $Log -Append -Encoding utf8
