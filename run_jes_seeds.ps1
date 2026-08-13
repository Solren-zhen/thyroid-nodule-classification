$wd = Join-Path $env:USERPROFILE "Desktop\thyroid"
Set-Location $wd
$py = "C:\miniconda3\envs\lymph_yolo\python.exe"

$jobs = @(
  @{ seed = "123";  root = "data/thyroid_jes_s123";  save = "checkpoints/thyroid/jes_control_s123";  log = "logs/train_jes_s123.log" },
  @{ seed = "2024"; root = "data/thyroid_jes_s2024"; save = "checkpoints/thyroid/jes_control_s2024"; log = "logs/train_jes_s2024.log" }
)

foreach ($j in $jobs) {
  Write-Output ("=== TRAIN seed " + $j.seed + " ===")
  & $py -u train_thyroid.py --data_root $j.root --ablation image --epochs 30 --batch_size 64 --workers 2 --seed $j.seed --pos_weight 1.0 --save_dir $j.save 2>&1 | Out-File -FilePath (Join-Path $wd $j.log) -Encoding utf8
  Write-Output ("=== DONE seed " + $j.seed + " ===")
}
Write-Output "ALL JES SEED TRAININGS DONE"