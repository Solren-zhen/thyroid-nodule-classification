$wd = Join-Path $env:USERPROFILE "Desktop\thyroid"
Set-Location $wd
$py = "C:\miniconda3\envs\lymph_yolo\python.exe"

$arms = @(
  @{ name = "ablation_img_tirads"; cols = "tirads" },
  @{ name = "ablation_img_agesex"; cols = "age,gender" },
  @{ name = "ablation_img_size";   cols = "width_mm,height_mm" }
)

foreach ($arm in $arms) {
  $log = Join-Path $wd ("logs\train_" + $arm.name + ".log")
  Write-Output ("=== TRAIN " + $arm.name + " ===")
  & $py -u train_thyroid.py --data_root data/thyroid/thyroidxl --ablation fusion --clinical_columns $arm.cols --epochs 30 --batch_size 32 --workers 4 --seed 42 --pos_weight 1.0 --save_dir ("checkpoints/thyroid/" + $arm.name) 2>&1 | Out-File -FilePath $log -Encoding utf8
  Write-Output ("=== DONE " + $arm.name + " ===")
}
Write-Output "ALL FEATURE ABLATIONS DONE"