
$python_ver = &{python -V} 2>&1
Write-Output "Python version "$python_ver

$start_dir = $pwd
Write-Output "Current directory: " $start_dir.Path

python -X importtime -c "import image2image_reg.cli"