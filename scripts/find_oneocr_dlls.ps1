# PowerShell script to find OneOCR DLL files
# Run this script as Administrator for best results

Write-Host "Searching for OneOCR DLL files..." -ForegroundColor Cyan
Write-Host ""

$searchPaths = @(
    "C:\Program Files\WindowsApps",
    "C:\Program Files (x86)\WindowsApps",
    "$env:LOCALAPPDATA\Microsoft\WindowsApps"
)

$foundFiles = @{
    "oneocr.dll" = $null
    "oneocr.onemodel" = $null
    "onnxruntime.dll" = $null
}

foreach ($path in $searchPaths) {
    if (Test-Path $path) {
        Write-Host "Searching in: $path" -ForegroundColor Yellow
        try {
            # Search for oneocr.dll
            if ($null -eq $foundFiles["oneocr.dll"]) {
                $dll = Get-ChildItem -Path $path -Recurse -Filter "oneocr.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
                if ($dll) {
                    $foundFiles["oneocr.dll"] = $dll.FullName
                    Write-Host "  Found oneocr.dll: $($dll.FullName)" -ForegroundColor Green
                    
                    # Check same directory for other files
                    $dir = $dll.DirectoryName
                    $model = Join-Path $dir "oneocr.onemodel"
                    $onnx = Join-Path $dir "onnxruntime.dll"
                    
                    if (Test-Path $model) {
                        $foundFiles["oneocr.onemodel"] = $model
                        Write-Host "  Found oneocr.onemodel: $model" -ForegroundColor Green
                    }
                    if (Test-Path $onnx) {
                        $foundFiles["onnxruntime.dll"] = $onnx
                        Write-Host "  Found onnxruntime.dll: $onnx" -ForegroundColor Green
                    }
                }
            }
            
            # Search for remaining files if not found in same directory
            if ($null -eq $foundFiles["oneocr.onemodel"]) {
                $model = Get-ChildItem -Path $path -Recurse -Filter "oneocr.onemodel" -ErrorAction SilentlyContinue | Select-Object -First 1
                if ($model) {
                    $foundFiles["oneocr.onemodel"] = $model.FullName
                    Write-Host "  Found oneocr.onemodel: $($model.FullName)" -ForegroundColor Green
                }
            }
            
            if ($null -eq $foundFiles["onnxruntime.dll"]) {
                $onnx = Get-ChildItem -Path $path -Recurse -Filter "onnxruntime.dll" -ErrorAction SilentlyContinue | 
                    Where-Object { $_.DirectoryName -like "*oneocr*" -or $_.DirectoryName -like "*OneOCR*" } | 
                    Select-Object -First 1
                if ($onnx) {
                    $foundFiles["onnxruntime.dll"] = $onnx.FullName
                    Write-Host "  Found onnxruntime.dll: $($onnx.FullName)" -ForegroundColor Green
                }
            }
        } catch {
            Write-Host "  Error searching in $path : $_" -ForegroundColor Red
        }
    } else {
        Write-Host "Path does not exist: $path" -ForegroundColor Gray
    }
}

Write-Host ""
if ($foundFiles["oneocr.dll"] -and $foundFiles["oneocr.onemodel"] -and $foundFiles["onnxruntime.dll"]) {
    Write-Host "All DLL files found!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To copy these files, run:" -ForegroundColor Cyan
    Write-Host "  uv run python scripts/setup_oneocr_dlls.py `"$($foundFiles['oneocr.dll'])`" `"$($foundFiles['oneocr.onemodel'])`" `"$($foundFiles['onnxruntime.dll'])`""
} else {
    Write-Host "Could not find all required files:" -ForegroundColor Red
    foreach ($key in $foundFiles.Keys) {
        if ($null -eq $foundFiles[$key]) {
            Write-Host "  Missing: $key" -ForegroundColor Red
        }
    }
    Write-Host ""
    Write-Host "Please ensure OneOCR is installed via Microsoft Store or Windows Apps." -ForegroundColor Yellow
}
