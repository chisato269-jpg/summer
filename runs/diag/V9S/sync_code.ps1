# V9-S: Copy the launch code outside OneDrive (MUST-7, V9S-l). Edit code in the OneDrive project, then run this to sync.
# Usage: powershell -File sync_code.ps1
$src = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$dst = "C:\Users\chisa\rl_code\v9s"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
foreach ($d in "rlkit", "szklib", "Shizuku2", "verify") {
    robocopy "$src\$d" "$dst\$d" /MIR /XD "__pycache__" ".pytest_cache" /R:3 /W:2 /NFL /NDL /NJH /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy $d failed ($LASTEXITCODE)" }
}
"synced from $src to $dst"
Get-ChildItem $dst | Select-Object Name
