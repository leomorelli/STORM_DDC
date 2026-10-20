param(
    [ValidateRange(1, 128)]
    [int]$Workers = 4,
    [string]$InputMat = 'C:\Users\maula\Desktop\PhD Files\Codes\DDC-main\Main_DDC_Folder\DDC_Tools\Split_Example_3d_2darkstate_random_data.mat',
    [string]$OutputMat = (Join-Path $PSScriptRoot 'results\python_same_split_optimized.mat'),
    [string]$PythonExe = 'C:\Users\maula\miniforge3\envs\ddc-parallel\python.exe'
)

# One numerical-library thread per ROI process avoids nested thread pools.
$taskThreadVariables = @('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
$taskPreviousValues = @{}
foreach ($taskVariable in $taskThreadVariables) {
    $taskPreviousValues[$taskVariable] = [Environment]::GetEnvironmentVariable($taskVariable, 'Process')
    [Environment]::SetEnvironmentVariable($taskVariable, '1', 'Process')
}

# Use this source tree even if a different DDC package is installed elsewhere.
$taskPreviousPythonPath = [Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
$taskSourcePath = Join-Path $PSScriptRoot 'src'
$taskPythonPath = if ($taskPreviousPythonPath) { "$taskSourcePath;$taskPreviousPythonPath" } else { $taskSourcePath }
[Environment]::SetEnvironmentVariable('PYTHONPATH', $taskPythonPath, 'Process')
try {
    & $PythonExe -m ddc run $InputMat --output $OutputMat --workers $Workers --resolution 80 --stepper 100 --n-f 200 --timer 720 --seed 123
    if ($LASTEXITCODE -ne 0) { throw "DDC exited with code $LASTEXITCODE" }
}
finally {
    foreach ($taskVariable in $taskThreadVariables) {
        [Environment]::SetEnvironmentVariable($taskVariable, $taskPreviousValues[$taskVariable], 'Process')
    }
    [Environment]::SetEnvironmentVariable('PYTHONPATH', $taskPreviousPythonPath, 'Process')
}
