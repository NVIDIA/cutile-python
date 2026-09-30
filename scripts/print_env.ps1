# Set output file
$outputFile = "print_env.txt"

# Clear/create the output file
"" | Set-Content -Path $outputFile

# OS Information
"***OS Information***" | Add-Content -Path $outputFile
$osInfo = Get-ComputerInfo -Property WindowsVersion, OsArchitecture, OsName, WindowsInstallDateUTC
"OS Name: $($osInfo.OsName)" | Add-Content -Path $outputFile
"OS Architecture: $($osInfo.OsArchitecture)" | Add-Content -Path $outputFile
"Windows Version: $($osInfo.WindowsVersion)" | Add-Content -Path $outputFile
"Install Date (UTC): $($osInfo.WindowsInstallDateUTC)" | Add-Content -Path $outputFile
"" | Add-Content -Path $outputFile

# GPU Information
"***GPU Information***" | Add-Content -Path $outputFile
try {
    $gpuOutput = & nvidia-smi 2>&1
    if ($LASTEXITCODE -ne 0) {
        "nvidia-smi not found or no NVIDIA GPU detected" | Add-Content -Path $outputFile
    }
    else {
        $gpuOutput | Add-Content -Path $outputFile
    }
}
catch {
    "nvidia-smi not found or no NVIDIA GPU detected" | Add-Content -Path $outputFile
}
"" | Add-Content -Path $outputFile

# CPU Information
"***CPU***" | Add-Content -Path $outputFile
$cpuInfo = Get-WmiObject Win32_Processor
"Name: $($cpuInfo.Name)" | Add-Content -Path $outputFile
"Cores: $($cpuInfo.NumberOfCores)" | Add-Content -Path $outputFile
"Logical Processors: $($cpuInfo.NumberOfLogicalProcessors)" | Add-Content -Path $outputFile
"Architecture: $($cpuInfo.Architecture)" | Add-Content -Path $outputFile
"Speed (MHz): $($cpuInfo.MaxClockSpeed)" | Add-Content -Path $outputFile
"" | Add-Content -Path $outputFile

# CMake
"***CMake***" | Add-Content -Path $outputFile
try {
    $cmakePath = (Get-Command cmake -ErrorAction SilentlyContinue).Source
    if ($cmakePath) {
        "cmake found at: $cmakePath" | Add-Content -Path $outputFile
        $cmakeOutput = & cmake --version 2>&1
        $cmakeOutput | Add-Content -Path $outputFile
    }
    else {
        "cmake not found" | Add-Content -Path $outputFile
    }
}
catch {
    "cmake not found" | Add-Content -Path $outputFile
}
"" | Add-Content -Path $outputFile

# g++
"***g++***" | Add-Content -Path $outputFile
try {
    $gppPath = (Get-Command g++ -ErrorAction SilentlyContinue).Source
    if ($gppPath) {
        "g++ found at: $gppPath" | Add-Content -Path $outputFile
        $gppOutput = & g++ --version 2>&1
        $gppOutput | Add-Content -Path $outputFile
    }
    else {
        "g++ not found" | Add-Content -Path $outputFile
    }
}
catch {
    "g++ not found" | Add-Content -Path $outputFile
}
"" | Add-Content -Path $outputFile

# nvcc
"***nvcc***" | Add-Content -Path $outputFile
try {
    $nvccPath = (Get-Command nvcc -ErrorAction SilentlyContinue).Source
    if ($nvccPath) {
        "nvcc found at: $nvccPath" | Add-Content -Path $outputFile
        $nvccOutput = & nvcc --version 2>&1
        $nvccOutput | Add-Content -Path $outputFile
    }
    else {
        "nvcc not found" | Add-Content -Path $outputFile
    }
}
catch {
    "nvcc not found" | Add-Content -Path $outputFile
}
"" | Add-Content -Path $outputFile

# Python
"***Python***" | Add-Content -Path $outputFile
try {
    $pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($pythonPath) {
        "python found at: $pythonPath" | Add-Content -Path $outputFile
        $pythonOutput = & python -c "import sys; print('Python {0}.{1}.{2}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))" 2>&1
        $pythonOutput | Add-Content -Path $outputFile
    }
    else {
        "python not found" | Add-Content -Path $outputFile
    }
}
catch {
    "python not found" | Add-Content -Path $outputFile
}
"" | Add-Content -Path $outputFile

# uv
"***uv***" | Add-Content -Path $outputFile
try {
    $uvPath = (Get-Command uv -ErrorAction SilentlyContinue).Source
    if ($uvPath) {
        "uv found at: $uvPath" | Add-Content -Path $outputFile
        $uvOutput = & uv --version 2>&1
        $uvOutput | Add-Content -Path $outputFile
    }
    else {
        "uv not found" | Add-Content -Path $outputFile
    }
}
catch {
    "uv not found" | Add-Content -Path $outputFile
}
"" | Add-Content -Path $outputFile

# Environment Variables
"***Environment Variables***" | Add-Content -Path $outputFile
$envVars = @(
    'PATH',
    'PYTHONPATH',
    'CONDA_PREFIX',
    'CUDA_PATH',
    'CUDA_HOME',
    'NUMBAPRO_NVVM',
    'NUMBAPRO_LIBDEVICE'
)

foreach ($var in $envVars) {
    $value = [System.Environment]::GetEnvironmentVariable($var)
    if ([string]::IsNullOrEmpty($value)) {
        $value = "(not set)"
    }
    ("{0,-32}: {1}" -f $var, $value) | Add-Content -Path $outputFile
}
"" | Add-Content -Path $outputFile

# Package Management
"***Python Packages***" | Add-Content -Path $outputFile
try {
    $condaPath = (Get-Command conda -ErrorAction SilentlyContinue).Source
    if ($condaPath) {
        "conda found at: $condaPath" | Add-Content -Path $outputFile
        "" | Add-Content -Path $outputFile
        $condaOutput = & conda list 2>&1
        $condaOutput | Add-Content -Path $outputFile
    }
    else {
        try {
            $pipPath = (Get-Command pip -ErrorAction SilentlyContinue).Source
            if ($pipPath) {
                "conda not found" | Add-Content -Path $outputFile
                "pip found at: $pipPath" | Add-Content -Path $outputFile
                "" | Add-Content -Path $outputFile
                $pipOutput = & pip list 2>&1
                $pipOutput | Add-Content -Path $outputFile
            }
            else {
                "conda not found" | Add-Content -Path $outputFile
                "pip not found" | Add-Content -Path $outputFile
            }
        }
        catch {
            "conda not found" | Add-Content -Path $outputFile
            "pip not found" | Add-Content -Path $outputFile
        }
    }
}
catch {
    "conda not found" | Add-Content -Path $outputFile
}
"" | Add-Content -Path $outputFile

Write-Host "Environment information has been written to $outputFile"