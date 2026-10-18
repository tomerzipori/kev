param(
    [Parameter(Mandatory = $true)][string]$StopPath,
    [Parameter(Mandatory = $true)][int]$IntervalMilliseconds
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$counterPaths = @(
    '\GPU Adapter Memory(*)\Dedicated Usage',
    '\GPU Adapter Memory(*)\Shared Usage',
    '\GPU Process Memory(*)\Dedicated Usage',
    '\GPU Process Memory(*)\Shared Usage'
)
$clock = [System.Diagnostics.Stopwatch]::StartNew()
$nextSample = 0.0

while (-not (Test-Path -LiteralPath $StopPath)) {
    try {
        $os = Get-CimInstance -ClassName Win32_OperatingSystem -Property FreePhysicalMemory
        $counterSet = Get-Counter -Counter $counterPaths -MaxSamples 1
        $samples = @($counterSet.CounterSamples)
        $adapter = @($samples | Where-Object { $_.Path -match '\\GPU Adapter Memory\(' })
        $processMemory = @($samples | Where-Object { $_.Path -match '\\GPU Process Memory\(' })
        $shared = @($adapter | Where-Object { (($_.Path -split '\\')[-1]) -eq 'Shared Usage' })
        $dedicated = @($adapter | Where-Object { (($_.Path -split '\\')[-1]) -eq 'Dedicated Usage' })
        if ($shared.Count -eq 0 -or $dedicated.Count -eq 0) {
            throw 'GPU Adapter Memory shared or dedicated counters were not returned.'
        }

        $processNames = @{}
        Get-CimInstance -ClassName Win32_Process -Property ProcessId, Name | ForEach-Object {
            $processNames[[string]$_.ProcessId] = $_.Name
        }
        $processRows = @($processMemory | ForEach-Object {
            $pidMatch = [regex]::Match($_.InstanceName, '^pid_(\d+)_')
            $processId = if ($pidMatch.Success) { [int]$pidMatch.Groups[1].Value } else { $null }
            [ordered]@{
                counter = ($_.Path -split '\\')[-1]
                instance = $_.InstanceName
                process_id = $processId
                process_name = if ($null -ne $processId) { $processNames[[string]$processId] } else { $null }
                bytes = [long][math]::Round($_.CookedValue)
            }
        })
        $record = [ordered]@{
            status = 'sampled'
            observed_utc = [DateTime]::UtcNow.ToString('o')
            windows_available_physical_bytes = [long]$os.FreePhysicalMemory * 1024
            windows_adapter_shared_bytes = [long][math]::Round(($shared | Measure-Object -Property CookedValue -Sum).Sum)
            windows_adapter_dedicated_bytes = [long][math]::Round(($dedicated | Measure-Object -Property CookedValue -Sum).Sum)
            windows_adapter_counters = @($shared + $dedicated | ForEach-Object {
                [ordered]@{ counter = ($_.Path -split '\\')[-1]; instance = $_.InstanceName; bytes = [long][math]::Round($_.CookedValue) }
            })
            windows_gpu_process_counters = $processRows
        }
    }
    catch {
        $record = [ordered]@{
            status = 'failed'
            observed_utc = [DateTime]::UtcNow.ToString('o')
            error = $_.Exception.ToString()
            windows_available_physical_bytes = $null
            windows_adapter_shared_bytes = $null
            windows_adapter_dedicated_bytes = $null
            windows_adapter_counters = @()
            windows_gpu_process_counters = @()
        }
    }

    [Console]::Out.WriteLine(($record | ConvertTo-Json -Compress -Depth 6))
    [Console]::Out.Flush()
    $nextSample += $IntervalMilliseconds / 1000.0
    $sleepMilliseconds = [int][math]::Floor(($nextSample - $clock.Elapsed.TotalSeconds) * 1000)
    if ($sleepMilliseconds -gt 0) {
        [System.Threading.Thread]::Sleep($sleepMilliseconds)
    }
}
