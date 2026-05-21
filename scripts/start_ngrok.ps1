$ErrorActionPreference = "Stop"

$ngrokExe = if ($env:NGROK_BIN) { $env:NGROK_BIN } else { "$env:LOCALAPPDATA\ngrok\ngrok.exe" }
$port = if ($env:WALLY_API_PORT) { $env:WALLY_API_PORT } else { "8000" }
$domain = if ($env:NGROK_DOMAIN) { $env:NGROK_DOMAIN } else { "wallyfinder-api.ngrok.app" }

if (-not (Test-Path $ngrokExe)) {
    Write-Error "ngrok not found at $ngrokExe. Install from https://ngrok.com/download or set NGROK_BIN."
}

& $ngrokExe http $port --url "https://$domain"
