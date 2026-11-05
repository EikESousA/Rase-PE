# Rodada noturna dos estudos de robustez (Seção 4.x, subsec:robustez_metodo).
#
# Etapas, nesta ordem:
#   1. ablação de prompt (retoma o que faltar)
#   2. sementes 42 a 46 nos experimentos isolados EN1, EN2 e EN3
#   3. analyses.robustez evaluate  -> results/analises/robustez.json
#   4. analyses.robustez tables    -> dissertacao/Conteudo/Tabelas/{sementes,ablacao_prompt,custo_tokens}.tex
#
# Tudo é retomável: o que já foi gerado por completo é pulado. Pode rodar de novo
# depois de uma falha. Mantém o Windows acordado enquanto roda.
#
# Uso (PowerShell, na raiz do projeto ou em qualquer lugar):
#   powershell -ExecutionPolicy Bypass -File tools\rodar_noturno.ps1
# Resultado: logs\noturno_resumo.md (e um log por etapa em logs\noturno_*.log)

# 'Continue': no Windows PowerShell 5.1, com 'Stop', qualquer linha no stderr do Python abortaria o script.
$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Logs = Join-Path $Root 'logs'
New-Item -ItemType Directory -Force $Logs | Out-Null
$Resumo = Join-Path $Logs 'noturno_resumo.md'
$env:PYTHONIOENCODING = 'utf-8'
$env:GENERATE_DEBUG = '0'

# Impede a suspensão do sistema enquanto o script roda (a tela pode apagar).
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]'0x80000000'; $ES_SYSTEM_REQUIRED = [uint32]'0x00000001'
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null

$inicio = Get-Date
$linhas = [System.Collections.Generic.List[string]]::new()
function Registrar([string]$texto) {
    $carimbo = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    $linhaLog = "[$carimbo] $texto"
    $linhas.Add($linhaLog)
    Set-Content -Path $Resumo -Value ($linhas -join "`n") -Encoding utf8
    Add-Content -LiteralPath (Join-Path $Logs 'noturno_execucao.log') -Value $linhaLog -Encoding UTF8
    Write-Host $linhaLog
}
Registrar "# Rodada noturna — $($inicio.ToString('yyyy-MM-dd HH:mm'))"
Registrar ''

function Ollama-Ativo {
    try { Invoke-RestMethod -Uri 'http://localhost:11434/api/tags' -TimeoutSec 5 | Out-Null; return $true } catch { return $false }
}
if (-not (Ollama-Ativo)) {
    Registrar '- Ollama não respondia; iniciando `ollama serve`.'
    Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden
    Start-Sleep -Seconds 15
    if (-not (Ollama-Ativo)) { Registrar '- **FALHA:** Ollama não subiu. Nada foi executado.'; exit 1 }
}

# Cada etapa roda até 2 vezes (a segunda retoma do checkpoint da primeira).
function Etapa([string]$nome, [string[]]$argumentos, [int]$tentativas = 2) {
    $log = Join-Path $Logs "noturno_$nome.log"
    for ($t = 1; $t -le $tentativas; $t++) {
        $t0 = Get-Date
        Registrar "[INÍCIO] etapa=$nome tentativa=$t/$tentativas comando=python -u -m analyses.robustez $($argumentos -join ' ')"
        # -u desativa buffering do Python; Tee-Object deixa o progresso visível
        # no terminal enquanto preserva a cópia por etapa em logs/noturno_<etapa>.log.
        & $Python -u -m analyses.robustez @argumentos 2>&1 | Tee-Object -FilePath $log -Append
        $codigo = $LASTEXITCODE
        $dur = (Get-Date) - $t0
        $durTxt = '{0:hh\:mm\:ss}' -f $dur
        if ($codigo -eq 0) {
            Registrar "- [OK] **$nome** em $durTxt (tentativa $t); exit_code=0"
            return $true
        }
        Registrar "- [FALHA] **$nome**, exit_code=$codigo, após $durTxt (tentativa $t). Ver ``logs/noturno_$nome.log``."
        if (-not (Ollama-Ativo)) { Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep -Seconds 15 }
    }
    return $false
}

Registrar "Python: $Python"
Registrar "Logs detalhados: logs/noturno_execucao.log e logs/noturno_<etapa>.log"

$ok = $true
$ok = (Etapa 'ablacao' @('generate', '--study', 'ablacao')) -and $ok
foreach ($s in 42, 43, 44, 45, 46) {
    $ok = (Etapa "semente_$s" @('generate', '--study', 'sementes', '--seeds', "$s")) -and $ok
}

# Só avalia se todas as gerações terminaram; avaliação parcial misturaria execuções incompletas.
if ($ok) {
    if (Etapa 'evaluate' @('evaluate') 1) { Etapa 'tables' @('tables') 1 | Out-Null }
} else {
    Registrar '- evaluate e tables **não** rodaram porque alguma geração falhou. Rode o script de novo: ele retoma de onde parou.'
}

# Conferência do que existe no disco.
$nSementes = @(Get-ChildItem 'predicts\robustez\sementes\*.uso.jsonl' -ErrorAction SilentlyContinue).Count
$nAblacao = @(Get-ChildItem 'predicts\robustez\ablacao\*.uso.jsonl' -ErrorAction SilentlyContinue).Count
Registrar ''
Registrar "Arquivos de uso gerados: sementes $nSementes/90, ablação $nAblacao/24."
foreach ($f in 'results\analises\robustez.json', 'dissertacao\Conteudo\Tabelas\sementes.tex', 'dissertacao\Conteudo\Tabelas\ablacao_prompt.tex', 'dissertacao\Conteudo\Tabelas\custo_tokens.tex') {
    $marca = if (Test-Path $f) { 'existe' } else { 'FALTA' }
    Registrar "- ``$f``: $marca"
}
Registrar ''
Registrar "Duração total: $('{0:hh\:mm\:ss}' -f ((Get-Date) - $inicio)). Fim: $((Get-Date).ToString('yyyy-MM-dd HH:mm'))."

[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
