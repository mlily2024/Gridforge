@echo off
REM GridForge preprint build helper (Windows).
REM
REM Stages figures from scripts\output\, then runs Pandoc to produce
REM PDF and HTML outputs.

setlocal

set PAPER_DIR=%~dp0
set REPO_ROOT=%PAPER_DIR%..
set OUT_DIR=%REPO_ROOT%\scripts\output\07_paper

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo.
echo Staging figures...
python "%PAPER_DIR%build_figures.py"
if errorlevel 1 (
  echo Figure staging failed -- regenerate the source PNGs first.
  exit /b 1
)

where pandoc >nul 2>nul
if errorlevel 1 (
  echo Pandoc not found in PATH; install it first.
  exit /b 2
)

echo.
echo Building HTML...
pandoc ^
  --standalone ^
  --bibliography="%PAPER_DIR%references.bib" ^
  --citeproc ^
  --resource-path="%PAPER_DIR%" ^
  --metadata=link-citations:true ^
  -o "%OUT_DIR%\paper.html" ^
  "%PAPER_DIR%paper.md"
echo   HTML : %OUT_DIR%\paper.html

echo.
echo Building PDF (requires XeLaTeX)...
pandoc ^
  --pdf-engine=xelatex ^
  --bibliography="%PAPER_DIR%references.bib" ^
  --citeproc ^
  --resource-path="%PAPER_DIR%" ^
  --metadata=link-citations:true ^
  -o "%OUT_DIR%\paper.pdf" ^
  "%PAPER_DIR%paper.md"
if errorlevel 1 (
  echo PDF build failed ^(XeLaTeX missing?^). HTML output remains usable.
) else (
  echo   PDF  : %OUT_DIR%\paper.pdf
)

echo.
echo Done.
endlocal
