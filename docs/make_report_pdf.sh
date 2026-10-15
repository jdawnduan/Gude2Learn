#!/usr/bin/env bash
# docs/REPORT.md -> docs/REPORT.pdf: A4, two columns, small type, so the report stays within the organisers'
# 1-2 pages. Needs uv (fetches the markdown package into a throwaway env, nothing added to the project) and
# Google Chrome. Run after every edit of REPORT.md:  ./docs/make_report_pdf.sh
set -eu
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
TMP=/tmp/report_pdf_tmp.html
{
cat <<'HTML'
<!doctype html><meta charset="utf-8"><title>Where Apertus had difficulty</title>
<style>
@page{size:A4;margin:10mm 11mm}
body{font:8.5pt/1.28 "Helvetica Neue",Arial,sans-serif;color:#111;column-count:2;column-gap:6mm;margin:0}
h1{font-size:14pt;margin:0 0 3pt;column-span:all}
h2{font-size:9.5pt;margin:6pt 0 2pt;break-after:avoid}
p,ul,ol{margin:0 0 3.5pt} li{margin:0 0 1pt} ul,ol{padding-left:11pt}
code{font-family:Menlo,Consolas,monospace;font-size:7.6pt;white-space:nowrap}
table{border-collapse:collapse;font-size:7.6pt;margin:2pt 0 4pt;width:100%}
td,th{border-bottom:.4pt solid #999;padding:1pt 3pt;text-align:left;vertical-align:top}
</style>
HTML
uv run --with markdown python -m markdown -x tables REPORT.md
} > "$TMP"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/REPORT.pdf" "file://$TMP" >/dev/null 2>&1
echo "wrote docs/REPORT.pdf"
