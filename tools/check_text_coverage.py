"""Per-page text coverage of a converted PDF: fraction of text-layer words (minus running headers/footers) found in the markdown.
Usage: python tools/check_text_coverage.py <file.pdf> <output.md>. Low pages are usually text inside figures."""
import re, subprocess, sys
from collections import Counter
from pathlib import Path
pdf, md_path = Path(sys.argv[1]), Path(sys.argv[2])
md = md_path.read_text()
# undo LaTeX/HTML sub/superscript formatting so V_{REF} -> VREF, I^{2}C / I<sup>2</sup>C / I²C -> I2C
md = re.sub(r'\\(?:text|mathrm|mathit|textit|mathbf)\s*', '', md)
md = re.sub(r'</?su[bp]>|[_^{}]', '', md).replace('²', '2').replace('\\,', '').replace('\\ ', ' ')
# split markdown into pages using the separator lines "*Page N*"
parts = re.split(r'\n\*Page (\d+)\*\n', md)
md_pages = {1: parts[0]}
for num, body in zip(parts[1::2], parts[2::2]):
    md_pages[int(num)] = body
norm = lambda s: re.findall(r"[a-z0-9]+", s.lower().replace('µ', 'u'))
n = int(re.search(r'Pages:\s+(\d+)', subprocess.run(['pdfinfo', pdf], capture_output=True, text=True).stdout).group(1))
texts = {p: subprocess.run(['pdftotext', '-layout', '-f', str(p), '-l', str(p), pdf, '-'], capture_output=True, text=True).stdout for p in range(1, n + 1)}
# words on >=40% of pages are running headers/footers, intentionally dropped
page_freq = Counter(w for t in texts.values() for w in set(norm(t)))
boiler = {w for w, c in page_freq.items() if n >= 4 and c >= 0.4 * n}
tot_have = tot_all = 0
for p in range(1, n + 1):
    src = [w for w in norm(texts[p]) if len(w) >= 3 and not w.isdigit() and w not in boiler]
    have = Counter(norm(md_pages.get(p, '')))
    missing = [w for w in src if w not in have]
    tot_have += len(src) - len(missing); tot_all += len(src)
    cov = 1 - len(missing) / max(1, len(src))
    flag = '' if cov > 0.9 else '  <--'
    print(f"p{p:<3} words={len(src):4} coverage={cov:6.1%} missing={' '.join(sorted(set(missing))[:25])}{flag}")
print(f"TOTAL coverage {tot_have / max(1, tot_all):.1%} of {tot_all} words")
