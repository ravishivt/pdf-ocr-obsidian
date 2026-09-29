import os
import io
import json
import base64
import hashlib
import httpx
import math
import shutil
import subprocess
import tempfile
import zipfile
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
from google import genai
from google.genai import errors as genai_errors, types as genai_types
from mistralai import Mistral, DocumentURLChunk
from mistralai.models import OCRResponse
from PIL import Image, ImageStat
from pypdf import PdfReader, PdfWriter
from werkzeug.utils import secure_filename
from dotenv import load_dotenv, set_key

load_dotenv()

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = Path(os.getenv('UPLOAD_FOLDER', 'uploads'))
OUTPUT_FOLDER = Path(os.getenv('OUTPUT_FOLDER', 'output'))
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {'pdf'}
PAGE_SEPARATOR_DEFAULT = os.getenv('PAGE_SEPARATOR', '---')

# OCR engines: env var holding each engine's API key (None = uses a logged-in CLI), and where to get one
ENGINES = {
    'claude-code': {'label': 'Claude Code (Claude subscription)', 'key_env': None, 'key_url': None},
    'antigravity': {'label': 'Antigravity (Google AI subscription)', 'key_env': None, 'key_url': None},
    'gemini': {'label': 'Gemini API', 'key_env': 'GEMINI_API_KEY', 'key_url': 'https://aistudio.google.com/apikey'},
    'mistral': {'label': 'Mistral API', 'key_env': 'MISTRAL_API_KEY', 'key_url': 'https://console.mistral.ai/api-keys'},
}
DEFAULT_ENGINE = os.getenv('OCR_ENGINE', 'claude-code')

# Agent CLI engines (claude / agy print mode). Each request runs one agent session on a page chunk.
CLAUDE_CODE_MODEL = os.getenv('CLAUDE_CODE_MODEL', 'opus')
ANTIGRAVITY_MODEL = os.getenv('ANTIGRAVITY_MODEL', 'gemini-3.1-pro-high')
CLI_PAGES_PER_REQUEST = int(os.getenv('CLI_PAGES_PER_REQUEST', 5))
CLI_CONCURRENCY = int(os.getenv('CLI_CONCURRENCY', 3))
CLI_TIMEOUT_S = 900

# Gemini engine settings. GEMINI_MODEL is a comma-separated fallback chain: when a model is
# overloaded, retired, or has no quota, the next one is used.
GEMINI_MODELS = [m.strip() for m in os.getenv(
    'GEMINI_MODEL', 'gemini-3.6-flash,gemini-3.5-flash,gemini-3-flash-preview,gemini-3.5-flash-lite,gemini-3.1-flash-lite'
).split(',') if m.strip()]
# Free tier allows ~20 requests/day per model, so send many pages per request
GEMINI_PAGES_PER_REQUEST = int(os.getenv('GEMINI_PAGES_PER_REQUEST', 10))
GEMINI_CONCURRENCY = int(os.getenv('GEMINI_CONCURRENCY', 3))
# Per-request timeout; a hung request falls through to the next model
GEMINI_TIMEOUT_MS = 420_000
PAGE_CACHE_FOLDER = Path(os.getenv('CACHE_FOLDER', '.cache')) / 'pages'
# Figures are cropped from pages rendered at max(FIGURE_DPI, highest embedded image ppi on the page)
FIGURE_DPI = int(os.getenv('FIGURE_DPI', 300))
FIGURE_MAX_DPI = 600
# Margin added around model-predicted figure boxes, in 0-1000 normalized page units
FIGURE_BOX_PADDING = 8

# Images appearing on this many+ distinct pages are treated as headers/footers/watermarks
REPEATED_PAGE_THRESHOLD = 3

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# --- Helper Functions ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def replace_images_in_markdown(markdown_str: str, image_mapping: dict) -> str:
    """Replace Mistral image refs ![id](id) with standard markdown ![](images/filename)."""
    for original_id, new_name in image_mapping.items():
        markdown_str = markdown_str.replace(
            f"![{original_id}]({original_id})",
            f"![](images/{new_name})"
        )
    return markdown_str

def is_blank_pil(img: Image.Image) -> bool:
    """Return True if the image is near-uniform color (no real content)."""
    return ImageStat.Stat(img.convert('L')).stddev[0] < 5

def is_blank_image(img_bytes: bytes) -> bool:
    """Return True if the image contains only whitespace (near-uniform color, no real content)."""
    try:
        return is_blank_pil(Image.open(io.BytesIO(img_bytes)))
    except Exception:
        return False


def get_page_count(pdf_path: Path) -> int:
    """Return the PDF's page count via pdfinfo (poppler), or 0 if it can't be determined."""
    try:
        info_result = subprocess.run(
            ['pdfinfo', str(pdf_path)], capture_output=True, text=True, check=True
        )
        pages_match = re.search(r'Pages:\s+(\d+)', info_result.stdout)
        return int(pages_match.group(1)) if pages_match else 0
    except FileNotFoundError:
        raise RuntimeError(
            "pdfinfo not found. Install poppler:\n"
            "  macOS: brew install poppler\n"
            "  Ubuntu/Debian: sudo apt-get install poppler-utils"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"pdfinfo failed: {e.stderr}")


def extract_images_pdfimages(pdf_path: Path, images_dir: Path, pdf_base_sanitized: str) -> dict:
    """
    Extract embedded images from a PDF using pdfimages (poppler).

    Filtering applied:
    - Skips blank/whitespace images (near-uniform pixel values)
    - Skips images whose content hash appears on REPEATED_PAGE_THRESHOLD or more distinct pages
      (catches repeated headers/footers/watermarks that appear throughout the document)
    - Deduplicates: identical image content is saved only on its first occurrence

    Naming: {pdf_base}_p{page}_img{n}.png, where n is 1-based per page.

    Returns:
        dict mapping 1-based page number -> list of saved image filenames
    """
    page_count = get_page_count(pdf_path)
    if page_count == 0:
        print(f"  pdfimages: could not determine page count for {pdf_path.name}")
        return {}

    # Pass 1: extract images page by page, collect raw data + page-presence counts
    hash_page_count: dict[str, set] = {}
    raw: list[tuple] = []  # (page_num, img_bytes, content_hash)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for page_num in range(1, page_count + 1):
            prefix = str(tmp_path / 'img')
            subprocess.run(
                ['pdfimages', '-png', '-f', str(page_num), '-l', str(page_num), str(pdf_path), prefix],
                capture_output=True
            )
            for img_file in sorted(tmp_path.glob('img-*.png')):
                img_bytes = img_file.read_bytes()
                img_file.unlink()
                if is_blank_image(img_bytes):
                    continue
                content_hash = hashlib.md5(img_bytes).hexdigest()
                hash_page_count.setdefault(content_hash, set()).add(page_num)
                raw.append((page_num, img_bytes, content_hash))

    # Pass 2: save images — skip repeated elements and exact duplicates
    images_by_page: dict[int, list[str]] = {}
    seen_hashes: set[str] = set()
    page_counters: dict[int, int] = {}

    for page_num, img_bytes, content_hash in raw:
        if len(hash_page_count[content_hash]) >= REPEATED_PAGE_THRESHOLD:
            continue  # Appears too many times — likely a header/footer/watermark
        if content_hash in seen_hashes:
            continue  # Exact duplicate already saved
        seen_hashes.add(content_hash)

        local_idx = page_counters.get(page_num, 0) + 1
        page_counters[page_num] = local_idx
        filename = f"{pdf_base_sanitized}_p{page_num}_img{local_idx}.png"
        try:
            (images_dir / filename).write_bytes(img_bytes)
            images_by_page.setdefault(page_num, []).append(filename)
        except OSError as e:
            print(f"  Warning: could not write image {filename}: {e}")

    total = sum(len(v) for v in images_by_page.values())
    print(f"  pdfimages: saved {total} images across {len(images_by_page)} pages (after filtering/dedup).")
    return images_by_page


# --- LLM Engines (Gemini API, Claude Code, Antigravity) ---

TRANSCRIBE_PROMPT = """Convert this PDF to Markdown for an Obsidian vault. The attached PDF contains {count} page(s): pages {first} to {last} of the original document.

For each page, in order, output a marker line exactly like `<<<PAGE n>>>` (n is the ORIGINAL page number, starting at {first}), followed by that page's content as Markdown. Output a marker for every page, even blank ones.

Rules:
- Transcribe all text verbatim and completely, in natural reading order (handle multi-column layouts). Do not summarize, paraphrase, translate, or add commentary.
- Use Markdown headings that reflect the document's section hierarchy, and Markdown lists for lists.
- Tables: GitHub-flavored Markdown tables, preserving every row, column, and cell value, including units, footnote markers, and symbols (±, µ, Ω, °). For merged cells, repeat the value in each cell it spans. Use an HTML <table> only if a table cannot be represented in Markdown.
- Math and equations: LaTeX, with $...$ inline and $$...$$ for display.
- Skip running page headers, footers, and page numbers (document IDs, copyright lines, "Submit Document Feedback", etc.).
- Skip purely decorative elements (logos, icons, background art).
- Figures (photos, diagrams, schematics, plots, charts, mechanical drawings, block diagrams): do not transcribe text inside them. At the figure's position in reading order, insert exactly:
  ![brief description](box:ymin,xmin,ymax,xmax)
  where ymin,xmin,ymax,xmax are integers giving the figure's bounding box on that page, normalized to 0-1000 with (0,0) at the top-left. The box must fully enclose the figure including its axis labels, dimension lines, and legends (err on the side of slightly too large, never cut anything off), but not its caption. Transcribe the caption (e.g. "Figure 7-3. Efficiency vs Output Current") as normal text right after the placeholder.
- Output only the page markers and Markdown. Do not wrap the output in code fences."""

GEMINI_CONFIG = genai_types.GenerateContentConfig(
    automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
)

# Agent CLIs get the page chunk as a file rather than inline
CLI_PROMPT_SUFFIX = ("\n\nThe PDF is the file chunk.pdf in the current directory. Read all of its pages "
                     "(including the page images), then reply with ONLY the transcription in the format above: "
                     "no preamble, no summary, no commentary.")

# Cached pages are only reused when the prompt they were produced with is unchanged
PROMPT_HASH = hashlib.sha256(TRANSCRIBE_PROMPT.encode()).hexdigest()[:12]

PAGE_MARKER_RE = re.compile(r'^\s*<<<PAGE (\d+)>>>[ \t]*$', re.MULTILINE)
FIGURE_RE = re.compile(r'!\[([^\]]*)\]\(box:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)')


def pdf_page_range_bytes(pdf_bytes: bytes, first: int, last: int, page_count: int) -> bytes:
    """Return a PDF containing only pages first..last (1-based, inclusive)."""
    if first == 1 and last == page_count:
        return pdf_bytes
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    for i in range(first - 1, last):
        writer.add_page(reader.pages[i])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def parse_page_markers(text: str, first: int, last: int) -> dict[int, str]:
    """Split model output on <<<PAGE n>>> markers into {page_num: markdown}."""
    text = re.sub(r'^\s*```(?:markdown|md)?\s*\n|\n```\s*$', '', text)
    parts = PAGE_MARKER_RE.split(text)
    found = [(int(num), body.strip()) for num, body in zip(parts[1::2], parts[2::2])]
    nums = [n for n, _ in found]
    # Tolerate the model numbering pages relative to the excerpt (1..count) instead of the original
    count = last - first + 1
    if first != 1 and nums and all(1 <= n <= count for n in nums) and not any(first <= n <= last for n in nums):
        found = [(n + first - 1, body) for n, body in found]
    pages = {n: body for n, body in found if first <= n <= last}
    if not found and first == last and text.strip():
        pages[first] = text.strip()  # Single page, model omitted the marker
    return pages


def is_quota_exhausted(err: genai_errors.APIError) -> bool:
    """True for a 429 that waiting a minute won't fix (daily quota hit, or no free-tier quota at all)."""
    if 'limit: 0' in (err.message or ''):
        return True
    for detail in (err.details or {}).get('error', {}).get('details', []):
        for violation in detail.get('violations', []):
            if 'PerDay' in violation.get('quotaId', ''):
                return True
    return False


def retry_delay_seconds(err: genai_errors.APIError, default: float) -> float:
    """Server-suggested retry delay from a 429's RetryInfo, e.g. "52s"."""
    for detail in (err.details or {}).get('error', {}).get('details', []):
        match = re.fullmatch(r'([\d.]+)s', detail.get('retryDelay', ''))
        if match:
            return float(match.group(1))
    return default


class GeminiModelChain:
    """
    Calls Gemini, trying GEMINI_MODELS in order.

    - Overloaded / timed out (5xx, timeout): try the next model; retry the chain in later rounds.
    - Retired (404) or out of quota (429 with daily/zero limit): drop the model for the rest of the run.
    - Per-minute rate limit (429): wait the server-suggested delay and retry the same model.
    """
    ROUNDS = 4
    MAX_RATE_LIMIT_WAITS = 4

    def __init__(self, client, models: list[str]):
        self.client = client
        self.models = models
        self.dropped: set[str] = set()
        self.lock = threading.Lock()

    def _drop(self, model: str, reason: str):
        with self.lock:
            if model not in self.dropped:
                print(f"  Gemini: {model} unusable ({reason}); skipping it from now on.")
                self.dropped.add(model)

    def generate(self, contents, config):
        last_err: Exception | None = None
        for round_num in range(self.ROUNDS):
            if round_num:
                delay = 15 * 2 ** (round_num - 1)
                print(f"  Gemini: all models busy; retrying in {delay}s (round {round_num + 1}/{self.ROUNDS}).")
                time.sleep(delay)
            for model in self.models:
                if model in self.dropped:
                    continue
                rate_limit_waits = 0
                while True:
                    try:
                        return self.client.models.generate_content(model=model, contents=contents, config=config)
                    except httpx.TimeoutException as e:
                        last_err = e
                        print(f"  Gemini: {model} timed out; trying next model.")
                        break
                    except genai_errors.APIError as e:
                        last_err = e
                        if e.code == 429 and not is_quota_exhausted(e) and rate_limit_waits < self.MAX_RATE_LIMIT_WAITS:
                            rate_limit_waits += 1
                            delay = min(retry_delay_seconds(e, 30), 90) + 1
                            print(f"  Gemini: {model} rate limited; waiting {delay:.0f}s.")
                            time.sleep(delay)
                        elif e.code in (404, 429):
                            self._drop(model, f"{e.code} {e.status}")
                            break
                        elif e.code in (500, 502, 503, 504):
                            print(f"  Gemini: {model} busy ({e.code}); trying next model.")
                            break
                        else:
                            raise
            if all(m in self.dropped for m in self.models):
                break
        if last_err:
            raise last_err
        raise RuntimeError(f"All Gemini models are unavailable: {', '.join(self.models)}")


def page_cache_path(cache_dir: Path, engine: str, page_num: int) -> Path:
    return cache_dir / f"{engine}_{PROMPT_HASH}_p{page_num}.md"


def transcribe_range(transcriber, engine: str, pdf_bytes: bytes, first: int, last: int, page_count: int,
                     raw_log: list, cache_dir: Path) -> dict[int, str]:
    """
    Transcribe pages first..last with `transcriber(chunk_pdf_bytes, prompt) -> (text, finish_reason, model)`.
    Splits the range in half and retries if output comes back incomplete. Complete pages are written to
    cache_dir so a re-run doesn't spend quota on them again.
    """
    count = last - first + 1
    label = ENGINES[engine]['label']
    print(f"  {label}: transcribing pages {first}-{last}...")
    started = time.time()
    text, finish, model = transcriber(pdf_page_range_bytes(pdf_bytes, first, last, page_count),
                                      TRANSCRIBE_PROMPT.format(count=count, first=first, last=last))
    raw_log.append({'pages': [first, last], 'model': model, 'finish_reason': finish,
                    'seconds': round(time.time() - started), 'text': text})

    pages = parse_page_markers(text, first, last)
    missing = [p for p in range(first, last + 1) if p not in pages]
    if (missing or finish == 'MAX_TOKENS') and count > 1:
        print(f"  {label}: pages {first}-{last} incomplete (finish={finish}, missing={missing}); retrying in halves.")
        mid = (first + last) // 2
        return {
            **transcribe_range(transcriber, engine, pdf_bytes, first, mid, page_count, raw_log, cache_dir),
            **transcribe_range(transcriber, engine, pdf_bytes, mid + 1, last, page_count, raw_log, cache_dir),
        }
    if missing or finish not in ('STOP', 'UNKNOWN'):
        print(f"  WARNING: {label} page {first} finished with {finish}; output may be incomplete.")
    else:
        for page_num, markdown in pages.items():
            page_cache_path(cache_dir, engine, page_num).write_text(markdown, encoding='utf-8')
    return pages


def gemini_api_transcriber(api_key: str):
    """Transcriber using the Gemini API, with model fallback (see GeminiModelChain)."""
    # Retries and model fallback are handled by GeminiModelChain, so disable SDK-level retries
    client = genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(
            timeout=GEMINI_TIMEOUT_MS, retry_options=genai_types.HttpRetryOptions(attempts=1)
        ),
    )
    chain = GeminiModelChain(client, GEMINI_MODELS)

    def transcribe(chunk: bytes, prompt: str) -> tuple[str, str, str]:
        response = chain.generate(
            contents=[genai_types.Part.from_bytes(data=chunk, mime_type='application/pdf'), prompt],
            config=GEMINI_CONFIG,
        )
        candidate = response.candidates[0] if response.candidates else None
        finish = candidate.finish_reason.name if candidate and candidate.finish_reason else 'UNKNOWN'
        return response.text or '', finish, response.model_version
    return transcribe


def run_cli(cmd: list[str], chunk: bytes) -> str:
    """Run an agent CLI in a temp dir containing chunk.pdf; return stdout."""
    if not shutil.which(cmd[0]):
        raise RuntimeError(f"'{cmd[0]}' CLI not found on PATH. Install it and log in first.")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / 'chunk.pdf').write_bytes(chunk)
        try:
            result = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=CLI_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{cmd[0]} timed out after {CLI_TIMEOUT_S}s")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-500:]
        raise RuntimeError(f"{cmd[0]} exited with code {result.returncode}: {detail}")
    return result.stdout


def claude_code_transcriber():
    """Transcriber using Claude Code headless mode (`claude -p`); runs on the logged-in Claude subscription."""
    def transcribe(chunk: bytes, prompt: str) -> tuple[str, str, str]:
        out = run_cli(['claude', '-p', prompt + CLI_PROMPT_SUFFIX, '--model', CLAUDE_CODE_MODEL,
                       '--tools', 'Read', '--allowedTools', 'Read', '--output-format', 'json'], chunk)
        data = json.loads(out)
        if data.get('is_error'):
            raise RuntimeError(f"Claude Code error: {str(data.get('result'))[:300]}")
        finish = 'MAX_TOKENS' if data.get('stop_reason') == 'max_tokens' else 'STOP'
        model = ','.join((data.get('modelUsage') or {}).keys()) or CLAUDE_CODE_MODEL
        return data.get('result') or '', finish, model
    return transcribe


def antigravity_transcriber():
    """Transcriber using Antigravity CLI print mode (`agy -p`); runs on the logged-in Google AI subscription."""
    def transcribe(chunk: bytes, prompt: str) -> tuple[str, str, str]:
        out = run_cli(['agy', '-p', prompt + CLI_PROMPT_SUFFIX, '--model', ANTIGRAVITY_MODEL, '--add-dir', '.',
                       '--output-format', 'json', '--print-timeout', f"{CLI_TIMEOUT_S}s"], chunk)
        data = json.loads(out)
        if data.get('status') != 'SUCCESS':
            raise RuntimeError(f"Antigravity status {data.get('status')}: {str(data.get('response'))[:300]}")
        return data.get('response') or '', 'STOP', ANTIGRAVITY_MODEL
    return transcribe


def page_chunks(page_nums: list[int], size: int) -> list[tuple[int, int]]:
    """Group sorted page numbers into contiguous (first, last) ranges of at most `size` pages."""
    chunks: list[tuple[int, int]] = []
    for page in page_nums:
        if chunks and page == chunks[-1][1] + 1 and page - chunks[-1][0] < size:
            chunks[-1] = (chunks[-1][0], page)
        else:
            chunks.append((page, page))
    return chunks


def embedded_image_ppi(pdf_path: Path) -> dict[int, int]:
    """Map page number -> highest resolution (ppi) of any raster image embedded on that page."""
    result = subprocess.run(['pdfimages', '-list', str(pdf_path)], capture_output=True, text=True)
    ppi_by_page: dict[int, int] = {}
    for line in result.stdout.splitlines()[2:]:
        cols = line.split()
        try:
            page, x_ppi, y_ppi = int(cols[0]), int(cols[12]), int(cols[13])
        except (IndexError, ValueError):
            continue
        ppi_by_page[page] = max(ppi_by_page.get(page, 0), x_ppi, y_ppi)
    return ppi_by_page


def embedded_image_rects(pdf_path: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    """
    Map page number -> placement rects (x0, y0, x1, y1) of embedded raster images, normalized to
    0-1000 like Gemini boxes, via pdftohtml -xml (poppler).
    """
    rects: dict[int, list[tuple[float, float, float, float]]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(['pdftohtml', '-xml', '-q', '-zoom', '1', str(pdf_path.resolve()), str(Path(tmp) / 'doc')],
                       capture_output=True)
        xml_path = Path(tmp) / 'doc.xml'
        if not xml_path.exists():
            return rects
        xml = xml_path.read_text(encoding='utf-8', errors='replace')
    page = page_w = page_h = None
    for m in re.finditer(r'<page number="(\d+)"[^>]*height="([\d.]+)" width="([\d.]+)"'
                         r'|<image top="(-?[\d.]+)" left="(-?[\d.]+)" width="([\d.]+)" height="([\d.]+)"', xml):
        if m.group(1):
            page, page_h, page_w = int(m.group(1)), float(m.group(2)), float(m.group(3))
        elif page:
            top, left, w, h = (float(v) for v in m.groups()[3:])
            x0, y0 = max(0.0, left / page_w * 1000), max(0.0, top / page_h * 1000)
            x1, y1 = min(1000.0, (left + w) / page_w * 1000), min(1000.0, (top + h) / page_h * 1000)
            if x1 > x0 and y1 > y0:
                rects.setdefault(page, []).append((x0, y0, x1, y1))
    return rects


def snap_to_embedded_images(box: tuple[int, int, int, int],
                            image_rects: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    """
    If a figure box (x0, y0, x1, y1) is essentially one or more embedded raster images, return the exact
    union of those images' rects; model boxes are approximate, the PDF's image placement is not.
    Returns None for vector figures, or figures where images are only a small part (e.g. an icon).
    """
    bx0, by0, bx1, by1 = box
    hits = []
    for rx0, ry0, rx1, ry1 in image_rects:
        inter = max(0, min(bx1, rx1) - max(bx0, rx0)) * max(0, min(by1, ry1) - max(by0, ry0))
        if inter / ((rx1 - rx0) * (ry1 - ry0)) >= 0.5:  # image lies mostly inside the box
            hits.append((rx0, ry0, rx1, ry1))
    if not hits:
        return None
    union = (min(h[0] for h in hits), min(h[1] for h in hits), max(h[2] for h in hits), max(h[3] for h in hits))
    box_area = max(1, (bx1 - bx0) * (by1 - by0))
    if (union[2] - union[0]) * (union[3] - union[1]) / box_area < 0.5:
        return None
    return union


def render_page(pdf_path: Path, page_num: int, dpi: int, out_dir: Path) -> Image.Image:
    """Render one PDF page to an image at the given DPI via pdftoppm (poppler)."""
    prefix = out_dir / f"page{page_num}"
    subprocess.run(
        ['pdftoppm', '-png', '-r', str(dpi), '-f', str(page_num), '-l', str(page_num),
         '-singlefile', str(pdf_path), str(prefix)],
        capture_output=True, check=True
    )
    return Image.open(prefix.with_suffix('.png'))


def fit_to_content(gray: Image.Image, box: tuple[int, int, int, int], limits: tuple[int, int, int, int],
                   max_grow: int, pad: tuple[int, int], line_px: int) -> tuple[int, int, int, int]:
    """
    Adjust a model-predicted figure box (left, top, right, bottom pixels) so it neither clips the figure
    nor takes in its neighbors. Each edge first grows outward while it still cuts through ink (at most
    max_grow px), then adds up to pad (x, y) px of blank margin, stopping halfway to the next ink so
    nearby text or figures stay out. Edges never cross limits (the page bounds or other figures' boxes).

    Light marks (watermarks, faint tints) don't count as ink, and neither do thin lines (at most
    line_px wide) crossing the edge near its ends: those are frame or table borders that run past the
    figure, and following them would drag the edge through captions and body text. A table rule
    running along an edge (spanning it and continuing past the box) is a boundary: edges don't grow
    across it, and one just inside the box is shaved off.
    """
    step = 2
    width, height = gray.size
    edges = list(box)

    def strip_at(edge: int, pos: int, start: int | None = None, end: int | None = None) -> bytes:
        """The step-px strip just outside position pos of an edge, spanning start..end along the edge
        (default: the box's extent), collapsed to one row: 0xff where any pixel across it is dark."""
        left, top, right, bottom = edges
        if edge % 2 == 0:
            start, end = max(0, top if start is None else start), min(height, bottom if end is None else end)
            strip = gray.crop((pos - step if edge == 0 else pos, start, pos if edge == 0 else pos + step, end))
            strip = strip.transpose(Image.Transpose.ROTATE_90)
        else:
            start, end = max(0, left if start is None else start), min(width, right if end is None else end)
            strip = gray.crop((start, pos - step if edge == 1 else pos, end, pos if edge == 1 else pos + step))
        if strip.width == 0 or strip.height == 0:
            return b''
        dark = strip.point(lambda v: 255 if v < 160 else 0).reduce((1, strip.height)).point(lambda v: 255 if v else 0)
        return dark.tobytes()

    def has_ink(edge: int, pos: int) -> bool:
        row = strip_at(edge, pos)
        end_zone = pad[1 - edge % 2]
        return any(m.end() - m.start() > line_px or end_zone <= m.start() < len(row) - end_zone
                   for m in re.finditer(rb'\xff+', row))

    def is_table_rule(edge: int, pos: int) -> bool:
        """True if a line runs along the whole edge at pos and continues past at least one end of it."""
        row = strip_at(edge, pos)
        if not row or row.count(0xff) < 0.95 * len(row):
            return False
        lo, hi = (edges[1], edges[3]) if edge % 2 == 0 else (edges[0], edges[2])
        before, after = strip_at(edge, pos, lo - 4 * step, lo), strip_at(edge, pos, hi, hi + 4 * step)
        return any(len(s) == 4 * step and 0 not in s for s in (before, after))

    for edge in (1, 3, 0, 2):  # top, bottom, left, right
        out = -1 if edge < 2 else 1
        margin = pad[edge % 2]

        def can_move(pos):
            return pos - step >= limits[edge] if out < 0 else pos + step <= limits[edge]

        # Shave off a table rule just inside the edge, if there's blank space inside it
        pos = edges[edge]
        max_shave = min(max_grow, (edges[edge | 2] - edges[edge & 1]) // 2)
        while abs(pos - edges[edge]) < max_shave and not has_ink(edge, pos - out * step):
            pos -= out * step
        rule_end = pos
        while abs(rule_end - pos) <= line_px and is_table_rule(edge, rule_end - out * step):
            rule_end -= out * step
        if rule_end != pos and not any(has_ink(edge, rule_end - out * step * k) for k in range(1, line_px // step + 2)):
            edges[edge] = rule_end
            continue

        pos, grown = edges[edge], 0
        while grown < max_grow and can_move(pos) and has_ink(edge, pos) and not is_table_rule(edge, pos):
            pos, grown = pos + out * step, grown + step
        gap = 0
        while gap < 2 * margin and can_move(pos + out * gap) and not has_ink(edge, pos + out * gap):
            gap += step
        hit_ink = gap < 2 * margin and can_move(pos + out * gap)
        edges[edge] = pos + out * (gap // 2 if hit_ink else min(gap, margin))
    return tuple(edges)


def figure_limits(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]],
                  size: tuple[int, int]) -> tuple[int, int, int, int]:
    """How far each edge of box (left, top, right, bottom) may move: up to the page bounds, or the
    nearest edge of another figure that lies beside it (overlapping in the other axis)."""
    left, top, right, bottom = box
    limits = [0, 0, size[0], size[1]]
    for ol, ot, orr, ob in others:
        if ot < bottom and ob > top:  # beside horizontally
            if orr <= left:
                limits[0] = max(limits[0], orr)
            elif ol >= right:
                limits[2] = min(limits[2], ol)
        if ol < right and orr > left:  # above/below
            if ob <= top:
                limits[1] = max(limits[1], ob)
            elif ot >= bottom:
                limits[3] = min(limits[3], ot)
    return tuple(limits)


def crop_figures(markdown: str, page_num: int, pdf_path: Path, images_dir: Path, pdf_base_sanitized: str,
                 render_dir: Path, dpi: int, image_rects: list[tuple[float, float, float, float]]) -> tuple[str, list[str]]:
    """
    Replace ![desc](box:...) placeholders with figures cropped from a rendered page image.
    Boxes over embedded raster images snap to the images' exact placement; other (vector) figures
    use the model's box, fitted to the ink around it (see fit_to_content).
    """
    if not FIGURE_RE.search(markdown):
        return markdown, []
    page_img = render_page(pdf_path, page_num, dpi, render_dir)
    width, height = page_img.size
    gray = page_img.convert('L')
    # Model boxes are typically off by 1-2% of the page; cap growth so connected lines (frames,
    # table borders) can't drag the crop across the page
    max_grow = max(width, height) * 25 // 1000
    pad = (FIGURE_BOX_PADDING * width // 1000, FIGURE_BOX_PADDING * height // 1000)
    line_px = max(2, round(dpi / 72 * 1.5))  # 1.5 pt: rules and frame lines are thinner than this

    def normalized_box(match) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) in 0-1000 units from a (ymin,xmin,ymax,xmax) placeholder."""
        y0, x0, y1, x1 = (int(v) for v in match.groups()[1:])
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    def pixel_box(norm: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = norm
        return x0 * width // 1000, y0 * height // 1000, x1 * width // 1000, y1 * height // 1000

    boxes = [pixel_box(normalized_box(m)) for m in FIGURE_RE.finditer(markdown)]
    saved = []

    def replace(match):
        alt = match.group(1).strip()
        norm = normalized_box(match)
        box = pixel_box(norm)
        snapped = snap_to_embedded_images(norm, image_rects)
        if snapped:
            sx0, sy0, sx1, sy1 = snapped
            crop = page_img.crop((int(sx0 * width / 1000), int(sy0 * height / 1000),
                                  math.ceil(sx1 * width / 1000), math.ceil(sy1 * height / 1000)))
        else:
            others = [b for b in boxes if b != box]
            crop = page_img.crop(fit_to_content(gray, box, figure_limits(box, others, (width, height)),
                                                max_grow, pad, line_px))
        if crop.width < 10 or crop.height < 10 or is_blank_pil(crop):
            print(f"  Page {page_num}: skipped empty figure box '{alt}'.")
            return f"*[Figure: {alt}]*" if alt else ''
        name = f"{pdf_base_sanitized}_p{page_num}_fig{len(saved) + 1}.png"
        crop.save(images_dir / name)
        saved.append(name)
        return f"![{alt}](images/{name})"

    return FIGURE_RE.sub(replace, markdown), saved


def ocr_with_llm(engine: str, pdf_path: Path, api_key: str | None, pdf_output_dir: Path, images_dir: Path,
                 pdf_base_sanitized: str) -> tuple[list[str], list[str], list[str]]:
    """
    Transcribe a PDF with a multimodal LLM engine (Gemini API, Claude Code, or Antigravity), in parallel
    chunks of pages.

    The model reads the PDF natively (text layer + page images) and marks each figure with a bounding
    box; figures are cropped from a page render (snapped to embedded images where present), so vector
    graphics (plots, schematics) are captured, not just embedded raster images.

    Transcribed pages are cached per PDF and engine, so only pages that failed are requested on a re-run.
    A chunk that fails leaves warning placeholders for its pages instead of failing the document.

    Returns:
        (list of per-page markdown, list of saved image filenames, list of warnings)
    """
    label = ENGINES[engine]['label']
    page_count = get_page_count(pdf_path)
    if page_count == 0:
        raise RuntimeError("Could not determine page count.")
    pdf_bytes = pdf_path.read_bytes()

    cache_dir = PAGE_CACHE_FOLDER / hashlib.sha256(pdf_bytes).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    pages: dict[int, str] = {}
    for page_num in range(1, page_count + 1):
        cached = page_cache_path(cache_dir, engine, page_num)
        if cached.exists():
            pages[page_num] = cached.read_text(encoding='utf-8')
    if pages:
        print(f"  {label}: {len(pages)}/{page_count} pages loaded from cache ({cache_dir}).")

    if engine == 'gemini':
        transcriber, pages_per_request, concurrency = gemini_api_transcriber(api_key), GEMINI_PAGES_PER_REQUEST, GEMINI_CONCURRENCY
    elif engine == 'claude-code':
        transcriber, pages_per_request, concurrency = claude_code_transcriber(), CLI_PAGES_PER_REQUEST, CLI_CONCURRENCY
    else:
        transcriber, pages_per_request, concurrency = antigravity_transcriber(), CLI_PAGES_PER_REQUEST, CLI_CONCURRENCY

    ranges = page_chunks([p for p in range(1, page_count + 1) if p not in pages], pages_per_request)
    print(f"  {label}: {page_count - len(pages)} pages in {len(ranges)} request(s)...")

    raw_log: list[dict] = []
    failures: list[str] = []
    executor = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futures = {executor.submit(transcribe_range, transcriber, engine, pdf_bytes, s, e, page_count, raw_log, cache_dir): (s, e)
                   for s, e in ranges}
        for future, (first, last) in futures.items():
            try:
                pages.update(future.result())
            except (genai_errors.APIError, httpx.HTTPError, RuntimeError, json.JSONDecodeError) as e:
                reason = f"{e.code} {e.status}" if isinstance(e, genai_errors.APIError) else str(e)
                print(f"  ERROR: {label} failed for pages {first}-{last}: {reason}")
                failures.append(f"pages {first}-{last} ({reason})")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if not pages:
        raise RuntimeError(f"{label} could not transcribe any pages: {'; '.join(failures)}")

    if raw_log:
        raw_path = pdf_output_dir / "llm_response.json"
        raw_path.write_text(json.dumps(sorted(raw_log, key=lambda r: r['pages']), indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"  Raw responses saved to {raw_path}")
        print(f"  Model(s) used: {', '.join(sorted({r['model'] for r in raw_log if r['model']}))}")

    warnings = []
    missing_pages = [p for p in range(1, page_count + 1) if p not in pages]
    if missing_pages:
        warnings.append(f"{len(missing_pages)} page(s) could not be transcribed ({'; '.join(failures) or 'incomplete output'}). "
                        f"Convert the same PDF again later to retry just those pages; finished pages are cached.")

    ppi_by_page = embedded_image_ppi(pdf_path)
    image_rects_by_page = embedded_image_rects(pdf_path)
    page_markdowns, image_filenames = [], []
    with tempfile.TemporaryDirectory() as render_dir:
        for page_num in range(1, page_count + 1):
            markdown = pages.get(page_num)
            if markdown is None:
                print(f"  WARNING: Page {page_num} missing from {label} output.")
                markdown = (f"> [!warning] Page {page_num} could not be transcribed.\n"
                            f"> Convert the PDF again to retry this page.")
            dpi = min(max(FIGURE_DPI, ppi_by_page.get(page_num, 0)), FIGURE_MAX_DPI)
            markdown, figures = crop_figures(markdown, page_num, pdf_path, images_dir, pdf_base_sanitized,
                                             Path(render_dir), dpi, image_rects_by_page.get(page_num, []))
            image_filenames.extend(figures)
            page_markdowns.append(markdown)
    print(f"  Cropped {len(image_filenames)} figures.")
    return page_markdowns, image_filenames, warnings


# --- Mistral Engine ---

def ocr_with_mistral(pdf_path: Path, api_key: str, pdf_output_dir: Path, images_dir: Path,
                     pdf_base_sanitized: str) -> tuple[list[str], list[str], list[str]]:
    """
    OCR a PDF using Mistral OCR and pdfimages (poppler).

    Strategy:
    - pdfimages (poppler) is the primary image source (full resolution, filtered, deduplicated).
    - Mistral OCR determines WHERE each image appears inline in the markdown.
    - pdfimages images are matched positionally to Mistral image placeholders per page.
    - Extra pdfimages images (not matched by Mistral) are appended near the page boundary.
    - If Mistral finds more images than pdfimages, the Mistral version is used as a fallback
      and a warning is logged (indicating a potential extraction gap).

    Returns:
        (list of per-page markdown, list of saved image filenames, list of warnings)
    """
    client = Mistral(api_key=api_key)
    uploaded_file = None

    try:
        print(f"  Uploading {pdf_path.name} to Mistral...")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        uploaded_file = client.files.upload(
            file={"file_name": pdf_path.name, "content": pdf_bytes}, purpose="ocr"
        )

        print(f"  File uploaded (ID: {uploaded_file.id}). Getting signed URL...")
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=60)

        print(f"  Calling OCR API...")
        ocr_response: OCRResponse = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=True
        )
        print(f"  OCR processing complete for {pdf_path.name}.")

        # Save raw OCR response
        ocr_json_path = pdf_output_dir / "ocr_response.json"
        try:
            with open(ocr_json_path, "w", encoding="utf-8") as json_file:
                if hasattr(ocr_response, 'model_dump'):
                    json.dump(ocr_response.model_dump(), json_file, indent=4, ensure_ascii=False)
                else:
                    json.dump(ocr_response.dict(), json_file, indent=4, ensure_ascii=False)
            print(f"  Raw OCR response saved to {ocr_json_path}")
        except Exception as json_err:
            print(f"  Warning: Could not save raw OCR JSON: {json_err}")

        # --- Extract images via pdfimages/poppler (primary, full-resolution) ---
        print(f"  Extracting images via pdfimages...")
        pdfimages_by_page = extract_images_pdfimages(pdf_path, images_dir, pdf_base_sanitized)

        # Start the filenames list from all pdfimages-extracted images
        extracted_image_filenames = [f for imgs in pdfimages_by_page.values() for f in imgs]

        # --- Process OCR response: build markdown with merged image references ---
        updated_markdown_pages = []
        print(f"  Merging OCR output with pdfimages results...")

        for page_index, page in enumerate(ocr_response.pages):
            page_num = page_index + 1
            current_page_markdown = page.markdown
            page_image_mapping = {}

            page_imgs = list(pdfimages_by_page.get(page_num, []))
            page_imgs_iter = iter(page_imgs)
            # Fallback naming continues after pdfimages images on this page
            fallback_n = len(page_imgs)

            for image_obj in page.images:
                base64_str = image_obj.image_base64
                if not base64_str:
                    continue
                if base64_str.startswith("data:"):
                    try:
                        base64_str = base64_str.split(",", 1)[1]
                    except IndexError:
                        continue
                try:
                    image_bytes = base64.b64decode(base64_str)
                except Exception as decode_err:
                    print(f"  Warning: Base64 decode error for {image_obj.id} on page {page_num}: {decode_err}")
                    continue

                next_img = next(page_imgs_iter, None)
                if next_img:
                    # Use the high-res pdfimages version in place of the Mistral image
                    page_image_mapping[image_obj.id] = next_img
                else:
                    # No pdfimages match — use Mistral image as low-quality fallback
                    print(f"  WARNING: Page {page_num}: Mistral found '{image_obj.id}' but pdfimages had no "
                          f"match. Using Mistral version as fallback (lower quality).")
                    fallback_n += 1
                    orig_ext = Path(image_obj.id).suffix or ".jpeg"
                    fallback_name = f"{pdf_base_sanitized}_p{page_num}_img{fallback_n}{orig_ext}"
                    try:
                        (images_dir / fallback_name).write_bytes(image_bytes)
                        page_image_mapping[image_obj.id] = fallback_name
                        extracted_image_filenames.append(fallback_name)
                    except IOError as io_err:
                        print(f"  Warning: Could not write fallback image {fallback_name}: {io_err}")

            updated_page_markdown = replace_images_in_markdown(current_page_markdown, page_image_mapping)

            # Append any remaining pdfimages images not matched to a Mistral placement
            extra_imgs = list(page_imgs_iter)
            if extra_imgs:
                img_refs = "\n".join(f"![](images/{name})" for name in extra_imgs)
                supplement = (
                    f"\n\n> [!note] Additional Extracted Images — Page {page_num}\n"
                    f"> The following images were found on this page but were not placed inline by OCR.\n\n"
                    f"{img_refs}"
                )
                updated_page_markdown += supplement
                print(f"  Page {page_num}: appended {len(extra_imgs)} additional image(s) not placed by OCR.")

            updated_markdown_pages.append(updated_page_markdown)

        # Clean up Mistral file
        try:
            client.files.delete(file_id=uploaded_file.id)
            print(f"  Deleted temporary file {uploaded_file.id} from Mistral.")
        except Exception as delete_err:
            print(f"  Warning: Could not delete file {uploaded_file.id} from Mistral: {delete_err}")

        return updated_markdown_pages, extracted_image_filenames, []

    except Exception as e:
        error_str = str(e)
        json_index = error_str.find('{')
        if json_index != -1:
            try:
                error_json = json.loads(error_str[json_index:])
                error_msg = error_json.get("message", error_str)
            except Exception:
                error_msg = error_str
        else:
            error_msg = error_str
        print(f"  Error processing {pdf_path.name}: {error_msg}")
        if uploaded_file:
            try:
                client.files.delete(file_id=uploaded_file.id)
            except Exception:
                pass
        raise Exception(error_msg)


# --- Core Processing Logic ---

def process_pdf(pdf_path: Path, engine: str, api_key: str, session_output_dir: Path,
                page_separator: str | None = PAGE_SEPARATOR_DEFAULT) -> tuple[str, str, list[str], Path, Path, list[str]]:
    """
    Processes a single PDF file with the given OCR engine and saves the markdown + images.

    Returns:
        (pdf_base_name, final_markdown, list_of_image_filenames, markdown_path, images_dir, warnings)
    """
    pdf_base = pdf_path.stem
    base_sanitized_original = secure_filename(pdf_base)
    pdf_base_sanitized = base_sanitized_original
    print(f"Processing {pdf_path.name} with {ENGINES[engine]['label']}...")

    pdf_output_dir = session_output_dir / pdf_base_sanitized
    counter = 1
    while pdf_output_dir.exists():
        pdf_base_sanitized = f"{base_sanitized_original}_{counter}"
        pdf_output_dir = session_output_dir / pdf_base_sanitized
        counter += 1

    pdf_output_dir.mkdir(exist_ok=True)
    images_dir = pdf_output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    try:
        if engine == 'mistral':
            page_markdowns, image_filenames, warnings = ocr_with_mistral(pdf_path, api_key, pdf_output_dir, images_dir, pdf_base_sanitized)
        else:
            page_markdowns, image_filenames, warnings = ocr_with_llm(engine, pdf_path, api_key, pdf_output_dir, images_dir, pdf_base_sanitized)
    except genai_errors.APIError as e:
        print(f"  Error processing {pdf_path.name}: {e}")
        raise Exception(f"Gemini API error {e.code}: {e.message}") from e

    parts = []
    for i, page_markdown in enumerate(page_markdowns):
        parts.append(page_markdown)
        if i < len(page_markdowns) - 1:
            next_page_num = i + 2
            if page_separator:
                parts.append(f"\n\n{page_separator}\n*Page {next_page_num}*\n\n")
            else:
                parts.append(f"\n\n*Page {next_page_num}*\n\n")
    final_markdown_content = "".join(parts)
    output_markdown_path = pdf_output_dir / f"{pdf_base_sanitized}_output.md"

    try:
        with open(output_markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write(final_markdown_content)
        print(f"  Markdown generated successfully at {output_markdown_path}")
    except IOError as io_err:
        raise Exception(f"Failed to write final markdown file: {io_err}") from io_err

    return pdf_base_sanitized, final_markdown_content, image_filenames, output_markdown_path, images_dir, warnings


def create_zip_archive(source_dir: Path, output_zip_path: Path):
    print(f"  Creating ZIP archive: {output_zip_path} from {source_dir}")
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for entry in source_dir.rglob('*'):
                arcname = entry.relative_to(source_dir)
                zipf.write(entry, arcname)
        print(f"  Successfully created ZIP: {output_zip_path}")
    except Exception as e:
        print(f"  Error creating ZIP file {output_zip_path}: {e}")
        raise


# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html', default_page_separator=PAGE_SEPARATOR_DEFAULT,
                           engines=ENGINES, default_engine=DEFAULT_ENGINE)

@app.route('/check-api-key', methods=['GET'])
def check_api_key():
    """Report, per engine, whether it needs an API key and whether one is configured in the environment."""
    return jsonify({name: {"needs_key": bool(cfg['key_env']),
                           "has_key": bool(cfg['key_env'] and os.getenv(cfg['key_env']))}
                    for name, cfg in ENGINES.items()})

@app.route('/save-api-key', methods=['POST'])
def save_api_key():
    """Save the provided API key to the local .env file (gitignored)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    key = data.get('api_key', '').strip()
    if not key:
        return jsonify({"error": "No API key provided"}), 400
    engine = data.get('engine', DEFAULT_ENGINE)
    if engine not in ENGINES or not ENGINES[engine]['key_env']:
        return jsonify({"error": f"Engine {engine} does not use an API key"}), 400

    env_file = Path('.env')
    if not env_file.exists():
        env_file.write_text('')

    set_key(str(env_file), ENGINES[engine]['key_env'], key)
    load_dotenv(override=True)
    print(f"{ENGINES[engine]['label']} key saved to .env and reloaded.")
    return jsonify({"success": True})

@app.route('/process', methods=['POST'])
def handle_process():
    if 'pdf_files' not in request.files:
        return jsonify({"error": "No PDF files part in the request"}), 400

    files = request.files.getlist('pdf_files')

    engine = request.form.get('engine', DEFAULT_ENGINE)
    if engine not in ENGINES:
        return jsonify({"error": f"Unknown engine: {engine}"}), 400
    engine_cfg = ENGINES[engine]

    api_key = os.getenv(engine_cfg['key_env']) if engine_cfg['key_env'] else None
    if not engine_cfg['key_env']:
        print(f"Using {engine_cfg['label']} via its logged-in CLI.")
    elif api_key:
        print(f"Using {engine_cfg['label']} key from environment (first 4 chars): {api_key[:4]}...")
    else:
        api_key = request.form.get('api_key')
        if api_key:
            print(f"Using {engine_cfg['label']} key from web form (first 4 chars): {api_key[:4]}...")
        else:
            return jsonify({"error": f"{engine_cfg['label']} key is required. Set {engine_cfg['key_env']} in .env "
                                     f"or provide it in the form (get one at {engine_cfg['key_url']})."}), 400

    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No selected PDF files"}), 400

    valid_files, invalid_files = [], []
    for f in files:
        if f and allowed_file(f.filename):
            valid_files.append(f)
        elif f and f.filename != '':
            invalid_files.append(f.filename)

    if not valid_files:
        error_msg = "No valid PDF files found."
        if invalid_files:
            error_msg += f" Invalid files skipped: {', '.join(invalid_files)}"
        return jsonify({"error": error_msg}), 400

    session_id = str(uuid4())
    session_upload_dir = UPLOAD_FOLDER / session_id
    session_output_dir = OUTPUT_FOLDER / session_id
    session_upload_dir.mkdir(parents=True, exist_ok=True)
    session_output_dir.mkdir(parents=True, exist_ok=True)

    processed_files_results = []
    processing_errors = []
    if invalid_files:
        processing_errors.append(f"Skipped non-PDF files: {', '.join(invalid_files)}")

    page_separator = request.form.get('page_separator')
    if page_separator is None:
        page_separator = PAGE_SEPARATOR_DEFAULT

    for file in valid_files:
        original_filename = file.filename
        filename_sanitized = secure_filename(original_filename)
        temp_pdf_path = session_upload_dir / filename_sanitized

        try:
            print(f"Saving uploaded file temporarily to: {temp_pdf_path}")
            file.save(temp_pdf_path)

            processed_pdf_base, markdown_content, image_filenames, md_path, img_dir, warnings = process_pdf(
                temp_pdf_path, engine, api_key, session_output_dir, page_separator
            )

            zip_filename = f"{processed_pdf_base}_output.zip"
            zip_output_path = session_output_dir / zip_filename
            individual_output_dir = session_output_dir / processed_pdf_base
            create_zip_archive(individual_output_dir, zip_output_path)
            processing_errors.extend(f"{original_filename}: {w}" for w in warnings)

            download_url = url_for('download_file', session_id=session_id, filename=zip_filename, _external=True)

            processed_files_results.append({
                "original_filename": original_filename,
                "zip_filename": zip_filename,
                "download_url": download_url,
                "preview": {
                    "markdown": markdown_content,
                    "images": image_filenames,
                    "pdf_base": processed_pdf_base
                }
            })
            print(f"Successfully processed and zipped: {original_filename}")

        except Exception as e:
            print(f"ERROR: Failed processing {original_filename}: {e}")
            processing_errors.append(f"{original_filename}: Processing Error - {e}")
        finally:
            if temp_pdf_path.exists():
                try:
                    temp_pdf_path.unlink()
                except OSError as unlink_err:
                    print(f"Warning: Could not delete temp file {temp_pdf_path}: {unlink_err}")

    try:
        shutil.rmtree(session_upload_dir)
        print(f"Cleaned up session upload directory: {session_upload_dir}")
    except OSError as rmtree_err:
        print(f"Warning: Could not delete session upload directory {session_upload_dir}: {rmtree_err}")

    if not processed_files_results and processing_errors:
        return jsonify({"error": "All PDF processing attempts failed.", "details": processing_errors}), 500
    elif not processed_files_results:
        return jsonify({"error": "No files were processed successfully."}), 500
    else:
        return jsonify({
            "success": True,
            "session_id": session_id,
            "results": processed_files_results,
            "errors": processing_errors
        }), 200


@app.route('/view_image/<session_id>/<pdf_base>/<filename>')
def view_image(session_id, pdf_base, filename):
    """Serves an extracted image file for inline display."""
    safe_session_id = secure_filename(session_id)
    safe_pdf_base = secure_filename(pdf_base)
    safe_filename = secure_filename(filename)

    directory = OUTPUT_FOLDER / safe_session_id / safe_pdf_base / "images"
    file_path = directory / safe_filename

    if not str(file_path.resolve()).startswith(str(directory.resolve())):
        return "Invalid path", 400
    if not file_path.is_file():
        return "Image not found", 404

    print(f"Serving image: {file_path}")
    return send_from_directory(directory, safe_filename)


@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    """Serves the generated ZIP file for download."""
    safe_session_id = secure_filename(session_id)
    safe_filename = secure_filename(filename)
    directory = OUTPUT_FOLDER / safe_session_id
    file_path = directory / safe_filename

    if not str(file_path.resolve()).startswith(str(directory.resolve())):
        return "Invalid path", 400
    if not file_path.is_file():
        return "File not found", 404

    print(f"Serving ZIP for download: {file_path}")
    return send_from_directory(directory, safe_filename, as_attachment=True)


if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5200))
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']

    app.run(host=host, port=port, debug=debug_mode)
