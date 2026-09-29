# PDF to Markdown for Obsidian

Converts PDFs (datasheets, app notes, drawings) to Markdown with figures cropped as images and linked with standard `![](images/name.png)` links. A local Flask web app does the conversion. An older Mistral-only [Jupyter Notebook](#jupyter-notebook) is also included.

## Features

- **Multiple engines:** Claude Code or Antigravity (both run on your logged-in subscription), the Gemini API, or Mistral OCR.
- **Figures at source definition:** figures, including vector plots and schematics, are cropped from a page render at `max(FIGURE_DPI, highest embedded image ppi on the page)`. Boxes over embedded raster images snap to the image's exact placement in the PDF; other boxes are fitted to the surrounding ink so labels aren't clipped and neighboring text, figures, and table rules stay out.
- **Page cache:** transcribed pages are cached in `.cache/pages/` per engine, model, and effort, so re-running a PDF only requests pages that failed.
- **Batch upload, page separators with page numbers, ZIP download, and in-browser preview.**

## Self-hosted Local Web App

![alt text](doc/usage.gif)

### Prerequisites

Install [poppler](https://poppler.freedesktop.org/), which is used to render pages and crop figures:

```sh
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils
```

### Setup

```sh
pip install -r requirements.txt    # (I recommend creating a virtual environment to not clutter your OS)
python app.py
```
Then open your browser at `http://localhost:5200/`

Or use the provided `start.sh` (macOS) which handles venv activation, dependency install, and browser launch.

### OCR Engines

Pick the engine in the web UI. For the CLI engines you can also pick the model and effort; `CLI setting` leaves effort to the CLI's own configuration.

- **Claude Code** (default): runs `claude -p` on page chunks using your logged-in Claude subscription. Needs the `claude` CLI, logged in. With several accounts, each logged in to its own config dir (`~/.claude-<name>`, used via `CLAUDE_CONFIG_DIR`), pick the account in the UI.
- **Antigravity**: runs `agy -p` using your logged-in Google AI subscription (`agy models` lists the models). Needs the `agy` CLI, logged in.
- **Gemini API** (`GEMINI_API_KEY`, [AI Studio](https://aistudio.google.com/apikey)): the free tier allows about 20 requests/day per model and often returns 503s, so it falls back through a chain of models.
- **Mistral** (`MISTRAL_API_KEY`): Mistral OCR for text, `pdfimages` for embedded raster images.

The LLM engines all share one prompt. The model returns Markdown per page, starting each page with a `<<<PAGE n>>>` marker, and puts `![desc](box:ymin,xmin,ymax,xmax)` placeholders where figures go. Raw responses are saved to `llm_response.json`. To check that text came through completely, run `python tools/check_text_coverage.py <file.pdf> <output.md>`.

Settings (environment variables, optional):

| Variable | Default | Purpose |
| --- | --- | --- |
| `OCR_ENGINE` | `claude-code` | Engine selected by default in the UI |
| `CLAUDE_CODE_MODEL` / `ANTIGRAVITY_MODEL` | `opus` / newest `gemini-*-flash-high` | Model selected by default in the UI |
| `CLAUDE_CODE_EFFORT` / `ANTIGRAVITY_EFFORT` | unset (CLI setting) | Effort selected by default in the UI |
| `CLI_PAGES_PER_REQUEST` / `CLI_CONCURRENCY` | `5` / `3` | Pages per CLI run, and how many CLI runs happen in parallel |
| `GEMINI_MODEL` | `gemini-3.6-flash,...,gemini-3.1-flash-lite` | Comma-separated fallback chain |
| `GEMINI_PAGES_PER_REQUEST` / `GEMINI_CONCURRENCY` | `10` / `3` | Pages per request, and parallel requests |
| `FIGURE_DPI` | `300` | Minimum render DPI for cropped figures |

### Customizing Page Separators

The app inserts `---` between PDF pages by default. Set the `PAGE_SEPARATOR` environment
variable to change this text or leave it empty to merge pages without separators.
The web interface also lets you toggle and edit the separator before processing.

## Jupyter Notebook

### Installation

Ensure you have Python 3.9+ and poppler installed (see [Prerequisites](#prerequisites) above). Then install dependencies:

```sh
pip install mistralai jupyter python-dotenv Pillow
```

### Usage
#### 1. Set Up API Key

Before running the notebook, get your free API key from [Mistral's API Key Console](https://console.mistral.ai/api-keys). It's free.

Edit the `env.example` with your key, rename it to `.env` and you're good to go.

Or set it manually:

```sh
export MISTRAL_API_KEY='your_api_key_here'  # For Linux/macOS
set MISTRAL_API_KEY='your_api_key_here'    # For Windows
```

#### 2. Open the Notebook

```sh
jupyter notebook pdf-markdown-ocr.ipynb
```

Or open the [Notebook](pdf-markdown-ocr.ipynb) file directly in your IDE.

#### 3. Place PDFs in pdfs_to_process

Before first use, create a `pdfs_to_process` folder in the project directory and drop your PDFs in there.

#### 4. Run the Notebook

Go cell by cell and make sure everything runs as expected.

#### 5. Output Structure
Each processed PDF gets its own folder inside `ocr_output`, structured like this:

```
ocr_output/
  ├── MyDocument/
  │   ├── output.md            # Extracted markdown with wikilinks
  │   ├── ocr_response.json    # Raw OCR response (for reuse)
  │   ├── images/
  │   │   ├── MyDocument_img_1.jpeg
  │   │   ├── MyDocument_img_2.jpeg
pdfs-done/
  ├── MyDocument.pdf  # Moved here after OCR completion
```

#### 6. Move Output to Obsidian Vault

Move the generated `output.md` file into your **Obsidian vault** and also move the images to your attachments folder.

**Heads up!**: For now, Obsidian must be configured to support ![[image-name]] style links. If your setup is different, the script might not work as-is. Feel free to fork and tweak it.

### How It Works

1. The notebook scans `pdfs_to_process` for PDFs.
2. Each PDF is uploaded to Mistral AI for OCR processing.
3. The text is extracted and saved as markdown (`output.md`).
4. Images are extracted, saved in a subfolder, and referenced in the markdown using `![[image-name]]`.
5. The original PDF is moved to `pdfs-done` to avoid duplicate processing.
6. The full OCR response is saved as JSON for later use.
