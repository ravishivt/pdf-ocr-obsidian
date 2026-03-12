import os
import json
import base64
import hashlib
import shutil
import zipfile
import re
from pathlib import Path
from uuid import uuid4
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
from mistralai import Mistral, DocumentURLChunk
from mistralai.models import OCRResponse
import fitz  # PyMuPDF
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

def is_blank_image(img_bytes: bytes) -> bool:
    """Return True if the image contains only whitespace (near-uniform color, no real content)."""
    try:
        pix = fitz.Pixmap(img_bytes)
        if pix.alpha:
            pix = fitz.Pixmap(pix, 0)  # drop alpha channel
        if pix.n != 1:
            pix = fitz.Pixmap(fitz.csGRAY, pix)  # convert to grayscale
        samples = pix.samples
        if not samples:
            return True
        # If the pixel value range is tiny, the image is effectively uniform (blank)
        return (max(samples) - min(samples)) < 15
    except Exception:
        return False


def extract_images_pymupdf(pdf_path: Path, images_dir: Path, pdf_base_sanitized: str) -> dict:
    """
    Extract embedded images from a PDF using PyMuPDF.

    Filtering applied:
    - Skips blank/whitespace images (near-uniform pixel values)
    - Skips images whose content hash appears on REPEATED_PAGE_THRESHOLD or more distinct pages
      (catches repeated headers/footers/watermarks that appear throughout the document)
    - Deduplicates: identical image content is saved only on its first occurrence

    Naming: {pdf_base}_p{page}_img{n}.{ext}, where n is 1-based per page.

    Returns:
        dict mapping 1-based page number -> list of saved image filenames
    """
    doc = fitz.open(str(pdf_path))

    # Pass 1: collect raw image data and track which pages each hash appears on
    raw_by_page: dict[int, list[tuple]] = {}   # page_num -> [(img_bytes, ext, hash), ...]
    hash_page_count: dict[str, set] = {}        # hash -> set of page_nums it appears on

    for page_index in range(len(doc)):
        page_num = page_index + 1
        for img_info in doc.get_page_images(page_index, full=True):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception as e:
                print(f"  Warning: could not extract image xref {xref} on page {page_num}: {e}")
                continue

            img_bytes = base_image["image"]
            if is_blank_image(img_bytes):
                continue

            content_hash = hashlib.md5(img_bytes).hexdigest()
            hash_page_count.setdefault(content_hash, set()).add(page_num)
            raw_by_page.setdefault(page_num, []).append((img_bytes, base_image["ext"], content_hash))

    # Pass 2: save images — skip repeated elements and exact duplicates
    images_by_page: dict[int, list[str]] = {}
    seen_hashes: set[str] = set()

    for page_num in sorted(raw_by_page.keys()):
        local_idx = 1
        page_files = []
        for img_bytes, ext, content_hash in raw_by_page[page_num]:
            if len(hash_page_count[content_hash]) >= REPEATED_PAGE_THRESHOLD:
                continue  # Appears too many times — likely a header/footer/watermark
            if content_hash in seen_hashes:
                continue  # Exact duplicate already saved
            seen_hashes.add(content_hash)

            filename = f"{pdf_base_sanitized}_p{page_num}_img{local_idx}.{ext}"
            try:
                (images_dir / filename).write_bytes(img_bytes)
                page_files.append(filename)
                local_idx += 1
            except OSError as e:
                print(f"  Warning: could not write image {filename}: {e}")

        if page_files:
            images_by_page[page_num] = page_files

    doc.close()
    total = sum(len(v) for v in images_by_page.values())
    print(f"  PyMuPDF: saved {total} images across {len(images_by_page)} pages (after filtering/dedup).")
    return images_by_page


# --- Core Processing Logic ---

def process_pdf(pdf_path: Path, api_key: str, session_output_dir: Path, page_separator: str | None = PAGE_SEPARATOR_DEFAULT) -> tuple[str, str, list[str], Path, Path]:
    """
    Processes a single PDF file using Mistral OCR and PyMuPDF, saves results.

    Strategy:
    - PyMuPDF is the primary image source (full resolution, filtered, deduplicated).
    - Mistral OCR determines WHERE each image appears inline in the markdown.
    - Mistral images are matched positionally to PyMuPDF images per page.
    - Extra PyMuPDF images (not matched by Mistral) are appended near the page boundary.
    - If Mistral finds more images than PyMuPDF, the Mistral version is used as a fallback
      and a warning is logged (indicating a potential extraction gap).

    Returns:
        (pdf_base_name, final_markdown, list_of_image_filenames, markdown_path, images_dir)
    """
    pdf_base = pdf_path.stem
    base_sanitized_original = secure_filename(pdf_base)
    pdf_base_sanitized = base_sanitized_original
    print(f"Processing {pdf_path.name}...")

    pdf_output_dir = session_output_dir / pdf_base_sanitized
    counter = 1
    while pdf_output_dir.exists():
        pdf_base_sanitized = f"{base_sanitized_original}_{counter}"
        pdf_output_dir = session_output_dir / pdf_base_sanitized
        counter += 1

    pdf_output_dir.mkdir(exist_ok=True)
    images_dir = pdf_output_dir / "images"
    images_dir.mkdir(exist_ok=True)

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

        # --- Extract images via PyMuPDF (primary, full-resolution) ---
        print(f"  Extracting images via PyMuPDF...")
        pymupdf_by_page = extract_images_pymupdf(pdf_path, images_dir, pdf_base_sanitized)

        # Start the filenames list from all PyMuPDF-saved images
        extracted_image_filenames = [f for imgs in pymupdf_by_page.values() for f in imgs]

        # --- Process OCR response: build markdown with merged image references ---
        updated_markdown_pages = []
        print(f"  Merging OCR output with PyMuPDF images...")

        for page_index, page in enumerate(ocr_response.pages):
            page_num = page_index + 1
            current_page_markdown = page.markdown
            page_image_mapping = {}

            pymupdf_page_imgs = list(pymupdf_by_page.get(page_num, []))
            pymupdf_iter = iter(pymupdf_page_imgs)
            # Fallback naming continues after PyMuPDF images on this page
            fallback_n = len(pymupdf_page_imgs)

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

                next_pymupdf = next(pymupdf_iter, None)
                if next_pymupdf:
                    # Use the high-res PyMuPDF version in place of the Mistral image
                    page_image_mapping[image_obj.id] = next_pymupdf
                else:
                    # No PyMuPDF match — use Mistral image as low-quality fallback
                    print(f"  WARNING: Page {page_num}: Mistral found '{image_obj.id}' but PyMuPDF had no "
                          f"match. Using Mistral version as fallback (lower quality). "
                          f"Consider poppler pdfimages for more complete extraction.")
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

            # Append any remaining PyMuPDF images not matched to a Mistral placement
            extra_pymupdf = list(pymupdf_iter)
            if extra_pymupdf:
                img_refs = "\n".join(f"![](images/{name})" for name in extra_pymupdf)
                supplement = (
                    f"\n\n> [!note] Additional Extracted Images — Page {page_num}\n"
                    f"> The following images were found on this page but were not placed inline by OCR.\n\n"
                    f"{img_refs}"
                )
                updated_page_markdown += supplement
                print(f"  Page {page_num}: appended {len(extra_pymupdf)} additional image(s) not placed by OCR.")

            updated_markdown_pages.append(updated_page_markdown)

        parts = []
        for i, page_markdown in enumerate(updated_markdown_pages):
            parts.append(page_markdown)
            if i < len(updated_markdown_pages) - 1:
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

        # Clean up Mistral file
        try:
            client.files.delete(file_id=uploaded_file.id)
            print(f"  Deleted temporary file {uploaded_file.id} from Mistral.")
        except Exception as delete_err:
            print(f"  Warning: Could not delete file {uploaded_file.id} from Mistral: {delete_err}")

        return pdf_base_sanitized, final_markdown_content, extracted_image_filenames, output_markdown_path, images_dir

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
    return render_template('index.html', default_page_separator=PAGE_SEPARATOR_DEFAULT)

@app.route('/check-api-key', methods=['GET'])
def check_api_key():
    """Check if the API key is configured in the environment variables."""
    api_key = os.getenv("MISTRAL_API_KEY")
    return jsonify({"has_api_key": bool(api_key)})

@app.route('/save-api-key', methods=['POST'])
def save_api_key():
    """Save the provided API key to the local .env file (gitignored)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    key = data.get('api_key', '').strip()
    if not key:
        return jsonify({"error": "No API key provided"}), 400

    env_file = Path('.env')
    if not env_file.exists():
        env_file.write_text('')

    set_key(str(env_file), 'MISTRAL_API_KEY', key)
    load_dotenv(override=True)
    print("API key saved to .env and reloaded.")
    return jsonify({"success": True})

@app.route('/process', methods=['POST'])
def handle_process():
    if 'pdf_files' not in request.files:
        return jsonify({"error": "No PDF files part in the request"}), 400

    files = request.files.getlist('pdf_files')

    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key:
        print(f"Using API Key from environment (first 4 chars): {api_key[:4]}...")
    else:
        api_key = request.form.get('api_key')
        if api_key:
            print(f"Using API Key from web form (first 4 chars): {api_key[:4]}...")
        else:
            return jsonify({"error": "Mistral API Key is required. Set MISTRAL_API_KEY in .env or provide it in the form."}), 400

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

            processed_pdf_base, markdown_content, image_filenames, md_path, img_dir = process_pdf(
                temp_pdf_path, api_key, session_output_dir, page_separator
            )

            zip_filename = f"{processed_pdf_base}_output.zip"
            zip_output_path = session_output_dir / zip_filename
            individual_output_dir = session_output_dir / processed_pdf_base
            create_zip_archive(individual_output_dir, zip_output_path)

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
