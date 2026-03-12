import os
import json
import base64
import shutil
import zipfile
import re # Import regex module
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

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# --- Helper Functions ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def replace_images_in_markdown_with_wikilinks(markdown_str: str, image_mapping: dict) -> str:
    updated_markdown = markdown_str
    for original_id, new_name in image_mapping.items():
        updated_markdown = updated_markdown.replace(
            f"![{original_id}]({original_id})",
            f"![[{new_name}]]"
        )
    return updated_markdown

def extract_images_pymupdf(pdf_path: Path, images_dir: Path, pdf_base_sanitized: str) -> dict[int, list[str]]:
    """
    Extract all embedded images from the PDF using PyMuPDF.
    Returns dict mapping 1-based page number -> list of image filenames saved to images_dir.
    """
    images_by_page: dict[int, list[str]] = {}

    doc = fitz.open(str(pdf_path))
    for page_index in range(len(doc)):
        page_num = page_index + 1
        image_list = doc.get_page_images(page_index, full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception as e:
                print(f"  Warning: could not extract image xref {xref} on page {page_num}: {e}")
                continue

            img_bytes = base_image["image"]
            ext = "." + base_image["ext"]
            dest_name = f"{pdf_base_sanitized}_pymupdf_p{page_num}_{img_index:03d}{ext}"
            dest_path = images_dir / dest_name
            try:
                dest_path.write_bytes(img_bytes)
                images_by_page.setdefault(page_num, []).append(dest_name)
            except OSError as e:
                print(f"  Warning: could not write image {dest_path}: {e}")

    doc.close()
    total = sum(len(v) for v in images_by_page.values())
    print(f"  PyMuPDF extracted {total} images across {len(images_by_page)} pages.")
    return images_by_page


# --- Core Processing Logic ---

def process_pdf(pdf_path: Path, api_key: str, session_output_dir: Path, page_separator: str | None = PAGE_SEPARATOR_DEFAULT) -> tuple[str, str, list[str], Path, Path]:
    """
    Processes a single PDF file using Mistral OCR and PyMuPDF, saves results.

    Returns:
        A tuple (pdf_base_name, final_markdown_content, list_of_image_filenames, path_to_markdown_file, path_to_images_dir)
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
    ocr_response: OCRResponse | None = None
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
        ocr_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=True
        )
        print(f"  OCR processing complete for {pdf_path.name}.")

        # Optional: Save Raw OCR Response
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

        # --- Extract all images via PyMuPDF for supplemental coverage ---
        print(f"  Extracting images via PyMuPDF...")
        pdfimages_by_page = extract_images_pymupdf(pdf_path, images_dir, pdf_base_sanitized)

        # --- Process OCR Response -> Markdown & Images ---
        global_image_counter = 1
        updated_markdown_pages = []
        extracted_image_filenames = []
        mistral_count_by_page: dict[int, int] = {}  # 1-based page -> count of Mistral images

        print(f"  Extracting Mistral OCR images and generating Markdown...")
        for page_index, page in enumerate(ocr_response.pages):
            current_page_markdown = page.markdown
            page_image_mapping = {}
            page_num = page_index + 1

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
                    print(f"  Warning: Base64 decode error for image {image_obj.id} on page {page_num}: {decode_err}")
                    continue

                original_ext = Path(image_obj.id).suffix
                ext = original_ext if original_ext else ".png"
                new_image_name = f"{pdf_base_sanitized}_p{page_num}_img{global_image_counter}{ext}"
                global_image_counter += 1

                image_output_path = images_dir / new_image_name
                try:
                    with open(image_output_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    extracted_image_filenames.append(new_image_name)
                    page_image_mapping[image_obj.id] = new_image_name
                except IOError as io_err:
                    print(f"  Warning: Could not write image file {image_output_path}: {io_err}")
                    continue

            mistral_count_by_page[page_num] = len(page_image_mapping)
            updated_page_markdown = replace_images_in_markdown_with_wikilinks(current_page_markdown, page_image_mapping)

            # --- Append supplemental PyMuPDF images for this page ---
            pdfimg_files = pdfimages_by_page.get(page_num, [])
            mistral_count = mistral_count_by_page[page_num]
            extra_pdfimg = pdfimg_files[mistral_count:]  # Images beyond what Mistral found

            if extra_pdfimg:
                img_refs = "\n".join(f"![[{name}]]" for name in extra_pdfimg)
                supplement = (
                    f"\n\n> [!note] Supplemental Images — Page {page_num}\n"
                    f"> The following images were found by raw extraction but not placed inline by OCR.\n\n"
                    f"{img_refs}"
                )
                updated_page_markdown = updated_page_markdown + supplement
                extracted_image_filenames.extend(extra_pdfimg)
                print(f"  Page {page_num}: added {len(extra_pdfimg)} supplemental image(s) from PyMuPDF.")

            updated_markdown_pages.append(updated_page_markdown)

        separator = f"\n\n{page_separator}\n\n" if page_separator else "\n\n"
        final_markdown_content = separator.join(updated_markdown_pages)
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

    # Use the API key from environment first, then fall back to form input
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
