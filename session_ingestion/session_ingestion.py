import gc
import os
import tempfile
import uuid
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import fitz
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document
from markitdown import MarkItDown
import camelot

from configs import CHUNK_SIZE, CHUNK_OVERLAP
from session_ingestion.markdown_cleaning import (
    clean_page_texts,
    repair_hyphenated_breaks,
    normalize_whitespace,
    format_table_as_markdown,
    dedupe_text_against_tables,
)


_markitdown = MarkItDown()


_markdown_splitter = MarkdownTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def _split_pdf_into_single_pages(tmp_path, tmp_dir):
    page_paths = []
    with fitz.open(tmp_path) as src:
        for page_index in range(len(src)):
            single_page_doc = fitz.open()
            single_page_doc.insert_pdf(src, from_page=page_index, to_page=page_index)
            page_path = os.path.join(tmp_dir, f"page_{page_index}_{uuid.uuid4().hex}.pdf")
            single_page_doc.save(page_path)
            single_page_doc.close()
            page_paths.append(page_path)
    return page_paths


def _markdown_for_page(page_path, page_index, filename):
    try:
        result = _markitdown.convert_local(page_path)
        text = result.text_content
        if text and text.strip():
            return text
    except Exception as e:
        print(
            f"[ingestion] MarkItDown failed on '{filename}' page {page_index}, "
            f"falling back to plain-text extraction: {e}"
        )

    try:
        with fitz.open(page_path) as page_pdf:
            return page_pdf[0].get_text()
    except Exception as e:
        print(
            f"[ingestion] fallback text extraction also failed on '{filename}' "
            f"page {page_index}: {e}"
        )
        return ""


def _process_markdown_chunks(markdown_text, page_num, filename, items):
    chunks = _markdown_splitter.split_text(markdown_text)
    for chunk in chunks:
        if not chunk.strip():
            continue
        items.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "page": page_num,
                    "type": "text",
                },
            )
        )


def _process_tables(tmp_path, filename, items):
    try:
        tables = camelot.read_pdf(tmp_path, pages="all")
    except Exception as e:
        print(f"[ingestion] table extraction skipped for {filename}: {e}")
        return

    if not tables:
        return

    for table in tables:
        table_text = format_table_as_markdown(table.df)
        if not table_text:
            continue
        actual_page_num = table.page - 1
        items.append(
            Document(
                page_content=table_text,
                metadata={
                    "source": filename,
                    "page": actual_page_num,
                    "type": "table",
                },
            )
        )


def process_uploaded_pdf(uploaded_file):
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        raise ValueError(
            f"'{filename}' was received as an empty file (0 bytes). "
            f"This can happen if the upload was interrupted -- please "
            f"try uploading it again."
        )

    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"infobee_{uuid.uuid4().hex}.pdf")
    page_paths = []

    try:
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
            f.flush()
            os.fsync(f.fileno())
        if os.path.getsize(tmp_path) == 0:
            raise ValueError(
                f"'{filename}' was written as an empty file on the server. "
                f"Please try uploading it again."
            )

        items = []

        page_paths = _split_pdf_into_single_pages(tmp_path, tmp_dir)

        raw_page_texts = [
            (page_index, _markdown_for_page(page_path, page_index, filename))
            for page_index, page_path in enumerate(page_paths)
        ]
        cleaned_page_texts = clean_page_texts(raw_page_texts)

        for page_index, text in cleaned_page_texts:
            text = normalize_whitespace(repair_hyphenated_breaks(text))
            if text:
                _process_markdown_chunks(text, page_index, filename, items)

        gc.collect()

        _process_tables(tmp_path, filename, items)

        items = dedupe_text_against_tables(items)
    finally:
        for page_path in page_paths:
            try:
                if os.path.exists(page_path):
                    os.unlink(page_path)
            except PermissionError as e:
                print(f"[ingestion] could not delete temp page file {page_path}: {e}")
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except PermissionError as e:
            print(f"[ingestion] could not delete temp file {tmp_path} (still locked): {e}")

    return items


def process_uploaded_pdfs(uploaded_files):
    all_items = []
    for uploaded_file in uploaded_files:
        all_items.extend(process_uploaded_pdf(uploaded_file))
    return all_items