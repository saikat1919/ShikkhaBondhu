import gc
import os
import tempfile
import uuid
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import camelot

from configs import CHUNK_SIZE, CHUNK_OVERLAP


def _process_text_chunks(text, page_num, text_splitter, filename, items):
    chunks = text_splitter.split_text(text)
    for chunk in chunks:
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
        df = table.df
        actual_page_num = table.page - 1
        table_text = "\n".join([" | ".join(map(str, row)) for row in df.values])

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
    tmp_path = os.path.join(tmp_dir, f"askme_{uuid.uuid4().hex}.pdf")

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

        loader = PyMuPDFLoader(tmp_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

        items = []
        for page_doc in pages:
            page_num = page_doc.metadata.get("page")
            text_contents = page_doc.page_content
            if text_contents and text_contents.strip():
                _process_text_chunks(text_contents, page_num, text_splitter, filename, items)

        del pages
        gc.collect()

        _process_tables(tmp_path, filename, items)
    finally:
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