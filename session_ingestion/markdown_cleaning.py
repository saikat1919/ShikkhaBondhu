import re
from collections import Counter, defaultdict

from configs import BOILERPLATE_MIN_REPEAT_FRACTION, TABLE_DEDUP_OVERLAP_THRESHOLD

_PAGE_NUMBER_LINE = re.compile(r"^(page\s+)?\d{1,4}(\s+of\s+\d{1,4})?$", re.IGNORECASE)
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_TOKEN = re.compile(r"[a-z0-9]{3,}")


def _edge_indices(lines):
    n = len(lines)
    return list(range(min(2, n))) + list(range(max(0, n - 2), n))


def clean_page_texts(page_texts, min_repeat_fraction=None):
    if min_repeat_fraction is None:
        min_repeat_fraction = BOILERPLATE_MIN_REPEAT_FRACTION

    stage1 = []
    for page_index, text in page_texts:
        lines = text.split("\n")
        for i in _edge_indices(lines):
            if _PAGE_NUMBER_LINE.match(lines[i].strip()):
                lines[i] = ""
        stage1.append((page_index, "\n".join(lines)))

    if len(stage1) < 3:
        return stage1

    counts = Counter()
    for _, text in stage1:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        for line in ({lines[i] for i in _edge_indices(lines)} if lines else set()):
            counts[line] += 1

    threshold = max(2, round(len(stage1) * min_repeat_fraction))
    boilerplate = {line for line, c in counts.items() if c >= threshold}

    return [
        (page_index, "\n".join(ln for ln in text.split("\n") if ln.strip() not in boilerplate))
        for page_index, text in stage1
    ]


def repair_hyphenated_breaks(text):
    return _HYPHEN_LINEBREAK.sub(r"\1\2", text)


def normalize_whitespace(text):
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_NUMERIC_CELL = re.compile(r"-?\d+(\.\d+)?%?")


def _row_looks_like_header(row):
    cells = [str(c).strip() for c in row]
    if not cells:
        return False
    numeric = sum(1 for c in cells if _NUMERIC_CELL.fullmatch(c))
    return numeric <= len(cells) / 2


def format_table_as_markdown(df):

    def clean_cell(value):
        text = str(value).strip().replace("\n", " ")
        return text.replace("|", "\\|")

    rows = df.values.tolist()
    if not rows:
        return ""

    if _row_looks_like_header(rows[0]):
        header, body = rows[0], rows[1:]
    else:
        header = [f"Column {i + 1}" for i in range(len(rows[0]))]
        body = rows

    lines = [
        "| " + " | ".join(clean_cell(c) for c in header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    lines += ["| " + " | ".join(clean_cell(c) for c in row) + " |" for row in body]
    return "\n".join(lines)


def dedupe_text_against_tables(items, overlap_threshold=None):
    if overlap_threshold is None:
        overlap_threshold = TABLE_DEDUP_OVERLAP_THRESHOLD

    tables_by_page = defaultdict(list)
    for doc in items:
        if doc.metadata.get("type") == "table":
            tables_by_page[doc.metadata.get("page")].append(doc)

    if not tables_by_page:
        return items

    def token_set(text):
        return set(_TOKEN.findall(text.lower()))

    table_tokens_by_page = {
        page: set().union(*(token_set(t.page_content) for t in tables))
        for page, tables in tables_by_page.items()
    }

    kept = []
    for doc in items:
        if doc.metadata.get("type") != "text":
            kept.append(doc)
            continue

        table_tokens = table_tokens_by_page.get(doc.metadata.get("page"))
        chunk_tokens = token_set(doc.page_content)
        if not table_tokens or not chunk_tokens:
            kept.append(doc)
            continue

        overlap_ratio = len(chunk_tokens & table_tokens) / len(chunk_tokens)
        if overlap_ratio >= overlap_threshold:
            print(
                f"[ingestion] dropped duplicate text chunk on page "
                f"{doc.metadata.get('page')} of '{doc.metadata.get('source')}' "
                f"({overlap_ratio:.0%} overlap with a table on the same page)"
            )
            continue

        kept.append(doc)
    return kept