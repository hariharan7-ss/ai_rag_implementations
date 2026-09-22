"""Index the HR corpus into the eval namespace with stable chunk ids.

The app's upload endpoint mints a random uuid document_id ("doc_3f9a1c22"), so
a golden set can never point at the same chunk twice across a re-ingest. Here
the document_id is derived from the file name instead, so "HR-207 Leave
Policy.pdf" always yields hr_207_leave_policy_chunk_N.
"""

import argparse
import os
import re

import common  # noqa: F401  (must come first: sets sys.path + PINECONE_NAMESPACE)

from pypdf import PdfReader

from app.ingestion.cleaner import clean_text
from app.ingestion.chunker import recursive_chunk
from app.ingestion.metadata import create_metadata
from app.embeddings.model import embedding_model
from app.vectorstore.pinecone_store import pinecone_store

SUPPORTED = (".pdf", ".docx", ".txt", ".md")


def ascii_safe(text):
    # The Windows console is cp1252, so a non-ASCII character in a file name
    # would otherwise kill the print rather than the ingest.
    return str(text).encode("ascii", "replace").decode("ascii")


def slugify(name):
    """File stem -> document_id: "HR-207 Leave Policy" -> "hr_207_leave_policy"."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def read_pdf(path):
    text = ""
    for page in PdfReader(path).pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def read_docx(path):
    # Imported lazily on purpose: python-docx is not installed and is not in
    # requirements.txt, and .pdf/.txt/.md must keep working without it.
    try:
        import docx
    except ImportError:
        raise SystemExit(
            f"{ascii_safe(os.path.basename(path))} is a .docx but python-docx is not "
            "installed. Either pip install python-docx, or convert the file to .txt."
        )
    return "\n".join(p.text for p in docx.Document(path).paragraphs)


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def read_file(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    return read_text(path)


def corpus_files(corpus_dir):
    """Supported files sorted by name, plus the names that were ignored."""
    if not os.path.isdir(corpus_dir):
        raise SystemExit(
            f"corpus dir not found: {ascii_safe(corpus_dir)}\n"
            "Create it and drop the HR policy files in, then re-run."
        )
    keep = []
    ignored = []
    for name in sorted(os.listdir(corpus_dir)):
        path = os.path.join(corpus_dir, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() in SUPPORTED:
            keep.append(path)
        else:
            ignored.append(name)
    return keep, ignored


def main():
    parser = argparse.ArgumentParser(
        description="Index the HR corpus into the eval namespace with stable chunk ids."
    )
    parser.add_argument(
        "--corpus",
        default=common.CORPUS_DIR,
        help="corpus directory (default: eval/corpus)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="chunk and write chunks.jsonl without embedding or upserting",
    )
    args = parser.parse_args()

    files, ignored = corpus_files(args.corpus)

    print(f"namespace: {common.EVAL_NAMESPACE}")
    if args.dry_run:
        print("dry run: chunking only, no embedding and no upsert")
    if ignored:
        print(f"ignored {len(ignored)} file(s): {ascii_safe(', '.join(ignored))}")
    if not files:
        raise SystemExit(
            f"no {', '.join(SUPPORTED)} files in {ascii_safe(args.corpus)} "
            "-- nothing to ingest."
        )

    all_chunks = []
    seen = {}  # doc_id -> file name, so two files cannot claim one id
    indexed = 0

    for path in files:
        name = os.path.basename(path)
        doc_id = slugify(os.path.splitext(name)[0])
        if not doc_id:
            print(
                f"WARNING: {ascii_safe(name)} has no alphanumeric characters in its name "
                "to slug into a document_id -- skipped. Rename it; the golden set needs a "
                "readable chunk id prefix."
            )
            continue
        if doc_id in seen:
            print(
                f"WARNING: {ascii_safe(name)} slugs to '{doc_id}', already taken by "
                f"{ascii_safe(seen[doc_id])} -- skipped. Rename one of them; the golden "
                "set cannot tell their chunks apart."
            )
            continue
        seen[doc_id] = name

        chunks = recursive_chunk(clean_text(read_file(path)))
        if not chunks:
            print(f"WARNING: {ascii_safe(name)} produced 0 chunks -- skipped (empty file?)")
            continue

        metadata = create_metadata(doc_id, name, "file", name, chunks)

        if not args.dry_run:
            # Clear the old chunks first so a document that shrank does not leave
            # orphans behind. Delete-by-metadata-filter is a no-op on Pinecone
            # serverless (the store catches and prints that) -- re-ingest is still
            # safe, because chunk ids are stable and the upsert overwrites them.
            pinecone_store.delete_document(doc_id)
            embeddings = embedding_model.encode([c["text"] for c in metadata])
            pinecone_store.upsert_chunks(metadata, embeddings)

        all_chunks.extend(metadata)
        indexed += 1
        print(f"{ascii_safe(name)} -> {doc_id}  {len(chunks)} chunks")

    if not all_chunks:
        raise SystemExit("no chunks produced -- chunks.jsonl left untouched.")

    common.write_jsonl(common.CHUNKS_PATH, all_chunks)
    out = os.path.relpath(common.CHUNKS_PATH, common.APP_ROOT).replace(os.sep, "/")
    print(f"\n{indexed} files, {len(all_chunks)} chunks -> {out}")


if __name__ == "__main__":
    main()
