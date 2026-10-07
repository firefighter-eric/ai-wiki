#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import fitz

try:
    from .source_utils import source_hash
except ImportError:
    from source_utils import source_hash


def extract_pdf_text(pdf_path: Path) -> str:
    parts = []
    has_text = False
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text(sort=True).strip()
            has_text = has_text or bool(text)
            marker = f'<a id="page-{index}"></a>\n\n### PDF 第 {index} 页\n\n'
            parts.append(marker + (text or '[本页未提取到文本；需要图像检查或 OCR。]'))
    if not has_text:
        raise ValueError(f'No text extracted from PDF: {pdf_path}; OCR/visual review is required.')
    return "\n".join(parts).strip()


def build_output_path(pdf_path: Path, raw_root: Path, out_root: Path) -> Path:
    rel = pdf_path.relative_to(raw_root)
    return out_root / rel.with_suffix(".md")


def render_markdown(
    pdf_path: Path,
    raw_root: Path,
    extracted: str,
    source_url: str | None = None,
    title: str | None = None,
) -> str:
    rel = pdf_path.relative_to(raw_root)
    title = title or pdf_path.stem
    url_metadata = f"- Source URL: {source_url}\n" if source_url else ''
    return (
        f"# {title}\n\n"
        f"- Source PDF: `raw/pdf/{rel.as_posix()}`\n"
        f"- Source SHA256: `{source_hash(pdf_path)}`\n"
        f"{url_metadata}"
        f"- Generated from: `scripts/extract_pdf_text.py`\n\n"
        "- Extraction: `pymupdf-pages-v2` (sorted text, page anchors; tables/formulas/figures require review)\n\n"
        "## Extracted Text\n\n"
        f"{extracted}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract text from PDFs in raw/pdf/ into markdown files in raw/text/."
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        help="Optional list of PDF paths. Defaults to all PDFs under raw/pdf/.",
    )
    parser.add_argument(
        "--raw-root",
        default="raw/pdf",
        help="Root directory containing source PDFs.",
    )
    parser.add_argument(
        "--out-root",
        default="raw/text",
        help="Root directory for extracted markdown files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing markdown files in raw/text/.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.out_root).resolve()

    if args.pdfs:
        pdf_paths = [Path(p).resolve() for p in args.pdfs]
    else:
        pdf_paths = sorted(raw_root.rglob("*.pdf"))

    if not pdf_paths:
        print("No PDF files found.", file=sys.stderr)
        return 1

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            print(f"Missing PDF: {pdf_path}", file=sys.stderr)
            return 1

        out_path = build_output_path(pdf_path, raw_root, out_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.force:
            print(f"Skip existing {out_path}")
            continue

        try:
            extracted = extract_pdf_text(pdf_path)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        markdown = render_markdown(pdf_path, raw_root, extracted)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
