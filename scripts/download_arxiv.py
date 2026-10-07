#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

import requests

try:
    from .source_utils import write_original
except ImportError:
    from source_utils import write_original


def normalize_arxiv_id(value: str) -> str:
    value = value.strip()
    value = re.sub(r'^https?://(?:www\.)?(?:arxiv\.org/(?:abs|pdf|html)/|ar5iv\.labs\.arxiv\.org/html/)', '', value, flags=re.IGNORECASE)
    value = re.sub(r'^arxiv:', '', value, flags=re.IGNORECASE).split('?', 1)[0].split('#', 1)[0]
    value = value.removesuffix('.pdf')
    if re.fullmatch(r'(?:\d{4}\.\d{4,5}|[a-z][a-z.-]*/\d{7})(?:v[1-9]\d*)?', value, flags=re.IGNORECASE):
        return value
    raise ValueError(f"Unsupported arXiv identifier: {value}")


def load_fetch_web_text_module():
    return importlib.import_module('scripts.fetch_web_text' if __package__ else 'fetch_web_text')


def load_extract_pdf_text_module():
    return importlib.import_module('scripts.extract_pdf_text' if __package__ else 'extract_pdf_text')


def download_file(url: str, out_path: Path, force: bool) -> None:
    if out_path.exists():
        print(f"Skip existing {out_path}")
        return

    response = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0 (compatible; wiki-ingest/1.0)"},
    )
    response.raise_for_status()
    if not response.content.startswith(b'%PDF-'):
        raise ValueError(f'Expected a PDF, received other content from {url}')
    write_original(out_path, response.content)
    print(f"Wrote {out_path}")


def validate_full_article_html(html_text: str) -> None:
    """An arXiv abstract/metadata page is not a full-text HTML source."""
    abstract_page = re.search(r'class=[\"\'][^\"\']*\babstract\b[^\"\']*\bmathjax\b', html_text)
    if abstract_page and 'ltx_document' not in html_text:
        raise ValueError('arXiv abstract page is not full text; use the PDF fallback.')


def fetch_html_markdown(
    arxiv_id: str,
    html_path: Path,
    text_path: Path,
    title: str | None,
    force: bool,
) -> None:
    validate_text_identity(text_path, arxiv_id)
    if html_path.exists() and text_path.exists() and not force:
        print(f"Skip existing {html_path}")
        print(f"Skip existing {text_path}")
        return

    module = load_fetch_web_text_module()
    if html_path.exists():
        html_text = html_path.read_text(encoding='utf-8')
        validate_full_article_html(html_text)
        saved_url = f'https://arxiv.org/html/{arxiv_id}'
        if text_path.exists():
            url_match = re.search(r'^- Source URL: (.+)$', text_path.read_text(encoding='utf-8'), flags=re.MULTILINE)
            if url_match:
                saved_url = url_match.group(1).strip()
                if normalize_arxiv_id(saved_url) != arxiv_id:
                    raise ValueError('Existing source uses another arXiv ID/version; choose a new stem.')
        extractor = module.ArticleExtractor(saved_url)
        extractor.feed(html_text)
        body = extractor.render()
        module.validate_extracted_body(body, min_chars=2_000)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(module.render_markdown(title or extractor.title or arxiv_id, saved_url, body, str(html_path)), encoding='utf-8')
        print(f'Wrote {text_path} from saved HTML')
        return
    html_urls = [
        f"https://arxiv.org/html/{arxiv_id}",
        f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
    ]

    last_error: Exception | None = None
    for url in html_urls:
        try:
            response = requests.get(
                url,
                timeout=60,
                headers={"User-Agent": "Mozilla/5.0 (compatible; wiki-ingest/1.0)"},
            )
            response.raise_for_status()
            validate_full_article_html(response.text)
            jsonld_title, jsonld_body = module.extract_jsonld_article(response.text)
            parser = module.ArticleExtractor(url)
            parser.feed(response.text)
            body = parser.render()
            if len(body) < 2_000 and jsonld_body:
                body = jsonld_body
            module.validate_extracted_body(body, min_chars=2_000)

            final_title = title or jsonld_title or parser.title or arxiv_id
            html_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.parent.mkdir(parents=True, exist_ok=True)
            write_original(html_path, response.text.encode('utf-8'))
            text_path.write_text(
                module.render_markdown(final_title, url, body, str(html_path)),
                encoding="utf-8",
            )
            print(f"Wrote {html_path}")
            print(f"Wrote {text_path}")
            return
        except (requests.RequestException, ValueError) as exc:
            last_error = exc

    raise RuntimeError(f"Failed to fetch arXiv HTML for {arxiv_id}: {last_error}")


def fallback_to_pdf(
    pdf_path: Path,
    pdf_root: Path,
    text_path: Path,
    source_url: str | None = None,
    title: str | None = None,
) -> None:
    if not pdf_path.is_file():
        raise RuntimeError(
            "arXiv HTML extraction failed and no local PDF is available for fallback: "
            f"{pdf_path}"
        )
    module = load_extract_pdf_text_module()
    extracted = module.extract_pdf_text(pdf_path)
    if len(extracted.strip()) < 2_000:
        raise RuntimeError(
            f"PDF fallback extraction is implausibly short ({len(extracted.strip())} characters)."
        )
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(
        module.render_markdown(pdf_path.resolve(), pdf_root.resolve(), extracted, source_url, title),
        encoding="utf-8",
    )
    print(f"Wrote {text_path} from PDF fallback")


def validate_text_identity(text_path: Path, arxiv_id: str) -> None:
    if not text_path.exists():
        return
    match = re.search(r'^- Source URL: (.+)$', text_path.read_text(encoding='utf-8'), flags=re.MULTILINE)
    if match and normalize_arxiv_id(match.group(1).strip()) != arxiv_id:
        raise ValueError('Existing source uses another arXiv ID/version; choose a new stem.')


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download arXiv sources into raw/pdf/, raw/html/, and raw/text/."
    )
    parser.add_argument("arxiv_id", help="arXiv id, abs/pdf/html URL, or arXiv:ID.")
    parser.add_argument("--stem", help="Output file stem. Defaults to the arXiv id.")
    parser.add_argument("--title", help="Markdown title for raw/text output.")
    parser.add_argument("--pdf-root", default="raw/pdf", help="Output root for downloaded PDFs.")
    parser.add_argument("--html-root", default="raw/html", help="Output root for saved HTML files.")
    parser.add_argument("--text-root", default="raw/text", help="Output root for generated markdown.")
    parser.add_argument("--skip-pdf", action="store_true", help="Do not download the PDF.")
    parser.add_argument("--skip-text", action="store_true", help="Do not fetch arXiv HTML into markdown.")
    parser.add_argument("--force", action="store_true", help="Rebuild derived text; never overwrite original HTML/PDF.")
    args = parser.parse_args()

    arxiv_id = normalize_arxiv_id(args.arxiv_id)
    stem = args.stem or arxiv_id.replace('/', '_')
    if Path(stem).name != stem or stem in {'.', '..'}:
        parser.error('Stem must be a filename without path components.')

    if args.skip_pdf and args.skip_text:
        print("Nothing to do: both --skip-pdf and --skip-text are set.", file=sys.stderr)
        return 1

    pdf_root = Path(args.pdf_root)
    pdf_path = pdf_root / f"{stem}.pdf"
    text_path = Path(args.text_root) / f'{stem}.md'
    validate_text_identity(text_path, arxiv_id)
    if not args.skip_pdf:
        download_file(f"https://arxiv.org/pdf/{arxiv_id}.pdf", pdf_path, args.force)

    if not args.skip_text:
        html_path = Path(args.html_root) / f"{stem}.html"
        if text_path.exists() and not args.force and (html_path.exists() or pdf_path.exists()):
            print(f'Skip existing {text_path}; use --force to rebuild from saved originals.')
            return 0
        try:
            fetch_html_markdown(arxiv_id, html_path, text_path, args.title, args.force)
        except RuntimeError as exc:
            print(f"HTML extraction unavailable: {exc}", file=sys.stderr)
            fallback_to_pdf(pdf_path, pdf_root, text_path, f'https://arxiv.org/pdf/{arxiv_id}', args.title)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
