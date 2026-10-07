#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import requests

try:
    from .source_utils import source_hash, write_original
except ImportError:
    from source_utils import source_hash, write_original


BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "main",
    "li",
    "ul",
    "ol",
    "pre",
    "blockquote",
    "table",
    "tr",
}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
SKIP_TAGS = {"script", "style", "noscript", "svg"}
FATAL_EXTRACTION_MARKERS = (
    "Conversion to HTML had a Fatal error",
    "LaTeXML encountered an error",
    "Fatal error occurred",
)


def normalize_whitespace(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_extracted_body(body: str, min_chars: int = 300) -> None:
    """Reject upstream error pages and implausibly short extraction results."""
    normalized = normalize_whitespace(body)
    for marker in FATAL_EXTRACTION_MARKERS:
        if marker.casefold() in normalized.casefold():
            raise ValueError(f"Upstream conversion failure detected: {marker}")
    if len(normalized) < min_chars:
        raise ValueError(
            f"Extracted body is too short ({len(normalized)} characters; minimum {min_chars})."
        )


class Element:
    def __init__(self, tag: str, attrs: dict[str, str] | None = None) -> None:
        self.tag = tag
        self.attrs = attrs or {}
        self.children: list[Element | str] = []

    def descendants(self, tag: str) -> list[Element]:
        result = []
        for child in self.children:
            if isinstance(child, Element):
                if child.tag == tag:
                    result.append(child)
                result.extend(child.descendants(tag))
        return result


class ArticleExtractor(HTMLParser):
    """Preserve nested prose, links, math, tables and addressable headings."""
    VOID_TAGS = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

    def __init__(self, base_url: str = '') -> None:
        super().__init__(convert_charrefs=True)
        self.root = Element('root')
        self.stack = [self.root]
        self.base_url = base_url
        self.section_number = 0

    @property
    def title(self) -> str | None:
        titles = self.root.descendants('title')
        return normalize_whitespace(self.plain(titles[0])) if titles else None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Element(tag, {key: value or '' for key, value in attrs})
        self.stack[-1].children.append(node)
        if tag not in self.VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)

    def plain(self, node: Element | str) -> str:
        if isinstance(node, str):
            return node
        return ''.join(self.plain(child) for child in node.children)

    def render_node(self, node: Element | str) -> str:
        if isinstance(node, str):
            return node
        tag = node.tag
        classes = node.attrs.get('class', '').lower().split()
        if tag in SKIP_TAGS | {'head', 'nav', 'aside', 'footer', 'button', 'form'}:
            return ''
        if any(re.search(r'(?:^|[-_])(sidebar|navigation|breadcrumb|toc)(?:$|[-_])', value) for value in classes):
            return ''
        if node.attrs.get('aria-hidden') == 'true':
            return ''
        if tag == 'math':
            tex = node.attrs.get('alttext')
            if not tex:
                annotations = node.descendants('annotation')
                tex = next((self.plain(item) for item in annotations if 'tex' in item.attrs.get('encoding', '').lower()), None)
            value = normalize_whitespace(tex or self.plain(node))
            return f'\n\n$$\n{value}\n$$\n\n' if node.attrs.get('display') == 'block' else f'${value}$'
        if tag == 'pre':
            code = self.plain(node).strip('\n')
            fence = '`' * max(3, max((len(run) + 1 for run in re.findall(r'`+', code)), default=3))
            return f'\n\n{fence}\n{code}\n{fence}\n\n'
        if tag == 'table':
            rows = []
            for row in node.descendants('tr'):
                cells = [child for child in row.children if isinstance(child, Element) and child.tag in {'th', 'td'}]
                if cells:
                    values = []
                    for cell in cells:
                        value = normalize_whitespace(self.render_children(cell)).replace('|', '\\|').replace('\n', '<br>')
                        for attribute in ('rowspan', 'colspan'):
                            if cell.attrs.get(attribute, '1') != '1':
                                value += f' [{attribute}={cell.attrs[attribute]}]'
                        values.append(value)
                    rows.append(values)
            if not rows:
                return '\n\n' + normalize_whitespace(self.plain(node)) + '\n\n'
            width = max(map(len, rows))
            rows = [row + [''] * (width - len(row)) for row in rows]
            first_row = node.descendants('tr')[0]
            if not any(isinstance(child, Element) and child.tag == 'th' for child in first_row.children):
                rows.insert(0, [f'列 {i + 1}' for i in range(width)])
            captions = node.descendants('caption')
            caption = normalize_whitespace(self.plain(captions[0])) + '\n\n' if captions else ''
            lines = ['| ' + ' | '.join(row) + ' |' for row in rows]
            lines.insert(1, '| ' + ' | '.join(['---'] * width) + ' |')
            return '\n\n' + caption + '\n'.join(lines) + '\n\n'
        value = self.render_children(node)
        if tag in HEADING_TAGS:
            heading = normalize_whitespace(value)
            if not heading:
                return ''
            self.section_number += 1
            return f'\n\n<a id="source-section-{self.section_number}"></a>\n\n{"#" * int(tag[1])} {heading}\n\n'
        if tag == 'a' and value.strip():
            href = node.attrs.get('href', '')
            target = urljoin(self.base_url, href)
            if href and urlparse(target).scheme in {'', 'http', 'https'}:
                return f'[{normalize_whitespace(value)}]({quote(target, safe=":/?#=&%+-._~")})'
        if tag == 'img':
            return f" [图片：{node.attrs.get('alt', '').strip() or '无替代文本'}] "
        if tag == 'code':
            return '`' + self.plain(node).strip() + '`'
        if tag in {'strong', 'b'}:
            return '**' + value + '**'
        if tag == 'br':
            return '\n'
        if tag == 'li':
            return '\n- ' + value.strip() + '\n'
        if tag in BLOCK_TAGS | {'figure', 'figcaption', 'dl', 'dt', 'dd'}:
            return '\n\n' + value.strip() + '\n\n'
        return value

    def render_children(self, node: Element) -> str:
        return ''.join(self.render_node(child) for child in node.children)

    def render(self) -> str:
        self.section_number = 0
        # GitHub blob pages can keep the actual file only in embedded JSON.
        # Read that explicit payload, rather than treating the loading UI as prose.
        if urlparse(self.base_url).hostname == 'github.com' and '/blob/' in urlparse(self.base_url).path:
            files = []
            def collect(value: object) -> None:
                if isinstance(value, dict):
                    lines = value.get('rawLines')
                    if isinstance(lines, list) and lines and all(isinstance(line, str) for line in lines):
                        files.append(lines)
                    for child in value.values():
                        collect(child)
                elif isinstance(value, list):
                    for child in value:
                        collect(child)
            for node in self.root.descendants('script'):
                if node.attrs.get('type') == 'application/json' and node.attrs.get('data-target') == 'react-app.embeddedData':
                    try:
                        collect(json.loads(self.plain(node)))
                    except (ValueError, TypeError):
                        continue
            if len(files) == 1:
                title = Element('h1')
                title.children = [files[0][0]]
                source = Element('pre')
                source.children = ['\n'.join(files[0])]
                return (self.render_node(title) + self.render_node(source)).strip()
        articles = self.root.descendants('article')
        mains = self.root.descendants('main')
        bodies = self.root.descendants('body')
        cards = [node for node in self.root.descendants('div')
                 if 'model-card-content' in node.attrs.get('class', '').split()]
        if len(cards) == 1:
            container = cards[0]
        elif len(articles) == 1 and (not mains or len(self.plain(articles[0]).strip()) >= 300):
            container = articles[0]
        else:
            container = (mains or bodies or [self.root])[0]
        rendered = self.render_node(container)
        # Code indentation is evidence too; never flatten it with prose whitespace.
        result = []
        fence = None
        blanks = 0
        for line in rendered.splitlines():
            marker = re.match(r'^(`{3,})', line)
            if marker:
                if fence is None:
                    fence = marker.group(1)
                elif marker.group(1) == fence:
                    fence = None
                result.append(line)
                blanks = 0
                continue
            if fence:
                result.append(line)
                continue
            line = re.sub(r'[ \t]+', ' ', line).strip()
            blanks = blanks + 1 if not line else 0
            if blanks <= 2:
                result.append(line)
        return '\n'.join(result).strip()


def render_markdown(
    title: str,
    url: str,
    body: str,
    html_path: str | None = None,
) -> str:
    lines = [f"# {title}", ""]
    if html_path:
        lines.append(f"- Source HTML: `{html_path}`")
        if Path(html_path).is_file():
            lines.append(f"- Source SHA256: `{source_hash(Path(html_path))}`")
    lines.append(f"- Source URL: {url}")
    lines.append("- Generated from: `scripts/fetch_web_text.py`")
    lines.append("- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)")
    # Give introductory prose before the first heading a stable citation target.
    # This does not renumber the heading anchors generated by ArticleExtractor.
    if '<a id="source-section-0"></a>' not in body:
        body = '<a id="source-section-0"></a>\n\n' + body
    lines.extend(["", "## Extracted Text", "", body, ""])
    return "\n".join(lines)


def extract_jsonld_article(html_text: str) -> tuple[str | None, str | None]:
    field_title = None
    headline_match = re.search(r'"headline":"(.*?)"', html_text, flags=re.DOTALL)
    if headline_match:
        try:
            field_title = json.loads(f'"{headline_match.group(1)}"')
        except json.JSONDecodeError:
            field_title = normalize_whitespace(headline_match.group(1))

    article_match = re.search(r'"articleBody":"(.*?)","wordCount"', html_text, flags=re.DOTALL)
    if article_match:
        try:
            article_body = json.loads(f'"{article_match.group(1)}"')
        except json.JSONDecodeError:
            article_body = article_match.group(1)
        article_body = normalize_whitespace(article_body)
        if article_body:
            return field_title, article_body

    matches = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for raw in matches:
        try:
            data = json.loads(html.unescape(raw.strip()))
        except json.JSONDecodeError:
            continue

        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            article_body = item.get("articleBody")
            title = item.get("headline") or item.get("name")
            if isinstance(article_body, str) and article_body.strip():
                return title if isinstance(title, str) else None, normalize_whitespace(article_body)
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a webpage and save extracted text as markdown.")
    parser.add_argument("url", help="Source URL to fetch.")
    parser.add_argument("output", help="Output markdown path.")
    parser.add_argument("--title", help="Override output title.")
    parser.add_argument("--html-out", help="Optional path to save the raw HTML response.")
    parser.add_argument('--force', action='store_true', help='Rebuild derived markdown from the saved HTML; originals are immutable.')
    parser.add_argument(
        "--min-chars",
        type=int,
        default=300,
        help="Minimum extracted body length. Default: 300.",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    default_html = out_path.parent.parent / 'html' / out_path.with_suffix('.html').name if out_path.parent.name == 'text' else out_path.with_suffix('.html')
    html_path = Path(args.html_out) if args.html_out else default_html
    if out_path.exists():
        existing = out_path.read_text(encoding='utf-8')
        saved_url = re.search(r'^- Source URL: (.+)$', existing, flags=re.MULTILINE)
        if saved_url and saved_url.group(1).strip() != args.url:
            print('Existing snapshot belongs to another URL; choose a new stem.', file=sys.stderr)
            return 1
    if html_path.exists() and out_path.exists() and not args.force:
        print(f'Skip existing snapshot {html_path} and text {out_path}')
        return 0
    if html_path.exists():
        html_text = html_path.read_text(encoding='utf-8')
    else:
        response = requests.get(args.url, timeout=30, headers={'User-Agent': 'Mozilla/5.0 (compatible; wiki-ingest/1.0)'})
        response.raise_for_status()
        html_text = response.text
    jsonld_title, jsonld_body = extract_jsonld_article(html_text)
    parser_ = ArticleExtractor(args.url)
    parser_.feed(html_text)
    body = parser_.render()
    if len(body) < 300 and jsonld_body:
        body = jsonld_body
    try:
        validate_extracted_body(body, min_chars=args.min_chars)
    except ValueError as exc:
        print(f"Failed to extract body from {args.url}: {exc}", file=sys.stderr)
        return 1

    title = args.title or jsonld_title or parser_.title or Path(urlparse(args.url).path).name or "Untitled"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not html_path.exists():
        write_original(html_path, html_text.encode('utf-8'))
        print(f"Wrote {html_path}")
    out_path.write_text(
        render_markdown(title, args.url, body, str(html_path)),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
