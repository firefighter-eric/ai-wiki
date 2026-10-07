#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .lint_wiki import parse_frontmatter
except ImportError:
    from lint_wiki import parse_frontmatter


ROOT = Path(__file__).resolve().parent.parent
QMD_TIMEOUT = 30.0


@dataclass(frozen=True)
class CollectionSpec:
    slug: str
    path: Path
    mask: str
    layer: str
    label: str
    priority: int


COLLECTION_SPECS = (
    CollectionSpec("index", ROOT, "index.md", "index", "索引层", 0),
    CollectionSpec("wiki", ROOT / "wiki", "**/*.md", "wiki", "知识层", 1),
    CollectionSpec("raw-text", ROOT / "raw" / "text", "**/*.md", "raw-text", "全文补查层", 2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search the local Obsidian wiki through qmd. "
            "Default mode prioritizes index.md and wiki/ over raw/text/."
        )
    )
    parser.add_argument("query", help="Query text.")
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Maximum number of merged results to print. Default: 8.",
    )
    parser.add_argument(
        "--mode",
        choices=("wiki-first", "fulltext"),
        default="wiki-first",
        help="wiki-first prioritizes wiki pages; fulltext allows raw/text to surface more aggressively.",
    )
    parser.add_argument(
        "--per-collection-limit",
        type=int,
        default=None,
        help="Override qmd result count for each collection search.",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Skip qmd's incremental index refresh before searching.",
    )
    parser.add_argument('--backend', choices=('auto', 'qmd', 'local'), default='auto', help='auto uses qmd with a read-only lexical fallback; local needs no index/models.')
    parser.add_argument('--json', action='store_true', help='Return structured paths, maturity and line locations.')
    parser.add_argument('--timeout', type=float, default=30, help='Maximum seconds for each qmd command.')
    return parser.parse_args()


def repo_prefix() -> str:
    digest = hashlib.sha1(str(ROOT).encode("utf-8")).hexdigest()[:8]
    return f"my-obsidian-{digest}"


def collection_name(spec: CollectionSpec) -> str:
    return f"{repo_prefix()}-{spec.slug}"


def normalize_path(value: str) -> str:
    text = value.strip()
    if text.startswith("qmd://"):
        text = text[len("qmd://") :]
        parts = text.split("/", 1)
        text = parts[1] if len(parts) == 2 else ""
    elif text.startswith(str(ROOT)):
        try:
            text = str(Path(text).resolve().relative_to(ROOT))
        except Exception:  # noqa: BLE001
            pass
    return text.lstrip("/")


def short_text(value: str, limit: int = 140) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def query_terms(query: str) -> list[str]:
    text = query.strip().lower()
    if not text:
        return []
    latin_terms = re.findall(r"[a-z0-9][a-z0-9._+-]+", text)
    cjk_chunks = re.findall(r"[\u4e00-\u9fff]+", text)
    cjk_stopwords = re.compile(
        r"(?:为什么|是什么|有什么|有哪些|如何|怎么|以及|比较|对比|区别|差异|和|与|及|的)"
    )
    cjk_terms = [
        part
        for chunk in cjk_chunks
        for part in cjk_stopwords.split(chunk)
        if len(part) >= 2
    ]
    terms = list(dict.fromkeys([*latin_terms, *cjk_terms]))
    return terms or [text]


def qmd_search_query(query: str) -> str:
    """Turn a natural-language question into stable BM25 search terms."""
    terms = query_terms(query)
    latin = [term for term in terms if re.fullmatch(r"[a-z0-9][a-z0-9._+-]+", term)]
    if len(latin) >= 2:
        return " ".join(latin)
    return " ".join(terms)


def matches_query_text(query: str, result: dict[str, Any]) -> bool:
    terms = query_terms(query)
    if not terms:
        return True
    haystack = " ".join(
        [
            result.get("path", ""),
            result.get("title", ""),
            result.get("snippet", ""),
        ]
    ).lower()
    matched = sum(1 for term in terms if term in haystack)
    required = 1 if len(terms) == 1 else 2
    return matched >= required


def installation_error() -> str:
    return "\n".join(
        [
            "qmd CLI not found.",
            "Install it with one of the official commands:",
            "  npm install -g @tobilu/qmd",
            "  bun install -g @tobilu/qmd",
            "",
            "If qmd later reports SQLite extension issues on macOS, install Homebrew sqlite:",
            "  brew install sqlite",
        ]
    )


def subtype_priority(path: str, query: str = "") -> int:
    if path == "index.md":
        return 0
    lowered = query.casefold()
    if re.search(r"(?:对比|比较|区别|差异|\bvs\b)", lowered):
        order = (
            "wiki/comparisons/",
            "wiki/topics/",
            "wiki/concepts/",
            "wiki/timelines/",
            "wiki/summaries/",
            "wiki/authors/",
            "raw/text/",
        )
    elif re.search(r"(?:时间线|演进|历史|timeline)", lowered):
        order = (
            "wiki/timelines/",
            "wiki/topics/",
            "wiki/concepts/",
            "wiki/comparisons/",
            "wiki/summaries/",
            "wiki/authors/",
            "raw/text/",
        )
    else:
        order = (
            "wiki/topics/",
            "wiki/concepts/",
            "wiki/comparisons/",
            "wiki/timelines/",
            "wiki/summaries/",
            "wiki/authors/",
            "raw/text/",
        )
    for index, prefix in enumerate(order, start=1):
        if path.startswith(prefix):
            return index
    return len(order) + 1


def run_qmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    qmd = shutil.which("qmd")
    if not qmd:
        raise FileNotFoundError(installation_error())
    env = os.environ.copy()
    # Repository-scoped, rebuildable runtime state; no global collection mutation.
    env['XDG_CACHE_HOME'] = str(ROOT / '.wiki-cache')
    env['QMD_CONFIG_DIR'] = str(ROOT / '.wiki-cache' / 'qmd-config')
    return subprocess.run(
        [qmd, '--index', repo_prefix(), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
        env=env,
        timeout=QMD_TIMEOUT,
    )


def ensure_collection(spec: CollectionSpec) -> None:
    name = collection_name(spec)
    # `qmd ls` opens the SQLite index and can fail before collection lookup
    # when the cache is temporarily unavailable. `collection show` is the
    # authoritative, lightweight existence check.
    probe = run_qmd(["collection", "show", name], check=False)
    if probe.returncode == 0:
        return

    add = run_qmd(
        [
            "collection",
            "add",
            str(spec.path),
            "--name",
            name,
            "--mask",
            spec.mask,
        ],
        check=False,
    )
    if add.returncode != 0:
        reprobe = run_qmd(["collection", "show", name], check=False)
        if reprobe.returncode == 0:
            return
        stderr = add.stderr.strip()
        stdout = add.stdout.strip()
        message = stderr or stdout or "unknown qmd error"
        raise RuntimeError(f"Failed to initialize qmd collection '{name}': {message}")


def load_title_index() -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for base in (ROOT / "wiki", ROOT / "raw" / "text"):
        if not base.exists():
            continue
        for path in base.rglob("*.md"):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except Exception:  # noqa: BLE001
                continue
            title = ""
            for line in lines[:30]:
                stripped = line.strip()
                if stripped.startswith("# "):
                    title = stripped[2:].strip()
                    break
            if title:
                mapping.setdefault(title, []).append(path.relative_to(ROOT).as_posix())

    mapping.setdefault("Wiki Index", []).append("index.md")
    return mapping


def resolve_real_path(
    item: dict[str, Any],
    layer: str,
    title_index: dict[str, list[str]],
) -> str:
    title = pick_first(item, ("title", "name"))
    if title:
        candidates = title_index.get(title, [])
        if layer == "wiki":
            preferred = [path for path in candidates if path.startswith("wiki/")]
            if preferred:
                return preferred[0]
        if layer == "raw-text":
            preferred = [path for path in candidates if path.startswith("raw/text/")]
            if preferred:
                return preferred[0]
        if layer == "index" and "index.md" in candidates:
            return "index.md"
        if candidates:
            return candidates[0]

    return normalize_path(pick_first(item, ("path", "file", "filepath", "id", "docid")))


def extract_result_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("results", "items", "matches", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise RuntimeError("Unsupported qmd JSON output shape.")


def pick_first(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def pick_score(item: dict[str, Any]) -> float:
    for key in ("score", "finalScore", "rerankScore"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def search_collection(
    spec: CollectionSpec,
    query: str,
    limit: int,
    title_index: dict[str, list[str]],
) -> list[dict[str, Any]]:
    name = collection_name(spec)
    completed = run_qmd(
        [
            "search",
            qmd_search_query(query),
            "-c",
            name,
            "--json",
            "-n",
            str(limit),
        ]
    )
    payload = json.loads(completed.stdout)
    results = []
    for item in extract_result_list(payload):
        title_value = pick_first(item, ("title", "name"))
        snippet_value = pick_first(item, ("snippet", "excerpt", "text", "preview"))
        docid_value = pick_first(item, ("docid", "id"))
        results.append(
            {
                "path": resolve_real_path(item, spec.layer, title_index),
                "title": title_value,
                "snippet": snippet_value,
                "docid": docid_value,
                "score": pick_score(item),
                "layer": spec.layer,
                "label": spec.label,
                "priority": spec.priority,
            }
        )
    return results


def merged_results(mode: str, query: str, per_collection_limit: int) -> list[dict[str, Any]]:
    title_index = load_title_index()
    all_results: list[dict[str, Any]] = []
    for spec in COLLECTION_SPECS:
        all_results.extend(search_collection(spec, query, per_collection_limit, title_index))

    deduped: dict[str, dict[str, Any]] = {}
    for result in all_results:
        if not matches_query_text(query, result):
            continue
        key = result["path"] or f"{result['layer']}::{result['title']}"
        previous = deduped.get(key)
        if previous is None or result["score"] > previous["score"]:
            deduped[key] = result

    results = [enrich_result(item) for item in deduped.values() if (ROOT / item['path']).is_file()]
    return order_results(results, mode, query)


def order_results(results: list[dict[str, Any]], mode: str, query: str) -> list[dict[str, Any]]:
    terms = query_terms(qmd_search_query(query))
    def title_coverage(item: dict[str, Any]) -> int:
        title = item.get('title', '').casefold()
        return sum(term in title for term in terms)
    if mode == "wiki-first":
        results.sort(
            key=lambda item: (
                item["priority"],
                -title_coverage(item),
                subtype_priority(item["path"], query),
                -item["score"],
                item["path"],
            )
        )
    else:
        results.sort(
            key=lambda item: (
                -item["score"],
                item["priority"],
                subtype_priority(item["path"], query),
                item["path"],
            )
        )
    return results


def enrich_result(item: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / item['path']
    text = path.read_text(encoding='utf-8')
    metadata, _ = parse_frontmatter(text)
    item['type'] = metadata.get('type', 'raw-text' if item['layer'] == 'raw-text' else 'index')
    item['status'] = metadata.get('status', 'organization' if metadata else 'unreviewed')
    item['line'] = item.get('line', next((i for i, line in enumerate(text.splitlines(), 1) if line.startswith('# ')), 1))
    item['citation'] = f"{item['path']}:{item['line']}"
    return item


def local_results(query: str, mode: str = 'wiki-first', per_collection_limit: int = 24) -> list[dict[str, Any]]:
    """Recall only: exact lexical matching over allowed files, never an answer."""
    terms = query_terms(qmd_search_query(query))
    if not terms:
        return []
    results = []
    for spec in COLLECTION_SPECS:
        candidates = []
        paths = [ROOT / 'index.md'] if spec.slug == 'index' else sorted(spec.path.rglob('*.md'))
        for path in paths:
            if not path.is_file():
                continue
            text = path.read_text(encoding='utf-8')
            lowered = text.casefold()
            title = next((line[2:].strip() for line in text.splitlines()[:30] if line.startswith('# ')), path.stem)
            combined = (title + ' ' + path.name + ' ' + text).casefold()
            matched = sum(term in combined for term in terms)
            if matched < min(2, len(terms)):
                continue
            lines = text.splitlines()
            best = max(range(len(lines)), key=lambda i: sum(term in ' '.join(lines[max(0, i-1):i+2]).casefold() for term in terms), default=0)
            snippet = ' '.join(lines[max(0, best-1):best+2])
            score = sum(1 + math.log1p(min(lowered.count(term), 20)) for term in terms if term in combined)
            score += 6 * sum(term in title.casefold() for term in terms)
            candidates.append(enrich_result({'path': path.relative_to(ROOT).as_posix(), 'title': title, 'snippet': snippet, 'line': best + 1, 'docid': '', 'score': score, 'layer': spec.layer, 'label': spec.label, 'priority': spec.priority}))
        candidates.sort(key=lambda item: (-item['score'], item['path']))
        results.extend(candidates[:per_collection_limit])
    return order_results(results, mode, query)


def retrieve(query: str, mode: str = 'wiki-first', limit: int = 8, backend: str = 'auto', no_update: bool = False, per_collection_limit: int | None = None) -> tuple[list[dict[str, Any]], str, str | None]:
    count = per_collection_limit or max(limit * 3, 12)
    if backend == 'local':
        return local_results(query, mode, count)[:limit], 'local-lexical', None
    try:
        for spec in COLLECTION_SPECS:
            ensure_collection(spec)
        if not no_update:
            run_qmd(['update'])
        return merged_results(mode, query, count)[:limit], 'qmd-bm25', None
    except (FileNotFoundError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
        if backend == 'qmd':
            raise
        message = short_text(getattr(exc, 'stderr', '') or str(exc), 180)
        return local_results(query, mode, count)[:limit], 'local-lexical', message


def print_results(query: str, mode: str, results: list[dict[str, Any]], limit: int) -> None:
    print(f"# Search: {query}")
    print(f"- Mode: {mode}")
    print(f"- Root: {ROOT}")
    if not results:
        print("- Results: 0")
        return

    print(f"- Results: {min(limit, len(results))}")
    print("")
    for index, item in enumerate(results[:limit], start=1):
        path_text = item["path"] or "(unknown path)"
        title_text = item["title"] or Path(path_text).stem
        print(f"{index}. [{item['layer']}] {path_text}")
        print(f"   标题: {short_text(title_text, 100)}")
        if item["snippet"]:
            print(f"   摘要: {short_text(item['snippet'])}")
        print(f"   层级: {item['label']} | 分数: {item['score']:.3f}")
        print(f"   成熟度: {item.get('status', 'unknown')} | 定位: {item.get('citation', path_text)}")
        if item["docid"]:
            print(f"   qmd: {item['docid']}")


def main() -> int:
    global QMD_TIMEOUT
    args = parse_args()
    if args.limit < 1 or args.timeout <= 0 or (args.per_collection_limit is not None and args.per_collection_limit < 1):
        print('Limits and timeout must be positive.', file=sys.stderr)
        return 1
    QMD_TIMEOUT = args.timeout
    try:
        results, backend, warning = retrieve(args.query, args.mode, args.limit, args.backend, args.no_update, args.per_collection_limit)
        if warning:
            print(f'qmd unavailable; using local lexical recall: {warning}', file=sys.stderr)
        if args.json:
            print(json.dumps({'query': args.query, 'backend': backend, 'mode': args.mode, 'warning': warning, 'results': results}, ensure_ascii=False, indent=2))
        else:
            print_results(args.query, args.mode, results, args.limit)
            print(f'- Backend: {backend}')
        return 0
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        message = stderr or stdout or str(exc)
        print(f"qmd command failed: {message}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Failed to parse qmd JSON output: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
