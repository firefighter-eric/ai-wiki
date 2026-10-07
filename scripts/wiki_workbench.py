#!/usr/bin/env python3
"""Read-only editorial queue and bounded reading packets for the persistent wiki."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .lint_wiki import local_target, markdown_links, parse_frontmatter, section
    from .source_utils import source_hash
except ImportError:
    from lint_wiki import local_target, markdown_links, parse_frontmatter, section
    from source_utils import source_hash

ROOT = Path(__file__).resolve().parent.parent


def collect_pages(root: Path = ROOT) -> dict[str, dict]:
    pages = {}
    for path in sorted((root / 'wiki').rglob('*.md')):
        metadata, body = parse_frontmatter(path.read_text(encoding='utf-8'))
        pages[path.relative_to(root).as_posix()] = {'metadata': metadata, 'body': body, 'path': path}
    return pages


def targets(path: str, text: str) -> set[str]:
    # local_target uses a repo root; use a synthetic relative path under that root.
    return {target for destination in markdown_links(text) if (target := local_target(ROOT / path, destination)) is not None}


def source_paths(stem: str, root: Path = ROOT) -> list[Path]:
    return [path for path in (root / 'raw' / 'html' / (stem + '.html'), root / 'raw' / 'pdf' / (stem + '.pdf')) if path.is_file()]


def source_identity(text: str) -> str | None:
    match = re.search(r'^- Source URL:\s*https?://(?:arxiv\.org/(?:abs|html|pdf)/|ar5iv\.labs\.arxiv\.org/html/)((?:\d{4}\.\d{4,5}|[a-z.-]+/\d{7})(?:v\d+)?)', text, flags=re.MULTILINE | re.IGNORECASE)
    return 'arxiv:' + re.sub(r'v\d+$', '', match.group(1)) if match else None


def audit(root: Path = ROOT, limit: int = 12, focus: str = '') -> dict:
    pages = collect_pages(root)
    inbound = defaultdict(set)
    evidence_users = defaultdict(list)
    for path, page in pages.items():
        for target in targets(path, page['body']):
            if target != path and target in pages:
                inbound[target].add(path)
        for heading in ('证据基础', '来源支持'):
            for target in targets(path, section(page['body'], heading)):
                if target.startswith('wiki/summaries/'):
                    evidence_users[target].append(path)
    queue = []
    topic_gaps = []
    for path, page in pages.items():
        meta = page['metadata']
        if meta.get('type') == 'topic':
            deps = sorted(targets(path, section(page['body'], '证据基础')))
            pending = [dep for dep in deps if dep in pages and pages[dep]['metadata'].get('status') != 'refined']
            if pending:
                topic_gaps.append({'path': path, 'status': meta.get('status'), 'pending_summaries': pending})
        if meta.get('type') != 'summary' or meta.get('status') != 'auto':
            continue
        if focus and focus.casefold() not in (path + ' '.join(inbound[path])).casefold():
            continue
        users = sorted(set(evidence_users[path]))
        score = 0
        reasons = []
        for user in users:
            status = pages.get(user, {}).get('metadata', {}).get('status')
            weight = 12 if status == 'formal' else 6 if status == 'building' else 3
            score += weight
            reasons.append({'page': user, 'weight': weight, 'reason': '正式 topic 证据缺口' if status == 'formal' else '待建设 topic 的证据' if status == 'building' else '组织页面的来源支持'})
        score += min(len(inbound[path]), 10)
        text_path = root / 'raw' / 'text' / (page['path'].stem + '.md')
        queue.append({'path': path, 'priority': score, 'reasons': reasons, 'inbound_count': len(inbound[path]), 'source_chain_ready': bool(source_paths(page['path'].stem, root)) and text_path.is_file()})
    queue.sort(key=lambda item: (not item['source_chain_ready'], -item['priority'], item['path']))
    counts = Counter((page['metadata'].get('type', 'unknown'), page['metadata'].get('status', 'organization')) for page in pages.values())
    identities = defaultdict(list)
    for path in sorted((root / 'raw' / 'text').glob('*.md')):
        identity = source_identity(path.read_text(encoding='utf-8')[:4000])
        if identity:
            identities[identity].append(path.relative_to(root).as_posix())
    orphans = sorted(path for path in pages if path.startswith(('wiki/summaries/', 'wiki/authors/')) and not inbound[path])
    return {'counts': {f'{kind}:{status}': count for (kind, status), count in sorted(counts.items())}, 'refined_without_evidence_schema': sum(page['metadata'].get('status') == 'refined' and page['metadata'].get('evidence_schema') != '1' for page in pages.values()), 'orphan_count': len(orphans), 'orphans': orphans[:limit], 'refinement_queue_count': len(queue), 'refinement_queue': queue[:limit], 'topic_gaps': topic_gaps, 'same_arxiv_id_candidates': {key: value for key, value in identities.items() if len(value) > 1}, 'identity_note': '仅从来源 URL 元数据识别 arXiv ID；相同 ID 的不同版本需要比较，不能自动合并。', 'queue_note': '优先级是透明的链接/证据依赖启发式，不是论文质量评分；状态不会被自动提升。'}


def document_sections(text: str) -> list[dict]:
    lines = text.splitlines()
    result = []
    fence = None
    for i, line in enumerate(lines):
        marker = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if marker:
            run, suffix = marker.groups()
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence) and not suffix.strip():
                fence = None
            continue
        match = re.match(r'^(#{1,6})\s+(.+)', line)
        if not match or fence:
            continue
        anchor = None
        for previous in lines[max(0, i - 3):i]:
            found = re.search(r'<a id="([^"]+)"', previous)
            if found:
                anchor = found.group(1)
        result.append({'heading': match.group(2), 'level': len(match.group(1)), 'line': i + 1, 'anchor': anchor})
    for i, item in enumerate(result):
        item['end_line'] = next((following['line'] - 1 for following in result[i + 1:] if following['level'] <= item['level']), len(lines))
    return result


def reading_packet(source: str, root: Path = ROOT, question: str = '', heading: str = '', max_chars: int = 8000) -> dict:
    stem = Path(source).stem if Path(source).suffix == '.md' else source
    if '/' in stem or '\\' in stem or stem in {'.', '..'}:
        # Accept a canonical in-repo file path, but never interpret it as an external read.
        candidate = (root / source).resolve()
        if not candidate.is_relative_to(root.resolve()) or not candidate.is_file() or candidate.suffix != '.md':
            raise ValueError('Supply an existing summary/raw-text filename, stem or repository-relative .md path.')
        stem = candidate.stem
    summary = root / 'wiki' / 'summaries' / (stem + '.md')
    text_path = root / 'raw' / 'text' / (stem + '.md')
    originals = source_paths(stem, root)
    if not summary.is_file() or not text_path.is_file() or not originals:
        raise ValueError(f'Incomplete source chain for {stem}; restore raw HTML/PDF and text before refinement.')
    text = text_path.read_text(encoding='utf-8')
    metadata, summary_body = parse_frontmatter(summary.read_text(encoding='utf-8'))
    sections = document_sections(text)
    lines = text.splitlines()
    matching = [item for item in sections if heading and heading.casefold() in item['heading'].casefold()]
    selected = next((item for item in matching if item['anchor']), matching[0] if matching else None)
    if heading and selected is None:
        raise ValueError(f'Section not found: {heading}. Inspect the table of contents first.')
    start = selected['line'] if selected else 1
    end = selected['end_line'] if selected else len(lines)
    content = '\n'.join(lines[start - 1:end])
    excerpt = content[:max_chars]
    related = sorted(path for path, page in collect_pages(root).items() if 'wiki/summaries/' + summary.name in targets(path, page['body']))
    return {'summary': summary.relative_to(root).as_posix(), 'status': metadata.get('status'), 'question': question or '明确本篇要支撑/修正的研究问题后再精读。', 'source_identity': source_identity(text), 'originals': [{'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size, 'sha256': source_hash(path)} for path in originals], 'text': text_path.relative_to(root).as_posix(), 'sections': sections, 'excerpt': {'start_line': start, 'end_line': start + excerpt.count('\n'), 'truncated': len(content) > max_chars, 'content': excerpt}, 'existing_uncertainty': section(summary_body, '争议与不确定点').strip(), 'affected_pages': related, 'review_contract': ['先读取摘要与目录，再按问题读取方法、实验、消融和局限；截断片段不代表已读全文。', '每个核心主张记录原文章节/PDF 页码、实验设置与成立条件；图表和公式必要时回看原始文件。', '区分作者报告、独立验证与跨来源综合；找出新来源加强/削弱/修正哪些已有说法。', '实际核证后再把 auto 提升为 refined；写回受影响页面、index 与追加 log，并执行 lint/回归评测。']}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    audit_parser = commands.add_parser('audit', help='Maturity, refinement priorities and navigation gaps.')
    audit_parser.add_argument('--limit', type=int, default=12)
    audit_parser.add_argument('--focus', default='', help='Filter queue by summary or linked topic name.')
    audit_parser.add_argument('--json', action='store_true')
    plan_parser = commands.add_parser('plan', help='Prepare a bounded packet from an existing source chain.')
    plan_parser.add_argument('source')
    plan_parser.add_argument('--question', default='')
    plan_parser.add_argument('--section', default='')
    plan_parser.add_argument('--max-chars', type=int, default=8000)
    plan_parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if getattr(args, 'limit', 1) < 1 or getattr(args, 'max_chars', 1) < 1:
        parser.error('Limits must be positive.')
    try:
        report = audit(limit=args.limit, focus=args.focus) if args.command == 'audit' else reading_packet(args.source, question=args.question, heading=args.section, max_chars=args.max_chars)
    except (ValueError, OSError) as exc:
        parser.exit(1, str(exc) + '\n')
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == 'audit':
        print('Wiki workbench（只读）')
        print(json.dumps(report['counts'], ensure_ascii=False))
        print(f"待精读：{report['refinement_queue_count']}；缺入链：{report['orphan_count']}")
        for item in report['refinement_queue']:
            print(f"- {item['priority']:>3} | {item['path']} | 来源链：{item['source_chain_ready']}")
        print(report['queue_note'])
    else:
        print(f"# Reading packet: {report['summary']}")
        print(f"- 成熟度：{report['status']} | 问题：{report['question']}")
        for item in report['sections']:
            print(f"- L{item['line']}–L{item['end_line']} {item['heading']}" + (f" #{item['anchor']}" if item['anchor'] else ''))
        print(f"\n摘录：{report['text']}:{report['excerpt']['start_line']}（截断：{report['excerpt']['truncated']}）\n")
        print(report['excerpt']['content'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
