#!/usr/bin/env python3
"""Known-page retrieval regression; this does not measure answer truthfulness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .search_wiki import ROOT, retrieve
except ImportError:
    from search_wiki import ROOT, retrieve


def evaluate(cases: list[dict], k: int = 5, backend: str = 'local') -> dict:
    rows = []
    for case in cases:
        for path in case['expected']:
            if not (ROOT / path).is_file():
                raise ValueError(f'Evaluation target no longer exists: {path}')
        hits, used_backend, warning = retrieve(case['query'], limit=k, backend=backend)
        paths = [item['path'] for item in hits]
        expected = set(case['expected'])
        ranks = [index for index, path in enumerate(paths, 1) if path in expected]
        recall = len(expected.intersection(paths)) / len(expected) if expected else None
        passed = not paths if case.get('expect_empty') else recall == 1.0
        rows.append({'query': case['query'], 'passed': passed, 'recall': recall, 'reciprocal_rank': 1 / min(ranks) if ranks else 0, 'backend': used_backend, 'warning': warning, 'paths': paths})
    positives = [row for row in rows if row['recall'] is not None]
    return {'k': k, 'cases': len(rows), 'passed': sum(row['passed'] for row in rows), 'recall_at_k': sum(row['recall'] for row in positives) / len(positives) if positives else 0, 'mrr': sum(row['reciprocal_rank'] for row in positives) / len(positives) if positives else 0, 'scope': 'Small known-page candidate-recall fixture; excludes generation, entailment, semantic paraphrases and overall corpus coverage.', 'results': rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=ROOT / 'tests' / 'fixtures' / 'retrieval_cases.json')
    parser.add_argument('--backend', choices=('local', 'auto', 'qmd'), default='local')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.k < 1:
        parser.error('k must be positive.')
    report = evaluate(json.loads(args.cases.read_text(encoding='utf-8')), args.k, args.backend)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Retrieval: {report['passed']}/{report['cases']} cases; recall@{args.k}={report['recall_at_k']:.3f}; MRR={report['mrr']:.3f}")
        for row in report['results']:
            if not row['passed']:
                print(f"FAIL: {row['query']} -> {row['paths']}")
        print(report['scope'])
    return 0 if report['passed'] == report['cases'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
