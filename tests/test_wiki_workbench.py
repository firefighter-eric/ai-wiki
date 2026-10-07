from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import lint_wiki, search_wiki
from scripts.source_utils import source_hash
from scripts.wiki_workbench import audit, document_sections, reading_packet


class WorkbenchTests(unittest.TestCase):
    def make_repo(self, root: Path) -> None:
        for folder in ('raw/html', 'raw/pdf', 'raw/text', 'wiki/summaries', 'wiki/topics', 'wiki/concepts', 'wiki/authors', 'wiki/comparisons', 'wiki/timelines'):
            (root / folder).mkdir(parents=True)
        (root / 'raw/html/Example.html').write_text('<article>original</article>')
        (root / 'raw/text/Example.md').write_text('# Example\n\n<a id="method"></a>\n\n## Method\n\nGrounded method details.\n\n## Limitations\n\nA bounded experiment.\n')
        (root / 'wiki/summaries/Example.md').write_text('---\ntype: summary\nstatus: auto\n---\n# Example\n\n## 争议与不确定点\n\nNeeds reading.\n')
        (root / 'wiki/topics/Topic.md').write_text('---\ntype: topic\nstatus: building\n---\n# Topic\n\n## 证据基础\n\n- [Example](../summaries/Example.md)\n')
        (root / 'index.md').write_text('# Index\n')

    def test_queue_follows_topic_evidence_and_never_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repo(root)
            before = (root / 'wiki/summaries/Example.md').read_bytes()
            report = audit(root)
            self.assertEqual(report['refinement_queue_count'], 1)
            self.assertEqual(report['refinement_queue'][0]['reasons'][0]['page'], 'wiki/topics/Topic.md')
            self.assertEqual(report['topic_gaps'][0]['pending_summaries'], ['wiki/summaries/Example.md'])
            self.assertEqual((root / 'wiki/summaries/Example.md').read_bytes(), before)

    def test_reading_packet_is_bounded_and_requires_complete_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repo(root)
            packet = reading_packet('Example', root, heading='Method', max_chars=18)
            self.assertTrue(packet['excerpt']['truncated'])
            self.assertEqual(packet['sections'][1]['anchor'], 'method')
            self.assertEqual(len(packet['excerpt']['content']), 18)
            (root / 'raw/html/Example.html').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                reading_packet('Example', root)

    def test_toc_ignores_code_headings_and_keeps_subsections(self) -> None:
        text = '# Title\n```\n## Not a section\n```\n## Methods\n### Submethod\nText\n## Results\n'
        sections = document_sections(text)
        self.assertNotIn('Not a section', [item['heading'] for item in sections])
        self.assertEqual(next(item for item in sections if item['heading'] == 'Methods')['end_line'], 7)

    def test_toc_keeps_nested_fence_examples_inside_code(self) -> None:
        text = '# Title\n````markdown\n```\n## Example only\n```\n````\n## Actual method\n'
        self.assertEqual([item['heading'] for item in document_sections(text)], ['Title', 'Actual method'])

    def test_qmd_failure_still_returns_local_candidates_with_maturity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repo(root)
            specs = (search_wiki.CollectionSpec('wiki', root / 'wiki', '**/*.md', 'wiki', '知识层', 1),)
            with patch.object(search_wiki, 'ROOT', root), patch.object(search_wiki, 'COLLECTION_SPECS', specs), patch.object(search_wiki, 'ensure_collection', side_effect=RuntimeError('database unavailable')):
                results, backend, warning = search_wiki.retrieve('Example')
            self.assertEqual(backend, 'local-lexical')
            self.assertIn('database unavailable', warning)
            self.assertEqual(results[0]['status'], 'auto')
            self.assertTrue(results[0]['citation'].startswith('wiki/summaries/Example.md:'))

    def test_refined_contract_rejects_unlocatable_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repo(root)
            summary = root / 'wiki/summaries/Example.md'
            with patch.object(lint_wiki, 'ROOT', root), patch.object(lint_wiki, 'WIKI_ROOT', root / 'wiki'):
                checker = lint_wiki.WikiLint()
                checker.check_evidence_contract(summary, {'status': 'refined', 'evidence_schema': '1', 'reviewed': '2026-10-07'}, '## 证据定位\n\n[Claim](../../raw/text/Example.md#missing)\n')
                self.assertIn('evidence-locator', [finding.code for finding in checker.findings])
                checker.findings = []
                checker.check_evidence_contract(summary, {'status': 'refined', 'evidence_schema': '1', 'reviewed': '2026-10-07'}, '## 证据定位\n\n[Claim](../../raw/text/Example.md#method)\n')
                self.assertEqual(checker.findings, [])

    def test_source_hash_detects_changed_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_repo(root)
            original = root / 'raw/html/Example.html'
            digest = source_hash(original)
            text = root / 'raw/text/Example.md'
            text.write_text(f'- Source HTML: `raw/html/Example.html`\n- Source SHA256: `{digest}`\n\nBody\n')
            original.write_text('changed outside the pipeline')
            with patch.object(lint_wiki, 'ROOT', root), patch.object(lint_wiki, 'WIKI_ROOT', root / 'wiki'):
                checker = lint_wiki.WikiLint()
                checker.check_raw_text_quality()
            self.assertIn('source-hash', [finding.code for finding in checker.findings])


if __name__ == '__main__':
    unittest.main()
