from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.migrate_wiki_metadata import parse_frontmatter, summary_status


class SummaryStatusTests(unittest.TestCase):
    def test_pending_source_is_auto(self) -> None:
        body = "# Test\n\n- 状态：已抽取全文，待精读\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.md"
            path.write_text(body, encoding="utf-8")
            self.assertEqual(summary_status(path, body), "auto")

    def test_curated_mixed_heading_is_refined(self) -> None:
        body = "# Test\n\n- 状态：已基于 arXiv HTML 整理\n\n## 自动抽取摘要或人工摘要\n\n人工整理。\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.md"
            path.write_text(body, encoding="utf-8")
            self.assertEqual(summary_status(path, body), "refined")

    def test_frontmatter_parser(self) -> None:
        metadata, body = parse_frontmatter("---\ntype: topic\nstatus: formal\n---\n# T\n")
        self.assertEqual(metadata, {"type": "topic", "status": "formal"})
        self.assertEqual(body, "# T\n")

    def test_explicit_auto_status_is_not_promoted(self) -> None:
        text = "---\ntype: summary\nstatus: auto\n---\n# T\n\n## 摘要\n\n待整理。\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.md"
            path.write_text(text, encoding="utf-8")
            _, body = parse_frontmatter(text)
            self.assertEqual(summary_status(path, body), "auto")


if __name__ == "__main__":
    unittest.main()
