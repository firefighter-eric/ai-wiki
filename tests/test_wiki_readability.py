from __future__ import annotations

import unittest
from scripts import lint_wiki


GUIDE = "## TL;DR（快速导读）\n\n这项方法把重复的前缀计算保存下来，后续请求可以复用相同内容，但需要检查缓存条件。\n"
ABSTRACT = "## 摘要\n\n方法先检查输入内容与执行条件，再复用已经完成的计算；节省成本取决于实际命中率和管理开销。\n"


class WikiReadabilityTests(unittest.TestCase):
    def check(self, body: str, metadata: dict[str, str] | None = None) -> list[str]:
        checker = lint_wiki.WikiLint.__new__(lint_wiki.WikiLint)
        checker.findings = []
        path = lint_wiki.ROOT / "wiki/summaries/Example.md"
        checker.check_readability(path, metadata or {"type": "summary", "status": "refined"}, body)
        return [finding.code for finding in checker.findings]

    def test_accepts_chinese_guide_and_keeps_auto_status(self) -> None:
        metadata = {"type": "summary", "status": "auto"}
        body = GUIDE + "\n阅读状态：方法细节与实验仍待精读。\n\n" + ABSTRACT
        self.assertEqual(self.check(body, metadata), [])
        self.assertEqual(metadata, {"type": "summary", "status": "auto"})

    def test_rejects_missing_or_misplaced_guide(self) -> None:
        self.assertIn("readability-tldr", self.check(ABSTRACT))
        self.assertIn("readability-tldr", self.check(ABSTRACT + "\n" + GUIDE))

    def test_rejects_english_or_status_only_opening(self) -> None:
        for lead in ("Read this page to learn about caching.", "阅读状态：本页材料已经归档，详细实验和训练条件仍然等待后续精读核证。"):
            body = "## TL;DR（快速导读）\n\n" + lead + "\n\n" + ABSTRACT
            self.assertIn("readability-tldr-content", self.check(body))

    def test_rejects_imported_english_abstract(self) -> None:
        body = GUIDE + "\n阅读状态：待精读。\n\n## 摘要\n\nWe present a method for caching common input prefixes.\n"
        self.assertIn("readability-summary", self.check(body, {"type": "summary", "status": "auto"}))

    def test_ignores_headings_in_nested_fence_example(self) -> None:
        body = "# Example\n\n````markdown\n```\n## Not a real section\n## TL;DR（快速导读）\n```\n````\n\n" + GUIDE + "\n" + ABSTRACT
        self.assertEqual(self.check(body), [])

    def test_rejects_duplicate_guide(self) -> None:
        self.assertIn("readability-tldr-duplicate", self.check(GUIDE + "\n" + GUIDE + "\n" + ABSTRACT))

    def test_building_topic_requires_maturity_note(self) -> None:
        metadata = {"type": "topic", "status": "building"}
        self.assertIn("readability-status", self.check(GUIDE, metadata))
        self.assertEqual(self.check(GUIDE + "\n阅读状态：待建设主题。\n", metadata), [])


if __name__ == "__main__":
    unittest.main()
