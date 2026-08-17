from __future__ import annotations

import unittest

from scripts.search_wiki import matches_query_text, qmd_search_query, query_terms, subtype_priority


class QueryTermsTests(unittest.TestCase):
    def test_mixed_chinese_query_keeps_product_names(self) -> None:
        terms = query_terms("SGLang 和 vLLM 的技术与架构差异")
        self.assertIn("sglang", terms)
        self.assertIn("vllm", terms)
        self.assertNotIn("sglang 和 vllm 的技术与架构差异", terms)

    def test_chinese_question_words_do_not_poison_term(self) -> None:
        self.assertIn("注意力机制", query_terms("注意力机制有哪些"))

    def test_post_filter_matches_two_distinct_terms(self) -> None:
        result = {
            "path": "wiki/comparisons/SGLang 与 vLLM 架构对比.md",
            "title": "SGLang 与 vLLM 架构对比",
            "snippet": "比较调度与 KV cache。",
        }
        self.assertTrue(matches_query_text("SGLang 和 vLLM 的技术与架构差异", result))

    def test_bm25_query_prefers_multiple_product_names(self) -> None:
        self.assertEqual(
            qmd_search_query("SGLang 和 vLLM 的技术与架构差异"),
            "sglang vllm",
        )

    def test_comparison_intent_prioritizes_comparison_pages(self) -> None:
        query = "SGLang 与 vLLM 架构差异"
        self.assertLess(
            subtype_priority("wiki/comparisons/SGLang 与 vLLM 架构对比.md", query),
            subtype_priority("wiki/concepts/vLLM.md", query),
        )


if __name__ == "__main__":
    unittest.main()
