from __future__ import annotations

import unittest

from scripts.fetch_web_text import validate_extracted_body


class ValidateExtractedBodyTests(unittest.TestCase):
    def test_rejects_upstream_fatal_page(self) -> None:
        with self.assertRaisesRegex(ValueError, "conversion failure"):
            validate_extracted_body(
                "Conversion to HTML had a Fatal error" + " filler" * 1_000,
                min_chars=100,
            )

    def test_rejects_implausibly_short_body(self) -> None:
        with self.assertRaisesRegex(ValueError, "too short"):
            validate_extracted_body("short body", min_chars=100)

    def test_accepts_substantive_body(self) -> None:
        validate_extracted_body("A substantive paragraph. " * 30, min_chars=100)


if __name__ == "__main__":
    unittest.main()
