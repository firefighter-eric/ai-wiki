from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.scan_secrets import scan_file


class SecretScannerTests(unittest.TestCase):
    def test_document_ui_identifiers_are_not_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "raw" / "html" / "document.html"
            path.parent.mkdir(parents=True)
            path.write_text(
                '<button id="ask-assistant-code-python-long-example" '
                'aria-label="Ask Assistant"></button>\n'
                '<a id="does-a-failed-task-still-consume-credits">FAQ</a>'
            )
            with patch("scripts.scan_secrets.ROOT", root):
                self.assertEqual(scan_file(path), [])

    def test_standalone_credentials_in_config_and_originals_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_key = "sk-" + "A" * 36
            for relative, severity in (
                ("config.toml", "error"),
                ("raw/html/document.html", "warning"),
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('api_key="' + fake_key + '"')
                with patch("scripts.scan_secrets.ROOT", root):
                    matches = scan_file(path)
                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0].severity, severity)
                self.assertEqual(matches[0].kind, "openai-style-key")


if __name__ == "__main__":
    unittest.main()
