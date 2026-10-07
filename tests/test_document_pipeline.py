from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from scripts.download_arxiv import download_file, fallback_to_pdf, fetch_html_markdown, normalize_arxiv_id, validate_text_identity
from scripts.extract_pdf_text import extract_pdf_text
from scripts.fetch_web_text import ArticleExtractor
from scripts.source_utils import write_original


class DocumentPipelineTests(unittest.TestCase):
    def test_nested_html_preserves_claim_context_and_locators(self) -> None:
        parser = ArticleExtractor('https://example.org/paper/')
        parser.feed('<html><nav>MENU NOISE</nav><main><article><h2>Method</h2><p>The <strong>new method</strong> uses <math alttext="x^2+y^2">xy</math> and <a href="../code">code</a>.</p><table><tr><th>Model</th><th>Score</th></tr><tr><td>A</td><td>42</td></tr></table></article></main></html>')
        text = parser.render()
        self.assertNotIn('MENU NOISE', text)
        self.assertIn('The **new method** uses $x^2+y^2$', text)
        self.assertIn('[code](https://example.org/code)', text)
        self.assertIn('<a id="source-section-1">', text)
        self.assertIn('| A | 42 |', text)

    def test_original_collision_leaves_bytes_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.html'
            write_original(path, b'original')
            with self.assertRaises(FileExistsError):
                write_original(path, b'changed')
            self.assertEqual(path.read_bytes(), b'original')

    def test_article_header_and_code_indentation_are_not_discarded(self) -> None:
        parser = ArticleExtractor()
        parser.feed('<article><header><p>Important abstract.</p></header><pre>if valid:\n    return result\n</pre></article>')
        text = parser.render()
        self.assertIn('Important abstract.', text)
        self.assertIn('if valid:\n    return result', text)

    def test_model_card_wins_over_collection_recommendation(self) -> None:
        parser = ArticleExtractor()
        parser.feed('<main><div class="model-card-content prose"><h2>Architecture</h2><p>Actual model details.</p></div><article>Recommended collection.</article></main>')
        text = parser.render()
        self.assertIn('Actual model details.', text)
        self.assertNotIn('Recommended collection.', text)
        self.assertIn('source-section-1', text)

    def test_short_recommendation_article_does_not_hide_main_document(self) -> None:
        parser = ArticleExtractor()
        parser.feed('<main><h2>Main method</h2><p>' + 'Substantive method. ' * 40 + '</p><article>Related link.</article></main>')
        self.assertIn('Main method', parser.render())

    def test_github_embedded_file_wins_over_loading_controls(self) -> None:
        parser = ArticleExtractor('https://github.com/org/model/blob/main/LICENSE')
        payload = json.dumps({'payload': {'rawLines': ['Model License', '', '1. Preserve this notice.', '2. A conditional grant.']}})
        parser.feed('<main><h3>Loading error</h3></main><script type="application/json" data-target="react-app.embeddedData">' + payload + '</script>')
        text = parser.render()
        self.assertIn('1. Preserve this notice.\n2. A conditional grant.', text)
        self.assertIn('source-section-1', text)
        self.assertNotIn('Loading error', text)

    def test_long_arxiv_abstract_page_is_rejected_without_replacing_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory) / 'source.html'
            text = Path(directory) / 'source.md'
            html.write_text('<blockquote class="abstract mathjax">' + 'Abstract metadata only. ' * 300 + '</blockquote>')
            text.write_text('Existing full-text extraction')
            with self.assertRaisesRegex(ValueError, 'abstract page'):
                fetch_html_markdown('1706.03762v1', html, text, 'Test', True)
            self.assertEqual(text.read_text(), 'Existing full-text extraction')

    def test_force_never_downloads_over_existing_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.pdf'
            path.write_bytes(b'%PDF-original')
            with patch('scripts.download_arxiv.requests.get') as request:
                download_file('https://arxiv.org/pdf/1706.03762v1', path, True)
                request.assert_not_called()
            self.assertEqual(path.read_bytes(), b'%PDF-original')

    def test_saved_html_rebuild_works_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory) / 'source.html'
            text = Path(directory) / 'source.md'
            original = '<article><h2>Methods</h2><p>' + 'Substantive scientific method. ' * 100 + '</p></article>'
            html.write_text(original)
            with patch('scripts.download_arxiv.requests.get') as request:
                fetch_html_markdown('1706.03762v1', html, text, 'Test', True)
                request.assert_not_called()
            self.assertEqual(html.read_text(), original)
            self.assertIn('Source SHA256', text.read_text())

    def test_arxiv_identity_and_version_are_not_silently_changed(self) -> None:
        self.assertEqual(normalize_arxiv_id('https://arxiv.org/pdf/1706.03762v2.pdf'), '1706.03762v2')
        self.assertEqual(normalize_arxiv_id('arXiv:hep-th/9603067'), 'hep-th/9603067')
        with self.assertRaises(ValueError):
            normalize_arxiv_id('../../other-file')
        with tempfile.TemporaryDirectory() as directory:
            text = Path(directory) / 'source.md'
            text.write_text('- Source URL: https://arxiv.org/html/1706.03762v1\n')
            with self.assertRaises(ValueError):
                validate_text_identity(text, '1706.03762v2')

    def test_pdf_page_anchors_and_image_only_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.pdf'
            with fitz.open() as document:
                document.new_page().insert_text((50, 50), 'First page')
                document.new_page().insert_text((50, 50), 'Second page')
                document.save(path)
            text = extract_pdf_text(path)
            self.assertIn('id="page-1"', text)
            self.assertIn('id="page-2"', text)
            self.assertLess(text.index('First page'), text.index('Second page'))
            blank = Path(directory) / 'blank.pdf'
            with fitz.open() as document:
                document.new_page()
                document.save(blank)
            with self.assertRaisesRegex(ValueError, 'OCR'):
                extract_pdf_text(blank)

    def test_pdf_fallback_retains_version_for_future_identity_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / 'source.pdf'
            text = root / 'source.md'
            with fitz.open() as document:
                page = document.new_page()
                body = '\n'.join(f'Experiment {i}: a substantive result under documented conditions.' for i in range(36))
                page.insert_textbox(page.rect + (35, 35, -35, -35), body, fontsize=9)
                document.save(pdf)
            fallback_to_pdf(pdf, root, text, 'https://arxiv.org/pdf/1706.03762v1', 'Test paper')
            self.assertTrue(text.read_text().startswith('# Test paper'))
            validate_text_identity(text, '1706.03762v1')
            with self.assertRaises(ValueError):
                validate_text_identity(text, '1706.03762v2')


if __name__ == '__main__':
    unittest.main()
