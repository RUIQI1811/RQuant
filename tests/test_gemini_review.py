import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import gemini_review


class GeminiReviewerInitializationTest(unittest.TestCase):
    def _config(self, root: Path) -> dict:
        prompt = root / "prompt.md"
        prompt.write_text("test prompt", encoding="utf-8")
        return {
            "prompt_path": prompt,
            "kline_dir": root / "charts",
            "output_dir": root / "review",
            "candidates": root / "candidates.json",
            "model": "test-model",
        }

    def test_dotenv_key_is_used_without_mutating_process_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(Path(temp_dir))
            fake_client = object()
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(
                    gemini_review,
                    "_read_dotenv_value",
                    return_value="test-secret",
                ),
                patch.object(
                    gemini_review.genai,
                    "Client",
                    return_value=fake_client,
                ) as client_factory,
            ):
                reviewer = gemini_review.GeminiReviewer(config)
                self.assertNotIn("GEMINI_API_KEY", os.environ)

        self.assertIs(reviewer.client, fake_client)
        client_factory.assert_called_once_with(api_key="test-secret")

    def test_missing_key_raises_instead_of_exiting_process(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(Path(temp_dir))
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(
                    gemini_review,
                    "_read_dotenv_value",
                    return_value=None,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                    gemini_review.GeminiReviewer(config)


if __name__ == "__main__":
    unittest.main()
