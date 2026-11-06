"""Exercise the actual decoder grammar on CPU, without model weights or CUDA."""

import importlib.util
import json
import unittest

from wild.selection import helpers, judge


@unittest.skipUnless(importlib.util.find_spec("xgrammar") and importlib.util.find_spec("vllm"),
                     "Requires the production grammar libraries; run on CPU in the pod environment")
class StructuredGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import xgrammar as xgr
        cls.xgr = xgr
        params = judge.classification_sampling_params(256).structured_outputs
        tokenizer = xgr.TokenizerInfo([chr(i) for i in range(128)], stop_token_ids=[0])
        cls.grammar = xgr.GrammarCompiler(tokenizer, max_threads=2).compile_json_schema(
            params.json, any_whitespace=not params.disable_any_whitespace)

    def accepts(self, value):
        matcher = self.xgr.GrammarMatcher(self.grammar, terminate_without_stop_token=True)
        return matcher.accept_string(value) and matcher.is_completed()

    def test_valid_labels_include_required_rationale(self):
        for domain in helpers.LABELS:
            for topic in (None, *helpers.SUBTOPICS.get(domain, ())):
                with self.subTest(domain=domain, topic=topic):
                    value = json.dumps({"domain": domain, "subtopic": topic, "rationale": "Reviewed user task."})
                    self.assertTrue(self.accepts(value))
                    self.assertEqual(helpers.parse_label(value)["status"], "classified")

    def test_failed_smoke_shapes_are_rejected_by_decoder(self):
        for value in [
            '{"domain": "coding", "subtopic": "implementation"}',
            '{"domain": "writing", "subtopic": "creative_fiction" }',
            '{"domain": "coding", "subtopic": "implementation"' + '\n' * 100,
            '{"domain": "coding", "subtopic": "implementation", "rationale": "Code.", "extra": 1}',
        ]:
            with self.subTest(value=value[:100]):
                matcher = self.xgr.GrammarMatcher(self.grammar, terminate_without_stop_token=True)
                self.assertFalse(matcher.accept_string(value))


if __name__ == "__main__":
    unittest.main()
