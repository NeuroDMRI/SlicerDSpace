"""Keep user-facing Agent prompts in the READMEs and homepage."""

from html import unescape
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / ".agents/skills/slicerdspace"
URL = "https://github.com/NeuroDMRI/SlicerDSpace/blob/main/.agents/skills/slicerdspace/SKILL.md"


def blocks(text):
    return re.findall(r"```text\n(.*?)\n```", text, re.S)


def normalized(text):
    return re.sub(r"\s+", " ", text).strip()


def readme_prompts(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    section = text.split("## Agent Skills\n", 1)[1].split("\n---\n", 1)[0]
    return [item for item in blocks(section) if URL in item]


class AgentPromptDocsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = readme_prompts("README.md")

    def test_skill_does_not_embed_or_link_user_prompt_examples(self):
        self.assertFalse((SKILL / "references/prompts.md").exists())
        instructions = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        self.assertNotIn("references/prompts.md", instructions)
        for example in self.examples:
            self.assertNotIn(normalized(example), normalized(instructions))

    def test_skill_identity_and_prompt_entrypoint_match_the_directory(self):
        instructions = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("name: " + SKILL.name + "\n", instructions)
        metadata = (SKILL / "agents/openai.yaml").read_text(encoding="utf-8")
        self.assertEqual(set(re.findall(r"\$([\w-]+)", metadata)), {SKILL.name})
        self.assertTrue(URL.endswith("/" + SKILL.name + "/SKILL.md"))
        for required in ("references/usage.md", "references/shell_templates.md", "scripts/enable_in_slicer.py"):
            self.assertTrue((SKILL / required).is_file(), required)

    def test_readmes_preserve_the_complete_copyable_prompts(self):
        for name in ("README.md", "README.zh-CN.md"):
            prompts = readme_prompts(name)
            self.assertEqual(len(prompts), 2, name)
            for prompt in prompts:
                self.assertIn(URL, prompt)
            self.assertNotIn("/path/to/subject", prompts[0])
            self.assertIn("/path/to/subject", prompts[1])
            self.assertIn("/path/to/results", prompts[1])

    def test_homepage_exposes_separate_complete_copy_targets(self):
        source = (ROOT / "docs/site/index.html").read_text(encoding="utf-8")
        for index, identity in enumerate(("agent-install-prompt", "agent-analysis-prompt")):
            matches = re.findall(r'<code id="' + identity + r'">(.*?)</code>', source, re.S)
            self.assertEqual(len(matches), 1, identity)
            self.assertEqual(normalized(unescape(matches[0])), normalized(self.examples[index]))


    def test_analysis_example_has_input_output_and_plan_confirmation(self):
        install, analysis = self.examples[:2]
        self.assertNotIn("/path/to/subject", install)
        self.assertNotIn("model-storage locations", install)
        self.assertIn("/path/to/subject", analysis)
        self.assertIn("/path/to/results", analysis)
        self.assertIn("DDParcel", analysis)
        self.assertIn("CSD/ACT tractography", analysis)
        self.assertIn("structural connectome construction", analysis)
        self.assertIn("wait for confirmation before running", analysis)
        self.assertIn("QC findings", analysis)



if __name__ == "__main__":
    unittest.main()
