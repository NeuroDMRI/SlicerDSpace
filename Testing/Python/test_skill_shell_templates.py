"""Exercise generated algorithm templates without running scientific backends."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from scripts import build_skill_shell_templates as generator
from scripts.extension_runtime.runtime_resources import runtime_files
from shared.algorithm_selection import BUNDLES, TRACKING
from shared.brain_mask import BRAIN_MASK_BACKENDS
from shared.run_subject_pipeline_cli import build_parser, resolve_analysis_options
from shared.user_config import explicit_cli_destinations


ROOT = Path(__file__).resolve().parents[2]
SHELL = ROOT / generator.SKILL / "assets/shell"


def parse(arguments):
    parser = build_parser()
    parser.allow_abbrev = False
    args = parser.parse_args(arguments)
    targets = resolve_analysis_options(args, explicit_dests=explicit_cli_destinations(parser, arguments))
    return args, targets


class SkillShellTemplateTests(unittest.TestCase):
    def test_generated_scripts_and_catalogue_match_the_source_recipes(self):
        outputs = generator.generated_files()
        scripts = {path for path in outputs if path.suffix == ".sh"}
        self.assertEqual(len(scripts), 34)
        self.assertEqual({path.relative_to(ROOT) for path in SHELL.glob("*.sh")}, scripts)
        for path, expected in outputs.items():
            with self.subTest(path=str(path)):
                self.assertFalse((ROOT / path).is_symlink())
                self.assertEqual((ROOT / path).read_text(encoding="utf-8"), expected)
        catalogue = ROOT / generator.SKILL / "references/shell_templates.md"
        for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", catalogue.read_text(encoding="utf-8")):
            self.assertTrue((catalogue.parent / target).is_file(), target)

    def test_bilingual_source_drift_is_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in (generator.GUIDE, generator.CHINESE_GUIDE):
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / path, root / path)
            path = root / generator.CHINESE_GUIDE
            path.write_text(path.read_text(encoding="utf-8").replace("--micro-fit-method ls", "--micro-fit-method wls"),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recipes differ"):
                generator.generated_files(root)

    def test_windows_checkout_line_endings_do_not_change_generated_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (generator.GUIDE, generator.CHINESE_GUIDE):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                text = (ROOT / relative).read_text(encoding="utf-8")
                path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
            self.assertEqual(generator.generated_files(root), generator.generated_files())

    def test_templates_cover_the_current_algorithm_selectors(self):
        selections = []
        for name, code in generator.subject_recipes().items():
            tokens = generator.tokens(code)
            args, _ = parse(["--subject-root", "/inputs/example", *tokens[2:]])
            selections.append((name, args))
        self.assertEqual({args.brain_mask_backend for name, args in selections if name.startswith("mask-")},
                         set(BRAIN_MASK_BACKENDS) - {"provided"})
        choices = {action.dest: set(action.choices) for action in build_parser()._actions if action.choices}
        for destination, prefix, excluded in (("micro_fit_method", "micro-dti", set()),
                                               ("dki_fit_method", "micro-", set()),
                                               ("connectome_backend", "connectome-", set())):
            actual = {getattr(args, destination) for name, args in selections if name.startswith(prefix)}
            self.assertEqual(actual, choices[destination] - excluded, destination)
        for destination, prefix, expected in (("tractography_backend", "tract-", TRACKING),
                                                ("bundle_backend", "bundle-", BUNDLES)):
            actual = {value for name, args in selections if name.startswith(prefix)
                      for value in getattr(args, destination).split(",")}
            self.assertEqual(actual, set(expected), destination)

    def test_template_assets_do_not_bloat_the_extension_runtime(self):
        selected = runtime_files(ROOT)
        self.assertNotIn("scripts/build_skill_shell_templates.py", selected)
        self.assertFalse(any(path.startswith(".agents/") for path in selected))


@unittest.skipUnless(shutil.which("bash") and os.name != "nt", "POSIX Bash fixture required")
class SkillShellExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="Skill shell templates ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.launcher = self.root / "fake Slicer"
        self.log = self.root / "argv.jsonl"
        self.launcher.write_text(
            "#!" + sys.executable + "\nimport json,os,sys\n"
            "with open(os.environ['SKILL_TEST_LOG'],'a') as f:\n"
            "    f.write(json.dumps(sys.argv[1:])+'\\n')\n"
            "sys.exit(int(os.environ.get('SKILL_TEST_EXIT','0')))\n", encoding="utf-8")
        self.launcher.chmod(0o700)
        self.env = dict(os.environ, SLICER=str(self.launcher), REPO=str(self.root / "runtime tree"),
                        SUBJECT=str(self.root / "subject folder"), SUBJECT_ID="example",
                        OUTPUTS=str(self.root / "analysis folder"), ASSETS=str(self.root / "model assets"),
                        DEPENDENCY_CONFIG=str(self.root / "dependency config.json"), SKILL_TEST_LOG=str(self.log))

    def calls(self):
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()] if self.log.exists() else []

    def run_template(self, name, *arguments, env=None):
        return subprocess.run(["bash", str(SHELL / (name + ".sh")), *arguments],
                              env=self.env if env is None else env, capture_output=True, text=True)

    def test_all_algorithm_templates_default_to_plan_and_preserve_scientific_arguments(self):
        for name, code in generator.subject_recipes().items():
            tokens = generator.tokens(code)
            run_name = tokens[1]
            with self.subTest(recipe=name):
                result = self.run_template(run_name)
                self.assertEqual(result.returncode, 0, result.stderr)
                call = self.calls()[-1]
                self.assertEqual(call[:3], ["--launch", "PythonSlicer",
                    str(Path(self.env["REPO"]) / "shared/run_subject_pipeline_cli.py")])
                args, targets = parse(call[3:])
                self.assertTrue(targets)
                self.assertTrue(args.plan)
                self.assertFalse(args.execute or args.doctor)
                self.assertTrue(args.no_user_config and args.strict_environment)
                self.assertFalse(args.skip_environment_check or args.allow_anatomy_space_mismatch or args.no_resume)
                self.assertEqual(args.subject_root, Path(self.env["SUBJECT"]))
                self.assertEqual(args.output_root, Path(self.env["OUTPUTS"]) / run_name)
                self.assertEqual(args.model_assets_dir, Path(self.env["ASSETS"]))
                self.assertEqual(args.dependency_config, Path(self.env["DEPENDENCY_CONFIG"]))
                route = [self.env["SLICER"] if value == "$SLICER" else value for value in tokens[2:]]
                self.assertEqual(call[-len(route):], route)
        self.assertEqual(len(self.calls()), 34)

    def test_modes_change_only_the_execution_flag(self):
        for mode in ("doctor", "plan", "execute"):
            result = self.run_template("ddparcel", mode)
            self.assertEqual(result.returncode, 0, result.stderr)
            args, _ = parse(self.calls()[-1][3:])
            self.assertTrue(getattr(args, mode))
        normalized = [[value for value in call if value not in {"--plan", "--doctor", "--execute"}]
                      for call in self.calls()]
        self.assertEqual(normalized, [normalized[0]] * 3)

    def test_dependency_config_is_optional_and_path_values_are_not_shell_code(self):
        environment = dict(self.env, SUBJECT=str(self.root / "subject with $literal (scan)"))
        environment.pop("DEPENDENCY_CONFIG")
        result = self.run_template("ddparcel", env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        call = self.calls()[-1]
        self.assertNotIn("--dependency-config", call)
        self.assertEqual(call[call.index("--subject-root") + 1], environment["SUBJECT"])

    def test_help_syntax_and_invalid_modes_do_not_launch_slicer(self):
        for script in SHELL.glob("*.sh"):
            result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, script.name + ": " + result.stderr)
        result = self.run_template("ddparcel", "--help", env={"PATH": os.environ["PATH"]})
        self.assertEqual(result.returncode, 0, result.stderr)
        for arguments in (("--execute",), ("invalid",), ("plan", "--execute")):
            result = self.run_template("ddparcel", *arguments)
            self.assertNotEqual(result.returncode, 0)
        environment = dict(self.env)
        environment.pop("SUBJECT")
        result = self.run_template("ddparcel", env=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_execute_rejects_both_environment_and_embedded_example_paths(self):
        result = self.run_template("ddparcel", "execute", env=dict(self.env, ASSETS="/path/to/model-assets"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Replace the example path", result.stderr)
        for name in ("anatomy_post", "mkcurve_matlab"):
            result = self.run_template(name, "execute")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Replace the example path", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_launcher_failure_is_not_reported_as_success(self):
        result = self.run_template("ddparcel", "doctor", env=dict(self.env, SKILL_TEST_EXIT="17"))
        self.assertEqual(result.returncode, 17)
        self.assertEqual(len(self.calls()), 1)


if __name__ == "__main__":
    unittest.main()
