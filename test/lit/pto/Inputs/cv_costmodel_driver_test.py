# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Native driver routing regressions; deliberately run without a CANN install."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

from ptoas._loader import ensure_core

FLAT = """module attributes {pto.target_arch = "a5"} {
  func.func @driver_probe() { return }
}
"""


def partitioned(backends):
    children = [f'module attributes {{pto.backend = "{backend}"}} {{ '
                f'func.func @child_{i}() {{ return }} }}'
                for i, backend in enumerate(backends)]
    return 'module attributes {pto.target_arch = "a5"} {\n' + '\n'.join(children) + '\n}'


class DriverTest(unittest.TestCase):
    def run_compiler(self, source, *flags):
        with tempfile.TemporaryDirectory(prefix="cv-driver-test-") as tmp:
            root = Path(tmp)
            src, out = root / "input.pto", root / "output"
            src.write_text(source)
            environment = dict(os.environ)
            for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
                environment.pop(name, None)
            environment["PYTHONPATH"] = str(Path(ensure_core().__file__).resolve().parent.parent)
            bootstrap = ("import sys; "
                         "sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in repr(f).lower()]; "
                         "from pathlib import Path; from ptoas import _cli; "
                         "raise SystemExit(_cli.launch(sys.argv[1:], wrapper=Path(_cli.__file__)))")
            command = [sys.executable, "-c", bootstrap, "--pto-arch=a5", *flags,
                       str(src), "-o", str(out)]
            result = subprocess.run(command, env=environment, text=True,
                                    capture_output=True, timeout=60, check=False)
            return result, out.read_text() if out.exists() else None

    def test_flat_checkpoint_is_ir_without_toolchain(self):
        for backend in ("emitc", "vpto"):
            with self.subTest(backend=backend):
                result, text = self.run_compiler(FLAT, "--emit-cv-costmodel-ir",
                                                 f"--pto-backend={backend}")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(text.startswith('"builtin.module"'), text)
                self.assertIn("pto.costmodel.checkpoint_version", text)
                self.assertNotIn("CANN toolchain", result.stderr)
                self.assertNotIn("ASCEND_HOME_PATH", result.stderr)

    def test_partitioned_export_preserves_tree_without_toolchain(self):
        for backends in (("emitc",), ("emitc", "emitc", "emitc"), ("emitc", "vpto")):
            for override in ((), ("--pto-backend=emitc",), ("--pto-backend=vpto",),
                             ("--emit-pto-ir",)):
                with self.subTest(backends=backends, override=override):
                    result, text = self.run_compiler(partitioned(backends),
                                                     "--emit-cv-costmodel-ir", *override)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(text.count('"builtin.module"'), 1 + len(backends))
                    for i, backend in enumerate(backends):
                        self.assertIn('sym_name = "child_' + str(i) + '"', text)
                        self.assertIn('pto.backend = "' + backend + '"', text)
                    # Outer container must not gain a backend that overrides its children.
                    self.assertNotIn("pto.backend", text.splitlines()[-1])
                    self.assertNotIn("CANN toolchain", result.stderr)
                    self.assertNotIn("ASCEND_HOME_PATH", result.stderr)
                    self.assertNotIn("fatobj compilation failed", result.stderr)

    def test_flat_default_still_emits_cpp(self):
        result, text = self.run_compiler(FLAT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("driver_probe", text)
        self.assertNotIn('"builtin.module"', text)
        self.assertNotIn("pto.costmodel.checkpoint_version", text)

    def test_partitioned_default_still_requests_toolchain(self):
        result, text = self.run_compiler(partitioned(("emitc", "emitc", "emitc")))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CANN toolchain", result.stderr)
        self.assertNotIn("cost model checkpoint", result.stderr)
        self.assertIsNone(text)


if __name__ == "__main__":
    unittest.main()
