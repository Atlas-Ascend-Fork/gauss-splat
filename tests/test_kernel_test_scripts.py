# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


def find_bash():
    bash = shutil.which('bash')
    if bash:
        return bash
    git = shutil.which('git')
    if git:
        candidate = Path(git).resolve().parent.parent / 'bin' / 'bash.exe'
        if candidate.is_file():
            return str(candidate)
    return None


@unittest.skipUnless(find_bash(), 'Bash is required')
class TestKernelTestScripts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='gauss-splat-script-tests-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.scripts = self.root / 'tests' / 'kernel_tests'
        self.scripts.mkdir(parents=True)
        for name in ('build_kernel_tests.sh', 'run_kernel_tests.sh'):
            source = Path(__file__).parent / 'kernel_tests' / name
            with (self.scripts / name).open('w', encoding='utf-8', newline='\n') as target:
                target.write(source.read_text(encoding='utf-8'))
        self.fake_bin = self.root / 'bin'
        self.fake_bin.mkdir()
        self.write_command('python', '#!/bin/bash\nprintf "%s\\n" "$TEST_PACKAGE_ROOT"\n')
        self.write_command('uname', '#!/bin/bash\necho x86_64\n')
        self.write_command('g++', r'''#!/bin/bash
if [ "${TEST_COMPILE_FAIL:-0}" = 1 ]; then
    echo "simulated compiler failure" >&2
    exit 1
fi
previous=""
for argument in "$@"; do
    if [ "$previous" = -o ]; then output="$argument"; fi
    previous="$argument"
done
printf '%s\n' '#!/bin/bash' 'echo stdout-marker' 'echo stderr-marker >&2' 'exit "${TEST_RUNTIME_EXIT:-0}"' > "$output"
chmod +x "$output"
''')
        cann = self.root / 'cann'
        (cann / 'include').mkdir(parents=True)
        package = self.root / 'package'
        (package / 'packages' / 'vendors' / 'customize' / 'op_api' / 'include').mkdir(parents=True)
        self.env = dict(os.environ, PATH=str(self.fake_bin) + os.pathsep + os.environ['PATH'],
                        ASCEND_TOOLKIT_HOME=cann.as_posix(), TEST_PACKAGE_ROOT=package.as_posix(),
                        TEST_COMPILE_FAIL='0', TEST_RUNTIME_EXIT='0')
        for name in ('test_one.cpp', 'test_two.cpp'):
            (self.scripts / name).touch()

    def write_command(self, name, source):
        path = self.fake_bin / name
        with path.open('w', encoding='utf-8', newline='\n') as target:
            target.write(source)
        path.chmod(0o755)

    def run_script(self, name):
        return subprocess.run([find_bash(), (self.scripts / name).as_posix()],
                              env=self.env, capture_output=True, text=True, timeout=30)

    def test_success_and_both_output_streams(self):
        build = self.run_script('build_kernel_tests.sh')
        self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
        run = self.run_script('run_kernel_tests.sh')
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        log = (self.scripts / 'build_out' / 'test_one_run.log').read_text()
        self.assertIn('stdout-marker', log)
        self.assertIn('stderr-marker', log)
        self.assertIn('Tests: 2, failed: 0', run.stdout)

    def test_failed_build_removes_stale_executables(self):
        self.assertEqual(self.run_script('build_kernel_tests.sh').returncode, 0)
        self.env['TEST_COMPILE_FAIL'] = '1'
        result = self.run_script('build_kernel_tests.sh')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Tests: 2, failed: 2', result.stdout)
        self.assertFalse((self.scripts / 'build_out' / 'test_one').exists())

    def test_runtime_failure_is_aggregated(self):
        self.assertEqual(self.run_script('build_kernel_tests.sh').returncode, 0)
        self.env['TEST_RUNTIME_EXIT'] = '7'
        result = self.run_script('run_kernel_tests.sh')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Tests: 2, failed: 2', result.stdout)
        self.assertIn('EXIT CODE: 7', result.stdout)

    def test_missing_executables_fail(self):
        result = self.run_script('run_kernel_tests.sh')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('MISSING EXECUTABLE', result.stdout)

    def test_no_tests_fail(self):
        for source in self.scripts.glob('test_*.cpp'):
            source.unlink()
        for script in ('build_kernel_tests.sh', 'run_kernel_tests.sh'):
            with self.subTest(script=script):
                result = self.run_script(script)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Tests: 0', result.stdout)

    def test_missing_cann_fails(self):
        self.env.pop('ASCEND_TOOLKIT_HOME')
        for script in ('build_kernel_tests.sh', 'run_kernel_tests.sh'):
            with self.subTest(script=script):
                result = self.run_script(script)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Source the CANN set_env.sh', result.stderr)


if __name__ == '__main__':
    unittest.main()
