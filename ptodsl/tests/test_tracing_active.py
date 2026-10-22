# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from ptodsl._tracing import active


class ActiveTracingTest(unittest.TestCase):
    def test_nested_scopes_restore_outer_state(self):
        runtime, session, inner_runtime, inner_session = [object() for _ in range(4)]
        self.assertIsNone(active.current_runtime())
        self.assertIsNone(active.current_session())
        with active.activate_runtime(runtime), active.activate_session(session):
            with active.activate_runtime(inner_runtime), active.activate_session(inner_session):
                self.assertIs(active.current_runtime(), inner_runtime)
                self.assertIs(active.current_session(), inner_session)
            self.assertIs(active.current_runtime(), runtime)
            self.assertIs(active.current_session(), session)
        self.assertIsNone(active.current_runtime())
        self.assertIsNone(active.current_session())

    def test_exception_restores_state(self):
        with self.assertRaisesRegex(ValueError, "trace failed"):
            with active.activate_runtime(object()), active.activate_session(object()):
                raise ValueError("trace failed")
        self.assertIsNone(active.current_runtime())
        self.assertIsNone(active.current_session())

    def test_overlapping_workers_keep_their_own_scopes(self):
        # Force A-enter, B-enter, A-exit, B-exit. A process-wide LIFO stack
        # cannot represent this legal overlap, even with the Python GIL.
        a_entered, b_entered, a_exited = Event(), Event(), Event()
        a_runtime, b_runtime, a_session, b_session = [object() for _ in range(4)]

        def first():
            try:
                with active.activate_runtime(a_runtime), active.activate_session(a_session):
                    a_entered.set()
                    self.assertTrue(b_entered.wait(5))
                    self.assertIs(active.current_runtime(), a_runtime)
                    self.assertIs(active.current_session(), a_session)
            finally:
                a_exited.set()
            self.assertIsNone(active.current_runtime())
            self.assertIsNone(active.current_session())

        def second():
            self.assertTrue(a_entered.wait(5))
            with active.activate_runtime(b_runtime), active.activate_session(b_session):
                b_entered.set()
                self.assertTrue(a_exited.wait(5))
                self.assertIs(active.current_runtime(), b_runtime)
                self.assertIs(active.current_session(), b_session)
            self.assertIsNone(active.current_runtime())
            self.assertIsNone(active.current_session())

        with ThreadPoolExecutor(max_workers=2) as executor:
            left, right = executor.submit(first), executor.submit(second)
            left.result(timeout=10)
            right.result(timeout=10)
        self.assertIsNone(active.current_runtime())
        self.assertIsNone(active.current_session())


if __name__ == "__main__":
    unittest.main()
