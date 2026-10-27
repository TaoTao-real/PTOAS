# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Contract tests for model-first candidate analysis and compile ordering."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ptoas import _cv_search as search
from ptoas._cv_search_events import LogicalEvents, Pointer


class RelativeVFTest(unittest.TestCase):
    def setUp(self):
        self.builder = LogicalEvents(dict(functions=[], buffers=[]), {})
        self.builder.core, self.builder.lane, self.builder.last = 'aiv0', 0, None
        self.function = dict(id='vf', operations=[], arguments=[])
        self.op = dict(id='call', name='func.call', attributes={})

    def test_cost_reused_across_buffer_identity(self):
        for root in ('buffer-a', 'buffer-b'):
            self.builder.vf(self.op, self.function, [Pointer(root, 'UB', 32, extent=1024)])
        self.assertEqual(len(self.builder.vfs), 1)
        self.assertNotEqual(self.builder.events[0]['cost']['bindings'], self.builder.events[1]['cost']['bindings'])

    def test_offset_and_alias_conditions_invalidate_cost(self):
        a = Pointer('a', 'UB', 32, extent=1024)
        from dataclasses import replace
        for args in ([a, a], [a, replace(a, offset=32)], [a, replace(a, root='b')]):
            self.builder.vf(self.op, self.function, args)
        self.assertEqual(len(self.builder.vfs), 3)

    def test_relative_access_range_is_checked(self):
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            self.builder.access(Pointer('a', 'UB', 32, offset=32, extent=64), 64, 'read')

    def test_unresolved_scalar_is_not_zero(self):
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            self.builder.vf(self.op, self.function, [dict(unknown='scalar')])

    def test_physical_allocation_rejected(self):
        self.builder.env = dict(address=4096)
        with self.assertRaisesRegex(ValueError, 'physical/unbound'):
            self.builder.op(dict(id='a', name='pto.alloc_tile', attributes={}, operands=['address'],
                results=[dict(id='tile', type=dict(kind='tile'))]))


class ResourceAndPathTest(unittest.TestCase):
    def test_two_aiv_domains_and_alias_once(self):
        b = dict(id='x', storage_id='shared:x', value='x', function='v', bytes=256,
            memory_space=dict(value=6))
        program = dict(functions=[dict(id='v', physical_instances=['AIV0', 'AIV1'], operations=[])], buffers=[b, dict(b, id='y')])
        plan = dict(buffers=[dict(buffer_id='x', count=1), dict(buffer_id='y', count=1)])
        target = dict(compiler_budget_bytes=dict(vec=512), alignment_bytes=dict(vec=256))
        result = search.resources(program, plan, target)
        self.assertEqual(result['bytes_by_domain'], {'AIV0:vec': 256, 'AIV1:vec': 256})
        target['compiler_budget_bytes']['vec'] = 128
        self.assertEqual(search.resources(program, plan, target)['status'], 'unresolved')

    def test_exchange_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('../outside', '/absolute'):
                with self.assertRaises(ValueError): search.local(folder, name)

    def test_multi_tile_type_is_structured(self):
        from ptoas.mlir import ir
        from ptoas.mlir.dialects import pto
        with ir.Context() as context:
            pto.register_dialect(context, load=True)
            type_ = ir.Type.parse('!pto.multi_tile_buf<!pto.tile_buf<vec, 16x32xf32>, count=3>')
            self.assertTrue(pto.MultiTileBufType.isinstance(type_))
            multi = pto.MultiTileBufType(type_)
            self.assertEqual(multi.count, 3)
            self.assertEqual(list(pto.TileBufType(multi.slot_type).shape), [16, 32])


class ModelResultTest(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(candidate_id='p'+str(i), preload_count=i, plan=dict(id=i), events_sha256=str(i)) for i in (0,1)]
        self.input = dict(input_fingerprint='input', candidates=self.rows)
        self.result = dict(schema='pto.cv.logical_model_result.v1', input_fingerprint='input', request_sha256='request',
            predictions=[dict(r, latency_us=t) for r,t in zip(self.rows,(100,70))],
            extensions={'tilesim.selection.v1': dict(action='optimize', candidate_id='p1', plan=self.rows[1]['plan'],
                ranked_candidates=['p1','p0'], predicted_gain=.3)})

    def validate(self):
        self.result.pop('fingerprint', None)
        self.result['fingerprint'] = search.cv.fingerprint(self.result)
        return search.validate_model_result(self.input, self.result, 'request')

    def test_valid_result(self): self.assertEqual(self.validate()['candidate_id'], 'p1')

    def test_unknown_or_duplicate_rank_is_rejected(self):
        for rank in (['p1','unknown'], ['p1','p1']):
            self.result['extensions']['tilesim.selection.v1']['ranked_candidates'] = rank
            with self.assertRaisesRegex(ValueError, 'ranked'): self.validate()

    def test_wrong_action_is_rejected(self):
        for action in ('baseline','unknown'):
            self.result['extensions']['tilesim.selection.v1']['action'] = action
            with self.assertRaisesRegex(ValueError, 'action'): self.validate()

    def test_stale_result_is_rejected(self):
        self.result['input_fingerprint'] = 'old'
        with self.assertRaisesRegex(ValueError, 'input/request'): self.validate()

    def test_claimed_gain_must_match_costs(self):
        self.result['extensions']['tilesim.selection.v1']['predicted_gain'] = .9
        with self.assertRaisesRegex(ValueError, 'costs'): self.validate()


class ModelFirstOrderingTest(unittest.TestCase):
    def run_flow(self, failure=None, baseline=False):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name) / 'run'; order = []
        predictions = [dict(candidate_id=p, plan=dict(candidate_id=p), latency_us=t)
            for p, t in (('p0', 100), ('p1', 70), ('p2', 75))]
        selected = dict(action='baseline' if baseline else 'optimize', candidate_id='p0' if baseline else 'p1',
            plan=dict(candidate_id='p0' if baseline else 'p1'), ranked_candidates=['p1', 'p2', 'p0'])
        def export(*a, **kw): order.append('prepare')
        def analyze(package, preloads, destination, **kw):
            order.append('analyze'); destination.mkdir(); return dict(candidates=predictions)
        def model(folder, config, command, destination):
            order.append('select'); destination.mkdir()
            search.cv._write(destination / 'result.json', dict(predictions=predictions, extensions={'tilesim.selection.v1': selected}))
            freeze = dict(selection=selected)
            search.cv._write(folder / 'selection.json', freeze)
            return freeze
        def compile_one(package, folder, output, **kw):
            self.assertTrue((folder / 'selection.json').exists())
            identity = json.loads(Path(kw['selection_path']).read_text())['selection']['candidate_id']
            order.append('compile:' + identity)
            if failure is not None: raise failure
            return dict(candidate_id=identity)
        with patch.object(search.cv, 'export_bundle', export), patch.object(search, 'export_candidates', analyze), \
             patch.object(search, 'run_model', model), patch.object(search, 'compile_selected', compile_one):
            result = search.optimize('source', {}, {}, {}, root, compiler='compiler',
                model_command=['model'], final_command=['final'])
        return result, order

    def test_select_before_single_selected_compile(self):
        result, order = self.run_flow()
        self.assertEqual(order, ['prepare', 'analyze', 'select', 'compile:p1'])
        self.assertEqual(result['vf_prepare_count'], 1)

    def test_two_infeasible_candidates_retain_baseline(self):
        result, order = self.run_flow(RuntimeError('INSUFFICIENT_SLOTS: test'))
        self.assertEqual(order[-2:], ['compile:p1', 'compile:p2'])
        self.assertEqual(result['status'], 'BASELINE_RETAINED')

    def test_identity_or_protocol_failure_is_terminal(self):
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            self.run_flow(RuntimeError('input identity mismatch'))

    def test_baseline_selection_does_not_fabricate_optimized_variant(self):
        result, order = self.run_flow(baseline=True)
        self.assertEqual(order[-1], 'compile:p0')
        self.assertEqual(result['status'], 'BASELINE_RETAINED')


if __name__ == '__main__': unittest.main()
