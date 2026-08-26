# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

import unittest

from ptodsl.tilelib import TileSpec, f32
from ptodsl.tilelib.templates.a5.tchannel_merge import (
    template_tchannel_merge_k2,
    template_tchannel_merge_k4,
)
from ptodsl.tilelib.templates.a5.tchannel_split import (
    template_tchannel_split_k2,
    template_tchannel_split_k4,
)
from ptodsl.vmi_tilelib_helper import instantiate_candidate


def _operand_spec(shape):
    return {
        "kind": "tile",
        "dtype": "f32",
        "shape": list(shape),
        "valid_shape": list(shape),
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }


class ChannelRearrangeTemplateTest(unittest.TestCase):
    def test_ordinary_k2_k4(self):
        wide = TileSpec((8, 64), f32)
        half = TileSpec((8, 32), f32)
        quarter = TileSpec((8, 16), f32)
        artifacts = (
            template_tchannel_split_k2.specialize(
                src=wide, dst0=half, dst1=half
            ),
            template_tchannel_split_k4.specialize(
                src=wide,
                dst0=quarter,
                dst1=quarter,
                dst2=quarter,
                dst3=quarter,
            ),
            template_tchannel_merge_k2.specialize(
                src0=half, src1=half, dst=wide
            ),
            template_tchannel_merge_k4.specialize(
                src0=quarter,
                src1=quarter,
                src2=quarter,
                src3=quarter,
                dst=wide,
            ),
        )
        for artifact in artifacts:
            text = artifact.mlir_text()
            self.assertNotIn("pto.vmi.", text)
            self.assertIn("scf.for", text)
            self.assertTrue("pto.vdintlv" in text or "pto.vintlv" in text)

    def test_selected_vmi_k2(self):
        split = instantiate_candidate(
            target="a5",
            op_name="pto.tchannel_split",
            operand_specs=[
                _operand_spec((8, 64)),
                _operand_spec((8, 32)),
                _operand_spec((8, 32)),
            ],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={},
        ).mlir_text()
        self.assertIn("pto.vmi.channel_split", split)
        self.assertEqual(split.count("pto.vmi.vstore"), 2)

        merge = instantiate_candidate(
            target="a5",
            op_name="pto.tchannel_merge",
            operand_specs=[
                _operand_spec((8, 32)),
                _operand_spec((8, 32)),
                _operand_spec((8, 64)),
            ],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={},
        ).mlir_text()
        self.assertIn("pto.vmi.channel_merge", merge)
        self.assertEqual(merge.count("pto.vmi.vload"), 2)


if __name__ == "__main__":
    unittest.main()

