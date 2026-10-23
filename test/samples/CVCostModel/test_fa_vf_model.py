# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import unittest
from fa_vf_model import Aff,Bridge


class AffineAccessTest(unittest.TestCase):
    def test_byte_stride_and_loop_footprint(self):
        bridge=object.__new__(Bridge);bridge.ranges={'row':(0,15),'col':(0,1)}
        self.assertEqual(bridge.bounds(Aff(32,(('row',128),('col',64)))),(32,2016))
    def test_reject_unaligned_and_non_affine(self):
        with self.assertRaisesRegex(ValueError,'unaligned'):Aff(3).divide(2)
        with self.assertRaisesRegex(ValueError,'non-affine'):Aff(0,(('x',1),))*Aff(0,(('y',1),))
    def test_unknown_loop_bound_is_not_zero(self):
        bridge=object.__new__(Bridge);bridge.ranges={}
        with self.assertRaises(KeyError):bridge.bounds(Aff(0,(('unknown',1),)))


if __name__=='__main__':unittest.main()
