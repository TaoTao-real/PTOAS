"""Integrity binding for experimental physical-v2 checkpoints.

This digest detects accidental changes. It is not a legality certificate.
The compiler checks the digest and CANN identity again before emission.
"""
import hashlib
from ptoas.mlir import ir


def content_digest(module):
    attrs = module.operation.attributes
    removed = {key: attrs[key] for key in ('pto.cv.vf_digest', 'pto.cv.vf_origin') if key in attrs}
    try:
        for key in removed:
            del attrs[key]
        return hashlib.sha256(str(module.operation).encode()).hexdigest()
    finally:
        for key, value in removed.items():
            attrs[key] = value


def validate(module):
    attrs = module.operation.attributes
    for key, value in {'pto.cv.vf_state': 'physical-v2', 'pto.cv.vf_arch': 'a5',
                       'pto.cv.vf_build_level': 'level3', 'pto.cv.vf_sync': 'ptoas-v1'}.items():
        if key not in attrs or ir.StringAttr(attrs[key]).value != value:
            raise ValueError(f'incompatible checkpoint: {key}')
    if 'pto.cv.vf_cann' not in attrs or 'pto.cv.vf_origin' not in attrs:
        raise ValueError('checkpoint has no CANN or origin binding')
    if ir.StringAttr(attrs['pto.cv.vf_digest']).value != content_digest(module):
        raise ValueError('checkpoint content digest mismatch')


def seal(module):
    attrs = module.operation.attributes
    if 'pto.cv.vf_origin' not in attrs:
        raise ValueError('cannot seal a checkpoint without its original VF identity')
    attrs['pto.cv.vf_digest'] = ir.StringAttr.get(content_digest(module))


def rename(module, symbol):
    """Assign the same experiment symbol to the paired C/V entries."""
    import re
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', symbol):
        raise ValueError('invalid experiment symbol')
    count = 0
    for sub in module.body.operations:
        for func in sub.regions[0].blocks[0].operations:
            if func.operation.name == 'func.func':
                func.operation.attributes['sym_name'] = ir.StringAttr.get(symbol)
                count += 1
    if count != 2:
        raise ValueError(f'expected paired C/V entry functions, found {count}')
    seal(module)
