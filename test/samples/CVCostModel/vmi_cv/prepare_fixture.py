"""Open tile-only scope wrappers in the fixed M32 FA fixture for VMI analysis.

All four ablation variants use this identical high-level input. This is an
experiment fixture conversion, not a general PTOAS scheduling pass.
"""
import argparse,hashlib,json
from pathlib import Path
from ptoas.mlir import ir
from ptoas.mlir.dialects import pto

SOURCE_SHA='3c67b7a42b05b7fa1dd705e38892a335dc336bd6f14e7c025a05340e4fa00e9b'
TILE_OPS={'pto.trowmax','pto.trowexpandsub','pto.alloc_tile','pto.tmov','pto.tmuls','pto.texp','pto.trowsum','pto.tcvt','pto.tmax','pto.tsub','pto.trowexpandmul','pto.tadd','pto.tmul','pto.trowexpanddiv'}
def opof(op):return getattr(op,'operation',op)
def walk(op):
 yield op
 for region in op.regions:
  for block in region.blocks:
   for child in block.operations:yield from walk(opof(child))
def signature(module):
 return [(op.name,[str(v.type) for v in op.operands],[str(v.type) for v in op.results],[(a.name,str(a.attr)) for a in op.attributes]) for op in walk(module.operation) if op.name!='pto.vecscope']
def prepare(source,destination):
 raw=source.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('Only the frozen M32 FA fixture is supported')
 with ir.Context() as ctx:
  pto.register_dialect(ctx,load=True);module=ir.Module.parse(raw.decode());module.operation.verify();before=signature(module)
  scopes=[op for op in walk(module.operation) if op.name=='pto.vecscope']
  opened=[]
  for scope in reversed(scopes):
   if len(scope.regions)!=1 or len(scope.regions[0].blocks)!=1 or len(scope.results):raise ValueError('Unsupported scope interface')
   body=scope.regions[0].blocks[0]
   contents=[opof(op) for op in body.operations]
   # These full-tile instances have confirmed VMI candidates in the frozen
   # source. Keep ordinary/fallback templates in their authored scopes: their
   # masks can otherwise escape before scope inference.
   if len(contents)!=1 or contents[0].name not in {'pto.tmuls','pto.texp','pto.tcvt','pto.tadd'}:continue
   tiles=[pto.TileBufType(v.type) for v in contents[0].operands if str(v.type).startswith('!pto.tile_buf<')]
   if not tiles or any(tuple(t.shape)!=(16,128) or (t.valid_shape and tuple(t.valid_shape)!=(16,128)) for t in tiles):continue
   opened.append(contents[0].name)
   for op in walk(scope):
    if op is scope:continue
    if op.name not in TILE_OPS and not op.name.startswith(('arith.','scf.')):raise ValueError('Unsupported scope operation: '+op.name)
    if any('vreg<' in str(v.type) or 'mask<' in str(v.type) for v in [*op.operands,*op.results]):raise ValueError('Raw vector values must retain their scope')
   for op in list(body.operations):opof(op).move_before(scope)
   scope.erase()
  assert before==signature(module),'Opening scopes changed operation order, types or attributes'
  module.operation.verify();text=str(module)+'\n';ir.Module.parse(text).operation.verify();destination.write_text(text)
  report={'schema':'ptoas.fa.scope-opening.v2','source_sha256':SOURCE_SHA,'result_sha256':hashlib.sha256(text.encode()).hexdigest(),'opened_tile_only_scopes':len(opened),'opened_ops':opened,'retained_scopes':len(scopes)-len(opened),'operation_order_types_attributes_unchanged':True,'scope':'fixed fixture; only confirmed full-tile VMI instances opened; fallback/raw/strict scopes retained'}
  destination.with_suffix('.scope-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();prepare(a.source,a.destination)
