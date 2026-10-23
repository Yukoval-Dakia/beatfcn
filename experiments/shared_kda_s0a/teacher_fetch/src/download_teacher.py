import json,struct,subprocess,hashlib,time,shutil,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
header=json.loads((ROOT/'reports/safetensors_header.json').read_text())
base=8+(ROOT/'reports/safetensors_header.json').stat().st_size
commit='c0f9345a1de0e61237dec7da32a70049c4ec0fa8'
url=f'https://huggingface.co/spellbrush/PupuM2D/resolve/{commit}/pupum2dV2_25hz_large/checkpoint/step-0500000_loss-0.176985/model.safetensors?download=true'
items=sorted([(k,v) for k,v in header.items() if k.startswith(('patch_embed.','teacher.'))],key=lambda e:e[1]['data_offsets'][0])
nh={'__metadata__':{'format':'pt','source_repo':'spellbrush/PupuM2D','source_commit':commit,'subset':'shared patch_embed and EMA teacher encoder','full_lfs_sha256_verified':'false'}}
off=0;ranges=[]
for k,v in items:
 a,b=v['data_offsets'];nh[k]={**v,'data_offsets':[off,off+b-a]};off+=b-a
 if ranges and ranges[-1][1]==a:ranges[-1][1]=b
 else:ranges.append([a,b])
assert shutil.disk_usage(ROOT).free-off>1024**3,'Insufficient >=1GiB reserve'
hb=json.dumps(nh,separators=(',',':')).encode();hb+=b' '*((-len(hb))%8)
tmp=ROOT/'weights/teacher_encoder.safetensors.partial';dest=ROOT/'weights/teacher_encoder.safetensors'
t0=time.perf_counter();records=[]
with tmp.open('wb') as f:
 f.write(struct.pack('<Q',len(hb)));f.write(hb)
 for i,(a,b) in enumerate(ranges):
  hp=ROOT/f'reports/download_range_{i}_headers.txt';pos=f.tell();ts=time.perf_counter()
  cmd=['curl','-L','--fail','--retry','2','--retry-delay','2','--max-time','1200','-r',f'{base+a}-{base+b-1}','-D',str(hp),url]
  proc=subprocess.run(cmd,stdout=f,stderr=subprocess.PIPE)
  f.flush();n=f.tell()-pos
  rec={'source_start':base+a,'source_end':base+b-1,'bytes':n,'seconds':time.perf_counter()-ts,'returncode':proc.returncode};records.append(rec);print(json.dumps(rec),flush=True)
  if proc.returncode or n!=b-a: raise RuntimeError(f'Download failed: {rec}, {proc.stderr[-1000:]}')
  ht=hp.read_text().lower();assert f'content-range: bytes {base+a}-{base+b-1}/3363632912' in ht,'Range not honored'
  assert shutil.disk_usage(ROOT).free>1024**3
os.replace(tmp,dest)
filehash=hashlib.sha256();tensorhash={}
with dest.open('rb') as f:
 while chunk:=f.read(16*1024**2):filehash.update(chunk)
with dest.open('rb') as f:
 for k,v in nh.items():
  if k=='__metadata__':continue
  a,b=v['data_offsets'];f.seek(8+len(hb)+a);h=hashlib.sha256();left=b-a
  while left:
   z=f.read(min(left,16*1024**2));assert z;h.update(z);left-=len(z)
  tensorhash[k]={'sha256':h.hexdigest(),'shape':v['shape'],'dtype':v['dtype'],'source_offsets':header[k]['data_offsets']}
r={'source_url':url,'source_commit':commit,'publisher_fullfile_lfs_sha256':'06b1c2c7b93d6184923b052b5be0f6bc50040e87d5ef87acee51e15f0868ae57','publisher_fullfile_lfs_sha256_verified':False,'integrity_scope':'HTTPS range length/content-range verified; local subset and each tensor hashed; publisher does not supply per-tensor hashes','subset_sha256':filehash.hexdigest(),'size':dest.stat().st_size,'parameter_count':sum(__import__('math').prod(v['shape']) for k,v in items),'tensor_count':len(items),'ranges':records,'total_seconds':time.perf_counter()-t0,'tensor_hashes':tensorhash}
(ROOT/'reports/teacher_download_manifest.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='tensor_hashes'}),flush=True)
