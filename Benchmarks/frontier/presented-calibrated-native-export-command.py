import subprocess,os,json,hashlib
from pathlib import Path
root=Path('.build/presented-calibrated').resolve(); out=root/'native-delivered';out.mkdir()
exe=root/'pipeline-build/Build/Products/Release/Lucid.app/Contents/MacOS/Lucid'
sequences=json.loads(Path('.build/frontier-sequences-2x/sequences.json').read_text())
rows=[]
for sequence in sequences:
 if '-h264-' not in sequence['id']:continue
 clip=out/(sequence['id']+'.mp4')
 subprocess.run(['ffmpeg','-nostdin','-v','error','-stream_loop','1','-i',sequence['degraded'],'-frames:v','32','-c','copy',str(clip)],check=True)
 for label in ['shipping4x','direct2x_trained']:
  folder=out/(sequence['id']+'-'+label)
  env=dict(os.environ,LUCID_COMPUTE_UNITS='gpu',LUCID_PIPELINE_MODEL=str(root/f'native/{label}_640x360.mlpackage'),LUCID_PIPELINE_PACKETS=str(folder))
  p=subprocess.run([str(exe),'--pipeline-ms',str(clip),'16'],env=env,capture_output=True,text=True,timeout=180)
  (out/(sequence['id']+'-'+label+'.log')).write_text(p.stdout+p.stderr);p.check_returncode()
  packets=sorted(folder.glob('*.luce'));assert len(packets)==4
  rows.extend({'source_id':sequence['source_id'],'sequence_id':sequence['id'],'variant':label,'packet':str(packet),
      'reference':sequence['reference'],'frame':int(packet.stem)%16,'sha256':hashlib.sha256(packet.read_bytes()).hexdigest()} for packet in packets)
  print(sequence['id'],label,'4 packets',flush=True)
(out/'manifest.json').write_text(json.dumps({'rows':rows,'executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),
 'fixture':'32-frame remuxed loop of each 16-frame development H264 clip. Warmup8; measured16; export stride4. References at packet sequence modulo16.',
 'input_manifest_sha256':hashlib.sha256(Path('.build/frontier-sequences-2x/sequences.json').read_bytes()).hexdigest()},indent=2)+'\n')
