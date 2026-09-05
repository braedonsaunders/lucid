import sys,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0,'Tools/frontier_eval')
from score_native_packets import packet_image
from evaluate_sequences import Scorer,decode,digest,save,summarize
r=Path('.build/presented-calibrated');source=decode('.build/frontier-sequences-2x/four_people-h264-1000000.mkv',16);reference=decode('.build/frontier-sequences-2x/four_people-reference.mkv',16)
scorer=Scorer(torch.device('mps'));report={'purpose':'native stage alignment and quality diagnosis on four_people H264 1Mbps development','rows':[],'source_alignment':[],'complete':False,'code_sha256':digest(__file__)}
for label in ['shipping4x','direct2x_trained']:
 report[label+'_tuning']=json.loads((r/f'stages-{label}/tuning.json').read_text())
 for index in [8,12,16,20]:
  frame=index%16
  for suffix in ['source','preprocessed','reconstructed','enhanced']:
   path=r/f"stages-{label}/{index:08d}{'-'+suffix if suffix!='enhanced' else ''}.luce"
   image,header=packet_image(path)
   if suffix=='source':
    rmse=[float(np.mean((np.asarray(image,dtype=float)-np.asarray(x,dtype=float))**2)**.5) for x in source]
    report['source_alignment'].append({'model':label,'index':index,'expected':frame,'best':int(np.argmin(rmse)),'expected_rmse':rmse[frame],'minimum_rmse':min(rmse)})
   if suffix in ['source','preprocessed']:image=image.resize(reference[frame].size,Image.Resampling.LANCZOS)
   report['rows'].append({'source_id':'four_people','sequence_id':'four_people-h264-1000000','frame':frame,'variant':label+'-'+suffix,'metrics':scorer.spatial(image,reference[frame]),'packet_sha256':digest(path)})
  print(label,index,flush=True)
report['summary']=summarize(report['rows']);report['complete']=True;save(r/'native-stages.json',report)
print(report['source_alignment'])
for label,data in report['summary'].items():print(label,data['source_balanced'])
