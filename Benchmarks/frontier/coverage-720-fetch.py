import sys,json,shutil
from pathlib import Path
sys.path.insert(0,'Tools/frontier_eval')
from fetch_sources import fetch,SVT
out=Path('.build/coverage-720-sources');out.mkdir(exist_ok=True)
shutil.copyfile('.build/frontier-eval-sources/SVT_MultiFormat_v10.pdf',out/'SVT_MultiFormat_v10.pdf')
rows=[]
for name in ['crowd_run','ducks_take_off','park_joy']:
 rows.append(fetch((name,name+'_2160p50.y4m','SVT: testing/developing/presenting technology standards; accompanying terms',SVT),out,16,4))
(out/'sources.json').write_text(json.dumps(rows,indent=2)+'\n')
