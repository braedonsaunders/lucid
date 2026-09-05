#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
python3 Tools/package_models.py
python3 - <<'CHECK'
import json,re,subprocess
from pathlib import Path
m=json.loads(Path('Lucid/Resources/Models.json').read_text())
source=Path('Lucid/Metal/LearnedUpscaler.swift').read_text()
actual={(int(w),int(h)) for w,h in re.findall(r'Variant\(width: *(\d+), *height: *(\d+)',source)}
assert actual=={(x['width'],x['height']) for x in m['models']}, 'Runtime ladder disagrees with model manifest'
for item in m['models']:
 subprocess.run(['git','ls-files','--error-unmatch','Model/'+item['name']+'.mlpackage'],check=True,stdout=subprocess.DEVNULL)
assert 'package_models.py' in Path('Lucid.xcodeproj/project.pbxproj').read_text()
print('Model ladder and Xcode packaging agree')
CHECK
