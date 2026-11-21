"""Continue the user's live Colab baseline; no model download or reload."""
from pathlib import Path
import contextlib
import io
import json
import os
import subprocess
import sys
import traceback
import zipfile

required = ('BASE','BASE_TABLE','BASE_SCORES','PAIRS','MODEL','AGENT','llm')
missing = [key for key in required if key not in globals()]
if missing:
    raise RuntimeError('Сессия потеряла данные: ' + ', '.join(missing) +
                       '. Откройте lab06_ready.ipynb из форка и выполните его на T4.')
assert AGENT.model_id == 'Qwen/Qwen2.5-3B-Instruct'
assert len(PAIRS) == 200 and len(BASE.verdicts) == 400
for key in ('PREDICTION_0','PREDICTION_1'):
    globals().setdefault(key, '')
RUN_REAL_MODEL = True
PRIOR_DISCARDED_CALLS = MODEL.calls - BASE.calls - 1

work = Path('/content/lab06_finish')
if not (work / '.git').exists():
    subprocess.run(['git','clone','--depth','1',
                    'https://github.com/aibergeniskendirov-hub/lab-06-AI-course.git',str(work)],check=True)
# Preserve any personal declaration already edited in the current runtime.
old_justification = Path('JUSTIFICATION.md')
if old_justification.exists():
    old_text = old_justification.read_text()
    if '## AI-use declaration' in old_text:
        new_text = (work/'JUSTIFICATION.md').read_text()
        (work/'JUSTIFICATION.md').write_text(new_text.split('## AI-use declaration',1)[0] +
            '## AI-use declaration' + old_text.split('## AI-use declaration',1)[1])
os.chdir(work)
sys.path.insert(0,str(work))
for module in ('write_reports','student_controls'):
    sys.modules.pop(module,None)
ready = json.loads(Path('lab06_ready.ipynb').read_text())
original = json.loads(Path('evidence/original_colab_run.ipynb').read_text())
# Retain authentic baseline code+outputs together; execute new controls, never attach old outputs to new code.
ready['cells'][:18] = original['cells'][:18]
bootstrap = """# Fetch report helpers when running this saved notebook from the beginning.
from pathlib import Path
import os, sys, subprocess
work = Path('/content/lab06_finish')
if not (work / '.git').exists():
    subprocess.run(['git','clone','--depth','1',
        'https://github.com/aibergeniskendirov-hub/lab-06-AI-course.git',str(work)],check=True)
os.chdir(work)
sys.path.insert(0,str(work))
RUN_REAL_MODEL = True
"""
ready['cells'][19]['source'] = (bootstrap + ''.join(ready['cells'][19]['source'])).splitlines(True)
# The saved notebook can also be run from scratch on Colab; existing baseline outputs are from the prior real run.
ready['cells'][0]['source'].append('\n\nBaseline outputs retained from the original real Colab run; controls below completed in its live session.\n')
stream = sys.stdout
class Tee(io.StringIO):
    def write(self,text):
        stream.write(text);stream.flush()
        return super().write(text)

counts = [c.get('execution_count') or 0 for c in ready['cells'][:18] if c['cell_type']=='code']
execution = max(counts,default=0)
for index in range(18,50):
    cell = ready['cells'][index]
    if cell['cell_type'] != 'code':continue
    print(f'Completing cell {index}...',flush=True)
    execution += 1
    buf=Tee()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(''.join(cell['source']),f'completed-cell-{index}','exec'),globals())
    except BaseException as exc:
        cell['execution_count']=execution
        cell['outputs']=[{'output_type':'stream','name':'stdout','text':buf.getvalue().splitlines(True)},
                         {'output_type':'error','ename':type(exc).__name__,'evalue':str(exc),
                          'traceback':traceback.format_exc().splitlines()}]
        Path('lab06_harness_says_no.ipynb').write_text(json.dumps(ready,ensure_ascii=False,indent=1))
        raise
    cell['execution_count']=execution
    cell['outputs']=[{'output_type':'stream','name':'stdout','text':buf.getvalue().splitlines(True)}]
    Path('lab06_harness_says_no.ipynb').write_text(json.dumps(ready,ensure_ascii=False,indent=1))
# The separately retained download cell is only a convenience; remove it from the executed artifact.
ready['cells']=ready['cells'][:50]
Path('lab06_harness_says_no.ipynb').write_text(json.dumps(ready,ensure_ascii=False,indent=1)+'\n')
archive='/content/lab06_completed.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name in ('lab06_harness_says_no.ipynb','REPORT.md','JUSTIFICATION.md',
                 'control_register.json','decisions.jsonl','results.json'):
        z.write(name,arcname=name)
print('ГОТОВО: скачивается lab06_completed.zip, включая notebook с настоящими outputs.')
from google.colab import files
files.download(archive)
