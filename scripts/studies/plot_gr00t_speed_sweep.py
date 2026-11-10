"""Build standard, complete three-panel figures from verified local studies."""
import argparse
import csv,dataclasses
from pathlib import Path
import yaml
from vlalagbench.study.point import read_point,verify_files
from vlalagbench.study.report import plot
from vlalagbench.study.spec import open_study
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--require-host', help='Reject results from any other host')
parser.add_argument('--require-gpu', help='Require this GPU model in every run record')
args = parser.parse_args()
root = args.root
for task in ('basket','drawer'):
 stem=f'gr00t_{task}_ellipse_600_20260914'
 plans=[open_study(root/(stem+suffix)) for suffix in ('','_continuous')]
 records={}
 sources={}
 for plan in plans:
  for point in plan.points:
   directory=plan.study_dir/'points'/point.name
   record=read_point(directory,plan.study_hash);verify_files(directory,record)
   hardware=yaml.safe_load((directory/'runs'/record.run_id/'result.yaml').read_text())['hardware']
   if args.require_host and hardware['hostname'] != args.require_host:
    raise ValueError(f'{directory}: unexpected host {hardware["hostname"]}')
   if args.require_gpu and {gpu['type'] for gpu in hardware['gpus']} != {args.require_gpu}:
    raise ValueError(f'{directory}: unexpected GPUs {hardware["gpus"]}')
   assert point.name not in records
   records[point.name]=record;sources[point.name]=plan.spec.name
 assert len(records)==72
 assert all(r.episodes==200 and r.deadline_steps==600 for r in records.values())
 assert all(r.values['trajectory.speed_mps']<=0.2 for r in records.values())
 title=f'Moving {task} (GR00T N1.7, 600-step deadline)'
 if args.require_gpu:
  title=f'Moving {task} (GR00T N1.7, {args.require_gpu.removeprefix("NVIDIA GeForce ")}, 600 steps)'
 spec=plans[0].spec.model_copy(deep=True,update={'name':title})
 spec.sweep['inference.mode']=['synchronous','lazy','continuous']
 merged=dataclasses.replace(plans[0],spec=spec,points=[p for plan in plans for p in plan.points])
 output=args.output/f'{task}_ellipse_600';output.mkdir(parents=True,exist_ok=True)
 keys=list(spec.sweep)
 with (output/'plot_data.csv').open('w',newline='') as stream:
  writer=csv.writer(stream,lineterminator='\n')
  writer.writerow([*keys,'successes','episodes','success_rate','ci95_low','ci95_high','source','run_id','commit'])
  for r in sorted(records.values(),key=lambda r:tuple(r.values[k] for k in keys)):
   writer.writerow([*(r.values[k] for k in keys),r.successes,r.episodes,r.success_rate,r.ci95_low,r.ci95_high,sources[r.name],r.run_id,r.commit])
 plot(merged,records,output/'success_rates.png')
 print(output,flush=True)
