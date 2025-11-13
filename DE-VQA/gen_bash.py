

import os

models=[]
for i,j,k in os.walk('/root///multmodal/LiveEdit-main/records'):
    for kk in k:
        if kk.endswith('Best'):
            models.append(i+'/'+kk)
res=[[],[],[],[]]
for m in models:
    dt='VLKEB' if 'VLKEB' in m else 'EVQA'
    cu=0 if 'VLKEB' in m else 1
    cm=f'python test_vllm_edit.py -en {m.split("/")[-5]} -mn {m.split("/")[-4]} -sen 1 -dvc cuda:{cu} -ckpt {m} -dn {dt} -dsn 500'
    res[cu].append(cm)

for ed  in ['lemoe_vl','tp_vl']:

    for md in ['blip2','minigpt4']:
        for dt in ['VLKEB','EVQA']:
            if md == 'blip2':
                cm=f'python test_vllm_edit.py -en {ed} -mn blip2 -sen 1 -dvc cuda:2 -dn {dt} -dsn 500'
                res[2].append(cm)
            else:

                cm=f'python test_vllm_edit.py -en {ed} -mn minigpt4 -sen 1 -dvc cuda:3 -dn {dt} -dsn 500'
                res[3].append(cm)

for i,r in enumerate(res):
    with open(f'bash{i}.sh','w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n'.join(r))