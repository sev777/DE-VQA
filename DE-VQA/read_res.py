

import os
import json
files=[]
for i,j,k in os.walk('/root///multmodal/LiveEdit-main/eval_results'):
    for kk in k:
        if kk=='mean_results.json':
            files.append(i+'/'+kk)

print()
ky=['model','data','method','t1i2','t2i1', 't2i2', 't1i4','t2i4','t1i3', 't3i1','t3i3','text_loc']
res=[ky]
for f in files:
    data=json.load(open(f,'r'))
    print()
    for i,j in data['total_mean'].items():
        if type(j)==dict:
            if  len(j)==9:
                temp=[f.split('/')[-4],f.split('/')[-3],f.split('/')[-5]]
                for k in ky[3:]:
                    if 't3' not in k and k!='text_loc':
                        temp.append(str(1-j[k]['acc']))
                    else:
                        temp.append(str(j[k]['acc']))
                res.append(temp)

for r in res:
    print('\t'.join(r))

# ['text_loc', 't3i3', 't1i4', 't2i4', 't1i2', 't1i3', 't2i1', 't2i2', 't3i1']


# T1I2	T2I1	T2I2	T1I4	T2I4	T1I3	T3I1	T3I3	T4I4
