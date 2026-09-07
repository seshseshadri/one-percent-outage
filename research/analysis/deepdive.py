import json, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
mat = pd.read_csv('data/matrix.csv', index_col=0).astype(bool)
md = pd.read_csv('data/submissions.csv', index_col=0)
rng = np.random.default_rng(2)
nm = lambda c: md.loc[c,'name']
pd.set_option('display.width',300)
def ci(x,n=3000):
    x=np.asarray(x,float); return np.percentile(rng.choice(x,(n,len(x))).mean(1),[2.5,97.5])*100

# (a) Full rescue table for A = #1 system, all top-40, with model tag
A='20251215_livesweagent_claude-opus-4-5'
top40=md.sort_values('resolved_rate',ascending=False).index[:40]
fails=~mat[A]
print(f'A = {nm(A)} {mat[A].mean()*100:.1f}%, fails on {fails.sum()}')
rows=[]
for B in top40:
    if B==A: continue
    b=mat[B]; lo,hi=ci(b[fails].values)
    rows.append(dict(system=nm(B), model=md.loc[B,'model'], glob=b.mean()*100, rescue=b[fails].mean()*100, lo=lo, hi=hi, n_res=int(b[fails].sum()), harm_if_all_switch=int((mat[A]&~b).sum()), union=int((mat[A]|b).sum())))
df=pd.DataFrame(rows); df['grank']=df.glob.rank(ascending=False).astype(int); df['rrank']=df.rescue.rank(ascending=False).astype(int)
print(df.sort_values('glob',ascending=False).round(1).to_string(index=False))

# (b) contamination sanity for Gemini 3.5 Flash: tasks it solves that few others solve
G='20260901_mini-v2.4.2_gemini-3-5-flash'
others=mat.drop(columns=[G]); sb=others.sum(axis=1)
g=mat[G]
print('\nGemini 3.5 Flash solves', g.sum(), 'tasks; of those, solved by <=3 other submissions:', ((sb<=3)&g).sum(), ' by 0 others:', ((sb==0)&g).sum())
print('For comparison, Claude 4.5 Opus (high) mini:', end=' ')
O='20260217_mini-v2.0.0_claude-4-5-opus-high'; sbo=mat.drop(columns=[O]).sum(axis=1)
print('solves',mat[O].sum(),'; <=3 others:',((sbo<=3)&mat[O]).sum(),'; 0 others:',((sbo==0)&mat[O]).sum())
print('Rescue-rate of Gemini 3.5 Flash by difficulty bucket of A-failures (solved_by count over all others):')
for lo_,hi_ in [(0,5),(6,20),(21,60),(61,200)]:
    m=fails&(sb>=lo_)&(sb<=hi_); print(f'  solved_by {lo_}-{hi_}: n={m.sum()}, gemini rescues {g[m].mean()*100 if m.sum() else 0:.0f}%')

# (c) real failure predictors, for A' = mini Claude 4.5 Opus high (has api_calls), backups in pool
def details(sub):
    d=json.load(open(f'swebench-experiments/evaluation/verified/{sub}/per_instance_details.json'))
    return pd.DataFrame(d).T.reindex(mat.index)
dA=details(O); a=mat[O]
api=dA.api_calls.astype(float); cost=dA.cost.astype(float)
y=(~a).astype(int)
print(f"\n(c) A' = {nm(O)} {a.mean()*100:.1f}%. Failure predictors:")
# consensus difficulty: fraction of other submissions (excluding any Claude 4.5 Opus runs) solving the task
excl=[c for c in mat.columns if 'opus-4-5' in c or 'opus_4_5' in c.lower() or 'claude-4-5-opus' in c or 'claude-opus-4-5' in c]
cons=1-mat.drop(columns=excl).mean(axis=1)
print('  AUC(api_calls -> A fails)=%.3f'%roc_auc_score(y,api), ' AUC(cost)=%.3f'%roc_auc_score(y,cost), ' AUC(consensus difficulty)=%.3f'%roc_auc_score(y,cons))
print('  api_calls: median when A succeeds %.0f, when A fails %.0f'%(api[a].median(),api[~a].median()))
# Routing evaluation: for backups C in [Gemini 3.5 Flash, MiniMax M2.5, Gemini 3 Flash], flag top-k by each score, route flagged to C
for C in [G,'20260217_mini-v2.0.0_minimax-2-5-high','20260217_mini-v2.0.0_gemini-3-flash-high','20251127_openhands_claude-opus-4-5']:
    c=mat[C]
    print(f'\n  backup C = {nm(C)} ({c.mean()*100:.1f}%): rescue={c[~a].mean()*100:.1f}% harm(A-only)={(a&~c).sum()} B-only={(~a&c).sum()} oracle union={(a|c).mean()*100:.1f}%')
    print('   flag-rate | api_calls: prec rec sys% | consensus: prec rec sys% | random: sys%')
    for frac in [0.1,0.2,0.3,0.4,0.5]:
        k=int(frac*500); out=[]
        for s in [api,cons]:
            flag=s.rank(ascending=False,method='first')<=k
            routed=np.where(flag,c,a); prec=y[flag].mean(); rec=flag[y==1].mean()
            out.append((prec*100,rec*100,routed.mean()*100))
        rf=pd.Series(rng.random(500),index=a.index).rank()<=k; rsys=np.where(rf,c,a).mean()*100
        print(f'   {frac:.0%}     | {out[0][0]:4.0f} {out[0][1]:4.0f} {out[0][2]:5.1f} | {out[1][0]:4.0f} {out[1][1]:4.0f} {out[1][2]:5.1f} | {rsys:5.1f}')
