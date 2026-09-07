import json, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
mat = pd.read_csv('data/matrix.csv', index_col=0).astype(bool)
md = pd.read_csv('data/submissions.csv', index_col=0)
tasks = pd.read_csv('data/tasks.csv').set_index('instance_id').reindex(mat.index)
rng = np.random.default_rng(3); nm=lambda c: md.loc[c,'name']
pd.set_option('display.width',300)

# --- data cleanliness ---
print('=== cleanliness ===')
pi=[c for c in mat.columns if md.loc[c,'fmt']=='per_instance']
print('per_instance submissions:',len(pi),' all report 500?', all(md.loc[c,'n_reported']==500 for c in pi))
print('results_json submissions:',(md.fmt=='results_json').sum())
d=md[md.resolved_pct.notna()]; print('metadata resolved_pct vs recomputed mismatch >0.2:', (abs(d.resolved_pct-d.resolved_rate)>0.2).sum())
print('difficulty label coverage:', tasks.difficulty.notna().sum())

# --- model-family tagging (crude) ---
def fam(c):
    s=(str(md.loc[c,'model'])+' '+c+' '+str(nm(c))).lower()
    if 'opus-4-5' in s or 'opus_4_5' in s or 'opus 4.5' in s or 'claude-4-5-opus' in s or 'claude-opus-4-5' in s or '4.5 opus' in s: return 'claude-4.5-opus'
    if 'claude' in s or 'sonnet' in s or 'opus' in s or 'haiku' in s: return 'claude-other'
    if 'gpt' in s or 'o3' in s or 'o4' in s or 'openai' in s: return 'openai'
    if 'gemini' in s: return 'gemini'
    return 'other'
md['fam']=[fam(c) for c in md.index]

# --- generalization: for each top-10 default, rescue rate grouped by same-model vs different-model ---
print('\n=== For each top-10 default A: mean rescue% of backups in top-40, by family; and rank inversion size ===')
top40=md.sort_values('resolved_rate',ascending=False).index[:40].tolist()
for A in top40[:10]:
    f=~mat[A]; rows=[]
    for B in top40:
        if B==A: continue
        rows.append(dict(fam=md.loc[B,'fam'], same=(md.loc[B,'fam']==md.loc[A,'fam']), glob=mat[B].mean()*100, rescue=mat[B][f].mean()*100))
    r=pd.DataFrame(rows)
    g=r.groupby('same').agg(n=('rescue','size'),glob=('glob','mean'),rescue=('rescue','mean')).round(1)
    best_glob=r.sort_values('glob',ascending=False).iloc[0]; best_res=r.sort_values('rescue',ascending=False).iloc[0]
    print(f"A={nm(A)[:45]:45s} {mat[A].mean()*100:.1f}% fam={md.loc[A,'fam']:16s} | same-fam: n={g.loc[True,'n'] if True in g.index else 0} rescue={g.loc[True,'rescue'] if True in g.index else float('nan')} | diff-fam: n={g.loc[False,'n']} rescue={g.loc[False,'rescue']} | best-global backup rescues {best_glob.rescue:.1f} vs best-fallback {best_res.rescue:.1f}")

# --- stability of headline: A=live-SWE-agent Opus4.5; B=OpenHands Opus4.5 (#3) vs C=Gemini 3.5 Flash / TRAE Doubao ---
print('\n=== headline stability (random 80% task subsets, 2000 draws) ===')
A='20251215_livesweagent_claude-opus-4-5'; B='20251127_openhands_claude-opus-4-5'
for C in ['20260901_mini-v2.4.2_gemini-3-5-flash','20250928_trae_doubao_seed_code','20260217_mini-v2.0.0_gemini-3-flash-high']:
    f=~mat[A].values; b=mat[B].values; c=mat[C].values; n=500; k=400
    wins=0; gwins=0
    for _ in range(2000):
        idx=rng.choice(n,k,replace=False); ff=f[idx]
        wins+= c[idx][ff].mean()>b[idx][ff].mean(); gwins+= b[idx].mean()>c[idx].mean()
    print(f'  B={nm(B)} vs C={nm(C)}: P(C rescues more)={wins/2000:.3f}  P(B better globally)={gwins/2000:.3f}')

# --- alt default A' = mini Claude 4.5 Opus high (has api_calls): full rescue table top-40 ---
print("\n=== A' = mini Claude 4.5 Opus (high): rescue table (top-40 pool) ===")
O='20260217_mini-v2.0.0_claude-4-5-opus-high'; f=~mat[O]
rows=[]
for B in top40:
    if B==O: continue
    b=mat[B]; rows.append(dict(system=nm(B)[:50], fam=md.loc[B,'fam'], glob=b.mean()*100, rescue=b[f].mean()*100, n_res=int(b[f].sum()), harm=int((mat[O]&~b).sum()), union=(mat[O]|b).mean()*100))
r=pd.DataFrame(rows); r['grank']=r.glob.rank(ascending=False).astype(int); r['rrank']=r.rescue.rank(ascending=False).astype(int)
print(r.sort_values('glob',ascending=False).round(1).to_string(index=False))

# --- difficulty label as a REAL pre-run failure predictor + router, for A' ---
print("\n=== Difficulty label as failure predictor for A' ===")
a=mat[O]; y=(~a).astype(int)
dmap={'<15 min fix':0,'15 min - 1 hour':1,'1-4 hours':2,'>4 hours':3}
dl=tasks.difficulty.map(dmap)
print('AUC(difficulty label)=%.3f'%roc_auc_score(y,dl), ' AUC(problem_statement len)=%.3f'%roc_auc_score(y,tasks.ps_len), ' AUC(patch len)=%.3f'%roc_auc_score(y,tasks.patch_len), ' AUC(n_files)=%.3f'%roc_auc_score(y,tasks.n_files))
print(pd.DataFrame({'n':dl.groupby(tasks.difficulty).size(),'A_fail%':(y.groupby(tasks.difficulty).mean()*100).round(1)}))
dA=pd.DataFrame(json.load(open(f'swebench-experiments/evaluation/verified/{O}/per_instance_details.json'))).T.reindex(mat.index); api=dA.api_calls.astype(float)
for C in ['20260901_mini-v2.4.2_gemini-3-5-flash','20260217_mini-v2.0.0_minimax-2-5-high','20250928_trae_doubao_seed_code','20251127_openhands_claude-opus-4-5']:
    c=mat[C]; print(f'\n backup {nm(C)} ({c.mean()*100:.1f}) rescue={c[f].mean()*100:.1f}%  base={a.mean()*100:.1f}  oracle={(a|c).mean()*100:.1f}')
    print('  policy: route by difficulty label >= "1-4 hours": sys=%.1f (flag %d)'%(np.where(dl>=2,c,a).mean()*100,(dl>=2).sum()))
    print('  policy: route by difficulty label >= "15min-1h":  sys=%.1f (flag %d)'%(np.where(dl>=1,c,a).mean()*100,(dl>=1).sum()))
    for q in [0.8,0.7]:
        t=api.quantile(q); print('  policy: route if api_calls > p%d (%.0f): sys=%.1f (flag %d)'%(q*100,t,np.where(api>t,c,a).mean()*100,(api>t).sum()))
    # quadrant breakdown of who the difficulty policy flags
    fl=dl>=2; print('  flagged by ">=1-4h": A-only(harm)=%d  C-only(gain)=%d  both=%d neither=%d'%((fl&a&~c).sum(),(fl&~a&c).sum(),(fl&a&c).sum(),(fl&~a&~c).sum()))
    fl=api>api.quantile(0.8); print('  flagged by api>p80:  A-only(harm)=%d  C-only(gain)=%d  both=%d neither=%d'%((fl&a&~c).sum(),(fl&~a&c).sum(),(fl&a&c).sum(),(fl&~a&~c).sum()))
