import itertools, numpy as np, pandas as pd
from scipy.stats import spearmanr
mat = pd.read_csv('data/matrix.csv', index_col=0).astype(bool)
md = pd.read_csv('data/submissions.csv', index_col=0)
rng = np.random.default_rng(1)
name = lambda c: f"{md.loc[c,'name']} [{c[:8]}]"

# 1. Difficulty structure
print('=== Task difficulty: number of submissions solving each task (all 176) ===')
solved_by = mat.sum(1)
print(solved_by.describe().round(1).to_string())
print('tasks solved by 0 submissions:', (solved_by==0).sum(), ' by <=5:', (solved_by<=5).sum(), ' by all:', (solved_by==mat.shape[1]).sum())
cohort = [c for c in mat.columns if 'mini-v2.0.0' in c]
sb = mat[cohort].sum(1)
print(f'\ncohort (n={len(cohort)}) solved-by histogram:'); print(sb.value_counts().sort_index().to_string())

# 2. Spearman between global rank and rescue rank, for each default (cohort and all)
def rank_corr(cols, label, min_rate=0):
    out=[]
    for A in cols:
        fails = ~mat[A]
        others=[c for c in cols if c!=A]
        g = mat[others].mean(); r = mat[others][fails].mean()
        out.append((name(A), mat[A].mean()*100, spearmanr(g,r).correlation))
    df=pd.DataFrame(out, columns=['default','global','spearman(global,rescue)'])
    print(f'\n=== {label}: Spearman(global rank, rescue rank) per default ===')
    print(df.sort_values('global',ascending=False).round(2).to_string(index=False))
rank_corr(cohort, 'same-harness cohort')
top_all = md.sort_values('resolved_rate',ascending=False).index[:40].tolist()
rank_corr(top_all, 'top-40 of all submissions')

# 3. Run-to-run noise floor: same model, different harness versions / dates
print('\n=== Same-model repeated runs (agreement, both-fail, one-only) ===')
pairs = [
 ('20251124_mini-v1.16.0_claude-opus-4-5-20251101','20260217_mini-v2.0.0_claude-4-5-opus-high'),
 ('20251211_mini-v1.17.2_gpt-5.2-2025-12-11-high','20260217_mini-v2.0.0_gpt-5-2-high'),
 ('20250929_mini-v1.13.3_sonnet-4-5-20250929','20260217_mini-v2.0.0_claude-4-5-sonnet-high'),
 ('20251127_openhands_claude-opus-4-5','20260217_mini-v2.0.0_claude-4-5-opus-high'),
 ('20251215_livesweagent_claude-opus-4-5','20260217_mini-v2.0.0_claude-4-5-opus-high'),
 ('20251205_sonar-foundation-agent_claude-opus-4-5','20251127_openhands_claude-opus-4-5'),
]
for a,b in pairs:
    if a in mat and b in mat:
        x,y=mat[a],mat[b]
        print(f'{name(a)} ({x.mean()*100:.1f}) vs {name(b)} ({y.mean()*100:.1f}): agree={(x==y).mean()*100:.1f}%  a_only={(x&~y).sum()} b_only={(~x&y).sum()} both={(x&y).sum()} neither={(~x&~y).sum()}  P(b|a fails)={y[~x].mean()*100:.1f}')

# 4. Broad pool inversion search: top-40 submissions, A = top-10, B,C any
print('\n=== Inversions in top-40 pool (A in top 10). B>C globally by >=3pts, C>B on A-fails, p_boot>=0.95 ===')
res=[]
for A in top_all[:10]:
    fails=~mat[A]; idx=np.where(fails.values)[0]
    for B,C in itertools.permutations([c for c in top_all if c!=A],2):
        gB,gC=mat[B].mean(),mat[C].mean()
        if gB-gC < 0.03: continue
        b=mat[B].values[idx].astype(float); c=mat[C].values[idx].astype(float)
        if c.mean()<=b.mean(): continue
        samp=rng.integers(0,len(idx),size=(1000,len(idx)))
        p=(c[samp].mean(1)>b[samp].mean(1)).mean()
        if p>=0.95:
            res.append(dict(A=name(A),B=name(B),C=name(C),gB=gB*100,gC=gC*100,rB=b.mean()*100,rC=c.mean()*100,n_fail=len(idx),p=p))
r=pd.DataFrame(res)
if len(r):
    r['score']=(r.gB-r.gC)+(r.rC-r.rB)
    pd.set_option('display.width',300)
    print(len(r),'qualifying triples'); print(r.sort_values('score',ascending=False).head(20).round(1).to_string(index=False))
else: print('none')
