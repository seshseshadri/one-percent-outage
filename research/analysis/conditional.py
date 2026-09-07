"""Conditional leaderboard analysis: does the best fallback differ from the best global model?"""
import sys, itertools
import numpy as np, pandas as pd

mat = pd.read_csv('data/matrix.csv', index_col=0)
md = pd.read_csv('data/submissions.csv', index_col=0)
rng = np.random.default_rng(0)

cohort_prefix = sys.argv[1] if len(sys.argv) > 1 else '20260217_mini-v2.0.0'
cols = [c for c in mat.columns if c.startswith(cohort_prefix)]
M = mat[cols].astype(bool)
short = {c: md.loc[c, 'name'] for c in cols}
print(f'cohort: {cohort_prefix}  n_models={len(cols)}  n_tasks={len(M)}')
glob = M.mean().sort_values(ascending=False)
print('\nGlobal leaderboard:')
for c, v in glob.items():
    print(f'  {v*100:5.1f}%  {short[c]}')

def boot_ci(x, n=2000):
    x = np.asarray(x, dtype=float)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return np.percentile(bs, [2.5, 97.5])

def conditional_table(A):
    a = M[A]
    fails = ~a
    print(f'\n=== Default A = {short[A]} ({a.mean()*100:.1f}%), A fails on {fails.sum()} tasks ===')
    rows = []
    for B in cols:
        if B == A: continue
        b = M[B]
        rows.append(dict(model=short[B], global_pct=b.mean()*100,
                         rescue_pct=b[fails].mean()*100, n_rescued=int(b[fails].sum()),
                         ci=boot_ci(b[fails].values),
                         both=int((a&b).sum()), a_only=int((a&~b).sum()), b_only=int((~a&b).sum()), neither=int((~a&~b).sum()),
                         union=int((a|b).sum())))
    df = pd.DataFrame(rows)
    df['global_rank'] = df.global_pct.rank(ascending=False).astype(int)
    df['rescue_rank'] = df.rescue_pct.rank(ascending=False).astype(int)
    df = df.sort_values('global_pct', ascending=False)
    pd.set_option('display.width', 250)
    print(df[['model','global_pct','global_rank','rescue_pct','rescue_rank','n_rescued','ci','a_only','b_only','both','neither','union']].round(1).to_string(index=False))
    return df

top = glob.index[0]
df = conditional_table(top)
# also second-best as default
conditional_table(glob.index[1])

# Search all (A,B,C) triples in cohort for inversions: B>C globally, C>B on A's failures
print('\n=== Strongest inversions across all (A,B,C) triples ===')
res = []
for A in cols:
    fails = ~M[A]
    for B, C in itertools.permutations([c for c in cols if c != A], 2):
        gB, gC = M[B].mean(), M[C].mean()
        rB, rC = M[B][fails].mean(), M[C][fails].mean()
        if gB > gC and rC > rB:
            # bootstrap: P(rC > rB) over task resampling
            idx = np.where(fails.values)[0]
            b = M[B].values[idx].astype(float); c = M[C].values[idx].astype(float)
            samp = rng.integers(0, len(idx), size=(2000, len(idx)))
            p = (c[samp].mean(1) > b[samp].mean(1)).mean()
            res.append(dict(A=short[A], B=short[B], C=short[C], gB=gB*100, gC=gC*100, gap_global=(gB-gC)*100,
                            rB=rB*100, rC=rC*100, gap_rescue=(rC-rB)*100, n_fail=len(idx), p_boot=p))
r = pd.DataFrame(res)
r['score'] = r.gap_global + r.gap_rescue
print(r.sort_values('score', ascending=False).head(25).round(2).to_string(index=False))
r.to_csv('data/inversions_%s.csv' % cohort_prefix, index=False)
