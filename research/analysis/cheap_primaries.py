import json, itertools, numpy as np, pandas as pd
from scipy.stats import spearmanr
mat = pd.read_csv('data/matrix.csv', index_col=0).astype(bool)
md = pd.read_csv('data/submissions.csv', index_col=0)
rng = np.random.default_rng(0); nm = lambda c: md.loc[c,'name']
pd.set_option('display.width', 250)
G = '20260901_mini-v2.4.2_gemini-3-5-flash'
det = json.load(open(f'swebench-experiments/evaluation/verified/{G}/per_instance_details.json'))
covered = pd.Series([i in det for i in mat.index], index=mat.index)
v200 = [c for c in mat.columns if 'mini-v2.0.0' in c]
defaults = {'Haiku 4.5':'20260217_mini-v2.0.0_claude-4-5-haiku-high','Sonnet 4.5':'20260217_mini-v2.0.0_claude-4-5-sonnet-high','GPT-5 mini':'20260217_mini-v2.0.0_gpt-5-mini'}

def analyze(A, models, mask, label):
    M = mat.loc[mask, models]; a = M[A]; f = ~a
    rows = []
    for B in models:
        if B == A: continue
        b = M[B]; x = b[f].values.astype(float)
        bs = rng.choice(x, (3000, len(x))).mean(1)
        rows.append(dict(backup=nm(B), glob=b.mean()*100, rescue=b[f].mean()*100, n_res=int(x.sum()), lo=np.percentile(bs,2.5)*100, hi=np.percentile(bs,97.5)*100, harm=int((a&~b).sum()), union=(a|b).mean()*100, _x=x))
    df = pd.DataFrame(rows)
    df['grank'] = df.glob.rank(ascending=False, method='min').astype(int); df['rrank'] = df.rescue.rank(ascending=False, method='min').astype(int)
    df['move'] = df.grank - df.rrank
    print(f'\n=== {label} | default {nm(A)} = {a.mean()*100:.1f}% ({f.sum()} failures of {len(M)}) | Spearman(global,rescue)={spearmanr(df.glob, df.rescue).correlation:.2f} ===')
    print(df.drop(columns='_x').sort_values('glob', ascending=False).round(1).to_string(index=False))
    # inverted pairs: B above C globally by >=2 pts, C above B on rescue; bootstrap P(C>B on rescue)
    inv = []
    for i, j in itertools.permutations(range(len(df)), 2):
        B, C = df.iloc[i], df.iloc[j]
        if B.glob - C.glob >= 2 and C.rescue > B.rescue:
            samp = rng.integers(0, len(B._x), (3000, len(B._x)))
            p = (C._x[samp].mean(1) > B._x[samp].mean(1)).mean()
            inv.append((B.backup, C.backup, B.glob - C.glob, C.n_res - B.n_res, p))
    inv.sort(key=lambda t: -t[4])
    print('  inverted pairs (B ≥2pts better globally, C rescues more): B | C | global gap | extra rescues | P_boot(C>B)')
    for t in inv: print(f'    {t[0]:26s} | {t[1]:26s} | {t[2]:4.1f} | {t[3]:+3d} | {t[4]:.2f}')
    if not inv: print('    none')

for name, A in defaults.items():
    analyze(A, v200, pd.Series(True, index=mat.index), 'v2.0.0 cohort, 11 models, 500 tasks')
    analyze(A, v200 + [G], covered, 'v2.x cohort incl. Gemini 3.5 Flash, 12 models, 441 covered tasks')
