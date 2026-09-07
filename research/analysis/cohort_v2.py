"""Same-harness cohort only (mini-SWE-agent v2.x): Gemini 3.5 Flash coverage handling + cheap-primary rescue orders."""
import json, numpy as np, pandas as pd
mat = pd.read_csv('data/matrix.csv', index_col=0).astype(bool)
md = pd.read_csv('data/submissions.csv', index_col=0)
tasks = pd.read_csv('data/tasks.csv').set_index('instance_id').reindex(mat.index)
rng = np.random.default_rng(0); nm = lambda c: md.loc[c,'name']
pd.set_option('display.width', 250)

cohort = [c for c in mat.columns if 'mini-v2.' in c]
G = '20260901_mini-v2.4.2_gemini-3-5-flash'
det = json.load(open(f'swebench-experiments/evaluation/verified/{G}/per_instance_details.json'))
covered = pd.Series([i in det for i in mat.index], index=mat.index)
print('cohort:', [nm(c) for c in cohort])
print(f'\n=== Check 1: Gemini 3.5 Flash covers {covered.sum()}/500. Missing {(~covered).sum()} ===')
miss = ~covered
others = [c for c in cohort if c != G]
solved_by_others = mat[others].sum(axis=1)
print('Missing tasks: solved by how many of the other 12 cohort models?')
print(pd.crosstab(pd.cut(solved_by_others, [-1,0,3,8,11,12], labels=['0','1-3','4-8','9-11','all 12']), miss, colnames=['missing']).to_string())
print('difficulty of missing vs covered:'); print(pd.crosstab(tasks.difficulty, miss, normalize='columns').round(2).to_string())
print('repo of missing tasks:'); print(tasks.repo[miss].value_counts().to_string())

# rescue rate two ways, for several defaults
print('\nGemini 3.5 Flash rescue rate by default, (a) missing=failure over all failures, (b) restricted to covered tasks:')
for A in others:
    f = ~mat[A]; g = mat[G]
    a_all = g[f].mean()*100; n_all = f.sum()
    a_cov = g[f & covered].mean()*100; n_cov = (f & covered).sum()
    print(f'  A={nm(A):28s} global(all)={g.mean()*100:.1f} global(covered)={g[covered].mean()*100:.1f} | rescue: missing=fail {a_all:5.1f}% (n={n_all})  covered-only {a_cov:5.1f}% (n={n_cov})  missing among A-fails={int((f&miss).sum())}')
