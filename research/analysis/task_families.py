"""Is there a model x task-family interaction in the same-harness cohort?"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
mat = pd.read_csv('data/matrix.csv', index_col=0).astype(bool)
md = pd.read_csv('data/submissions.csv', index_col=0)
tasks = pd.read_csv('data/tasks.csv').set_index('instance_id').reindex(mat.index)
rng = np.random.default_rng(0); nm = lambda c: md.loc[c,'name'].replace(' (high)','')
pd.set_option('display.width', 250)
cohort = [c for c in mat.columns if 'mini-v2.0.0' in c]
M = mat[cohort].astype(int); M.columns = [nm(c) for c in cohort]
M = M[M.mean().sort_values(ascending=False).index]
repo = tasks.repo.str.split('/').str[1]
diff = tasks.difficulty

def interaction_report(fam, label, min_n=15):
    counts = fam.value_counts(); keep = counts[counts >= min_n].index
    print(f'\n===== family = {label} (families with n>={min_n}: {list(keep)}) =====')
    print('n per family:', counts[keep].to_dict())
    R = M.groupby(fam).mean().loc[keep] * 100            # family x model success %
    print('\nSuccess % by family (rows) and model (cols, global order):'); print(R.round(0).astype(int).to_string())
    # additive fit (rates): r_fm ~ mu + a_f + b_m ; residual = interaction
    mu = M.mean().mean()*100; a = R.mean(1) - mu; b = R.mean(0) - mu
    resid = R - (mu + a.values[:,None] + b.values[None,:])
    print('\nResidual after removing family + model main effects (pct points):'); print(resid.round(0).astype(int).to_string())
    obs = (resid**2).values.sum()
    # permutation null: shuffle family labels across tasks (keeps model main effects & family sizes)
    null = []
    for _ in range(500):
        f2 = pd.Series(rng.permutation(fam.values), index=fam.index)
        R2 = M.groupby(f2).mean().loc[keep]*100; a2 = R2.mean(1)-mu; b2 = R2.mean(0)-mu
        null.append(((R2-(mu+a2.values[:,None]+b2.values[None,:]))**2).values.sum())
    null = np.array(null)
    print(f'\ninteraction sum-of-squares: observed={obs:.0f}, permutation null mean={null.mean():.0f} (95th pct {np.percentile(null,95):.0f}), p={(null>=obs).mean():.3f}')
    # per-family model rank vs global rank
    print('\nModel rank within each family (1=best; global order left->right):')
    rk = R.rank(axis=1, ascending=False, method='min').astype(int); print(rk.to_string())
    print('best model per family:', {f: f'{R.loc[f].idxmax()} ({R.loc[f].max():.0f}%, n={counts[f]})' for f in keep})
    return R, resid, keep

R1, res1, keep1 = interaction_report(repo, 'repo')
R2, res2, keep2 = interaction_report(diff, 'difficulty tier', min_n=40)

# Rescue view per repo: defaults Haiku 4.5 and Opus 4.5; which backup rescues most within each repo? with counts
print('\n===== Best backup per repo, by rescue COUNT (default fails in that repo) =====')
for A in ['Claude 4.5 Haiku', 'Claude 4.5 Opus', 'GPT 5 mini']:
    print(f'\n-- default {A}')
    f = M[A] == 0
    for fm in keep1:
        m = f & (repo == fm); n = m.sum()
        if n < 8: continue
        cnt = M.loc[m].drop(columns=[A]).sum().sort_values(ascending=False)
        top3 = ', '.join(f'{k} {v}' for k, v in cnt.head(3).items())
        g_best = M.drop(columns=[A]).mean().idxmax()
        print(f'   {fm:12s} fails={n:3d} | top rescuers: {top3} | global-best backup ({g_best}) rescues {cnt[g_best]}')

# heatmap of residuals
fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios':[1,0.5]})
for ax, (res, title) in zip(axes, [(res1, 'repo'), (res2, 'difficulty')]):
    im = ax.imshow(res.values, cmap='RdBu', vmin=-25, vmax=25, aspect='auto')
    ax.set_xticks(range(res.shape[1])); ax.set_xticklabels(res.columns, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(res.shape[0])); ax.set_yticklabels([f'{i} (n={int((repo if title=="repo" else diff).eq(i).sum())})' for i in res.index], fontsize=8)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]): ax.text(j, i, f'{res.values[i,j]:+.0f}', ha='center', va='center', fontsize=7)
    ax.set_title(f'model x {title} interaction residual (pct pts)\nafter removing model + family main effects')
plt.colorbar(im, ax=axes, shrink=0.6); plt.savefig('data/interaction_heatmap.png', dpi=130, bbox_inches='tight')
print('\nsaved data/interaction_heatmap.png')
