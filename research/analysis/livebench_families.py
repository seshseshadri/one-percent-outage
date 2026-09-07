import numpy as np, pandas as pd
rng=np.random.default_rng(0); pd.set_option('display.width',260)
df=pd.read_parquet('data/livebench_judgment.parquet')
df=df[df.turn==1]
# models with (near) full coverage, common question set
cov=df.groupby('model').question_id.nunique(); full=cov[cov>=418].index
d=df[df.model.isin(full)]
q_ok=d.groupby('question_id').model.nunique(); qs=q_ok[q_ok==len(full)].index
d=d[d.question_id.isin(qs)]
P=d.pivot_table(index='question_id',columns='model',values='score',aggfunc='mean')
meta=d.drop_duplicates('question_id').set_index('question_id')[['category','task']].reindex(P.index)
print('models',P.shape[1],'questions',P.shape[0]); print(meta.category.value_counts().to_dict()); print(meta.task.value_counts().to_dict())
g=P.mean().sort_values(ascending=False); P=P[g.index]
# keep top 25 models by global to keep it readable
P=P[g.index[:25]]
print('\nGlobal (mean score) top 25:'); print((g.head(25)*100).round(1).to_string())
def interaction(fam,label,nperm=500):
    R=P.groupby(fam).mean()*100; mu=P.values.mean()*100; a=R.mean(1)-mu; b=R.mean(0)-mu
    res=R-(mu+a.values[:,None]+b.values[None,:]); obs=(res**2).values.sum()
    null=[]
    for _ in range(nperm):
        f2=pd.Series(rng.permutation(fam.values),index=fam.index); R2=P.groupby(f2).mean()*100; a2=R2.mean(1)-mu; b2=R2.mean(0)-mu
        null.append(((R2-(mu+a2.values[:,None]+b2.values[None,:]))**2).values.sum())
    null=np.array(null)
    print(f'\n===== family={label}: interaction SS observed={obs:.0f} null mean={null.mean():.0f} 95th={np.percentile(null,95):.0f} p={(null>=obs).mean():.3f}')
    print('score % by family x model:'); print(R.round(0).astype(int).T.to_string())
    print('\nrank within family (1=best):'); print(R.rank(axis=1,ascending=False,method='min').astype(int).T.to_string())
    return R
R=interaction(meta.category,'category')
interaction(meta.task,'task')
# rescue view with binarized outcomes: success = score>=0.5 ; defaults = a couple of cheap models
B=(P>=0.5)
for A in [m for m in ['gpt-4o-mini-2024-07-18','claude-3-5-haiku-20241022','gemini-2.0-flash-001'] if m in B]:
    f=~B[A]; print(f'\n-- default {A} global={B[A].mean()*100:.0f}% fails={f.sum()}')
    for cat in meta.category.unique():
        m=f&(meta.category==cat); cnt=B.loc[m].drop(columns=[A]).sum().sort_values(ascending=False)
        gbest=B.drop(columns=[A]).mean().idxmax()
        print(f'   {cat:22s} fails={m.sum():3d} top rescuers: '+', '.join(f'{k} {v}' for k,v in cnt.head(3).items())+f' | global-best ({gbest}) rescues {cnt[gbest]}')
