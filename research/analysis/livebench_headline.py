import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
rng=np.random.default_rng(0); pd.set_option('display.width',260)
df=pd.read_parquet('data/livebench_judgment.parquet'); df=df[df.turn==1]
cov=df.groupby('model').question_id.nunique(); full=cov[cov>=418].index
d=df[df.model.isin(full)]; q_ok=d.groupby('question_id').model.nunique(); d=d[d.question_id.isin(q_ok[q_ok==len(full)].index)]
P=d.pivot_table(index='question_id',columns='model',values='score',aggfunc='mean')
meta=d.drop_duplicates('question_id').set_index('question_id')[['category','task']].reindex(P.index)
g=P.mean().sort_values(ascending=False); P=P[g.index[:25]]; B=(P>=0.5)
cat=meta.category
def ci(x,n=3000):
    x=np.asarray(x,float); return np.percentile(rng.choice(x,(n,len(x))).mean(1),[2.5,97.5])*100
# 1. headline pairwise flips at category level: X>Y globally, Y>X on a category, both significant
print('=== Pairwise category flips (top-25 models; X better globally by >=2 pts; Y better on category with bootstrap P>=0.99) ===')
rows=[]
cols=list(P.columns)
for i,X in enumerate(cols):
    for Y in cols[i+1:]:   # X is globally better (sorted)
        if g[X]-g[Y] < 0.02: continue
        for c in ['coding','language','instruction_following']:
            m=(cat==c).values; x=P[X].values[m]; y=P[Y].values[m]
            samp=rng.integers(0,len(x),(2000,len(x))); p=(y[samp].mean(1)>x[samp].mean(1)).mean()
            if p>=0.99: rows.append(dict(X=X,Y=Y,global_gap=(g[X]-g[Y])*100,category=c,n=m.sum(),X_cat=x.mean()*100,Y_cat=y.mean()*100,cat_gap=(y.mean()-x.mean())*100,p=p))
r=pd.DataFrame(rows); r['score']=r.global_gap+r.cat_gap
print(len(r),'flips'); print(r.sort_values('score',ascending=False).head(20).round(1).to_string(index=False))
# 2. conditional leaderboard for a default, overall and per category
for A in ['claude-3-5-sonnet-20241022','gemini-2.0-flash-001']:
    f=~B[A]; print(f'\n=== default {A}: global={B[A].mean()*100:.0f}%  fails={f.sum()} ===')
    out=[]
    for Bk in B.columns:
        if Bk==A: continue
        row=dict(backup=Bk, glob=B[Bk].mean()*100, rescue_all=B[Bk][f].mean()*100)
        for c in ['coding','language','instruction_following']:
            m=f&(cat==c); row[f'resc_{c[:4]}']=B[Bk][m].mean()*100; row[f'n_{c[:4]}']=int(B[Bk][m].sum())
        out.append(row)
    o=pd.DataFrame(out); o['grank']=o.glob.rank(ascending=False).astype(int); o['rrank_all']=o.rescue_all.rank(ascending=False).astype(int)
    for c in ['codi','lang','inst']: o[f'rrank_{c}']=o[f'resc_{c}'].rank(ascending=False).astype(int)
    print(o.sort_values('glob',ascending=False).round(0).to_string(index=False))
    print('  failures by category:', {c:int((f&(cat==c)).sum()) for c in ['coding','language','instruction_following']})
    # per-category best backup with CI
    for c in ['coding','language','instruction_following']:
        m=f&(cat==c); best=o.sort_values(f'resc_{c[:4]}',ascending=False).head(3)
        print(f'  best backup for {c} failures (n={m.sum()}): '+' | '.join(f"{b.backup} {b[f'resc_{c[:4]}']:.0f}% [{ci(B[b.backup][m].values)[0]:.0f}-{ci(B[b.backup][m].values)[1]:.0f}]" for _,b in best.iterrows()))
