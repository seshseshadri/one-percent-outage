"""Build a task x submission success matrix from the SWE-bench experiments repo.

Output: matrix.csv (rows = instance_id, cols = submission dir, values 1/0/NaN)
        submissions.csv (metadata per submission)
"""
import json, os, sys, glob
import pandas as pd, yaml

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'swebench-experiments/evaluation/verified'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'data'
os.makedirs(OUT, exist_ok=True)

rows, meta = {}, []
for sub in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, sub)
    if not os.path.isdir(d):
        continue
    pid = os.path.join(d, 'per_instance_details.json')
    rj = os.path.join(d, 'results', 'results.json')
    resolved, n_reported, fmt = None, None, None
    if os.path.exists(pid):
        det = json.load(open(pid))
        resolved = {k: bool(v.get('resolved')) for k, v in det.items()}
        n_reported = len(det); fmt = 'per_instance'
    elif os.path.exists(rj):
        r = json.load(open(rj))
        resolved = {i: True for i in r.get('resolved', [])}
        n_reported = None; fmt = 'results_json'
    else:
        continue
    m = {}
    for mf in ('metadata.yaml', 'metadata.yml'):
        p = os.path.join(d, mf)
        if os.path.exists(p):
            try:
                m = yaml.safe_load(open(p)) or {}
            except Exception as e:
                print('bad yaml', sub, e, file=sys.stderr)
    info, tags = m.get('info', {}) or {}, m.get('tags', {}) or {}
    meta.append(dict(
        submission=sub, date=sub[:8], name=info.get('name'), fmt=fmt,
        resolved_pct=info.get('resolved'), agent=tags.get('agent'),
        model=','.join(tags.get('model') or []) if isinstance(tags.get('model'), list) else tags.get('model'),
        model_org=tags.get('model_org'), os_system=tags.get('os_system'), os_model=tags.get('os_model'),
        attempts=(tags.get('system') or {}).get('attempts') if isinstance(tags.get('system'), dict) else None,
        reasoning_effort=tags.get('reasoning_effort'), instance_cost=info.get('instance_cost'),
        n_reported=n_reported,
    ))
    rows[sub] = resolved

# Universe of instances = union across per_instance submissions (should be the 500 Verified ids)
ids = set()
for sub, r in rows.items():
    ids |= set(r)
ids = sorted(ids)
print('instances in union:', len(ids))
# results.json format lists only resolved ids -> everything else is unresolved (0)
mat = pd.DataFrame({sub: [1 if r.get(i, False) else 0 for i in ids] for sub, r in rows.items()}, index=ids)
md = pd.DataFrame(meta).set_index('submission')
md['n_resolved'] = mat.sum()
md['resolved_rate'] = md['n_resolved'] / len(ids) * 100
md['coverage'] = md['n_reported'].fillna(len(ids)) / len(ids)
# Exclude submissions whose per-instance file disagrees with the leaderboard number (broken uploads).
bad = md.index[(md.resolved_pct.notna()) & ((md.resolved_pct - md.resolved_rate).abs() > 1.0)]
for b in bad:
    print('EXCLUDING (per-instance file inconsistent with metadata):', b, md.loc[b, 'resolved_pct'], md.loc[b, 'resolved_rate'], file=sys.stderr)
md = md.drop(bad); mat = mat.drop(columns=bad)
partial = md.index[md.coverage < 1]
for b in partial:
    print('WARNING partial coverage (missing instances counted as failures):', b, md.loc[b, 'n_reported'], file=sys.stderr)
mat.to_csv(os.path.join(OUT, 'matrix.csv'))
md.to_csv(os.path.join(OUT, 'submissions.csv'))
print(md[['name', 'resolved_pct', 'resolved_rate', 'agent', 'fmt']].sort_values('resolved_rate', ascending=False).to_string())
