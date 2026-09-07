# Research note: does real data support "failure prediction is not model routing"?

**Verdict: yes, with one important reframing.** The conditional-leaderboard effect is large, robust, and visually
obvious when the "models" being compared are *systems* (agent scaffold + model) on SWE-bench Verified. Within a
single scaffold with only the model swapped, the effect mostly disappears. The failure-predictor effect is
demonstrable with *real* signals (no synthetic data needed), but the routing gains are small in absolute terms
and must be shown honestly.

Time spent on the whole research phase (§1–§11, 2026-09-07): ~90 minutes wall-clock with Claude Code.

## 1. Data source

**SWE-bench Verified** leaderboard submissions, from the public repo `github.com/SWE-bench/experiments`
(`evaluation/verified/*`). Every submission ships per-instance outcomes: either `per_instance_details.json`
(`{instance_id: {resolved, api_calls, cost}}`, 41 submissions, all from the mini-SWE-agent bash-only leaderboard)
or `results/results.json` (a list of resolved instance ids, 134 submissions). Task metadata (repo, official
difficulty annotation) comes from the HF dataset `princeton-nlp/SWE-bench_Verified`.

Result: a **500 tasks × 173 submissions** binary success matrix. `data/swebench_verified_outcomes.json` (307 KB)
is the demo-ready export; `data/matrix.csv`, `submissions.csv`, `tasks.csv` are the analysis tables.

Other candidates considered and rejected for this exercise: HELM and Open LLM Leaderboard per-sample data (huge,
heterogeneous, weak "hard task" structure), LiveCodeBench/Aider polyglot (fewer public per-instance model results
in one place). SWE-bench Verified is the cleanest single place with many systems on the *same* 500 tasks.

Regenerate:
```bash
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/SWE-bench/experiments.git swebench-experiments
cd swebench-experiments && git sparse-checkout init --no-cone && git sparse-checkout set \
  'evaluation/verified/*/results/results.json' 'evaluation/verified/*/metadata.yaml' 'evaluation/verified/*/per_instance_details.json' && git checkout && cd ..
python analysis/build_matrix.py      # -> data/matrix.csv, data/submissions.csv
python analysis/conditional.py       # same-harness cohort analysis
python analysis/robustness.py        # difficulty structure, rank correlations, noise floor, broad-pool inversions
python analysis/deepdive.py          # headline example, contamination check, real failure predictors
python analysis/generalize.py        # generalization across defaults, subsample stability, difficulty labels
```

## 2. Data cleanliness

| Concern | Finding |
|---|---|
| Same benchmark / task set | Yes. Union of instance ids across all submissions is exactly the 500 Verified tasks. |
| Missing results | `results.json` submissions list only resolved ids; everything else is treated as failed (this is how the leaderboard scores them). One submission (Gemini 3.5 Flash, 2026-09-01) reports only 441/500 instances; missing ones are counted as failures. Flagged in `coverage` and **not used as a headline example**. |
| Broken uploads | Two `per_instance_details.json` files disagree with their own metadata score by >40 points (Gemini 3 Pro 2026-02-26 shows 0/500 resolved; a mini-v0.0.0 Claude 3.7 run shows 51 vs claimed 264). Both excluded automatically by `build_matrix.py`. |
| Repeated runs | None are exact repeats. Nearest: same model under different mini-SWE-agent versions / reasoning effort. Used as a noise floor (see §4). |
| Differing harnesses | The biggest confounder. Two clean views: (a) a same-harness cohort of 11 models all run through mini-SWE-agent v2.0.0 on 2026-02-17; (b) the full pool, where a "system" = scaffold + model. |
| Model version ambiguity | Model tags are free text and sometimes missing (e.g. OpenHands + Opus 4.5 has no model tag). I tag families with a regex over name + tag. Some systems use several models or multiple attempts internally. |
| Time drift | Submissions span 2024-12 to 2026-09. Newer models could in principle have seen the tasks; no evidence of it (see §5). |

## 3. Phenomenon 1: the fallback ranking is not the global ranking

### Same-harness cohort (11 models, one scaffold): effect is weak
With default A = Claude 4.5 Opus (high) (76.8%, 116 failures), rescue rate P(B succeeds | A fails) tracks the global
ranking closely: Spearman(global rank, rescue rank) = **0.90–0.95** for every choice of default. The strongest
"inversions" have bootstrap probability ≈ 0.8 on ~120 failures. **Not demo-worthy.** Failures are dominated by
shared difficulty: of 500 tasks, 59 are solved by nobody in the cohort and 211 by everybody.

### Full pool (systems): effect is large and robust
Default A = **live-SWE-agent + Claude 4.5 Opus**, #1 on the leaderboard at 79.2%, fails 104 tasks.

| Backup system | Global | Global rank | Rescue rate P(succ \| A fails) | 95% CI | Rescue rank (of 39) |
|---|---|---|---|---|---|
| Sonar Foundation Agent + Claude 4.5 Opus | 79.2 | 1 | 17.3% | 9.6–24.0 | 17 |
| TRAE + Doubao-Seed-Code | 78.8 | 2 | **30.8%** | 22.1–40.4 | **1** |
| **OpenHands + Claude Opus 4.5** | 77.6 | 3 | **9.6%** | 4.8–15.4 | **39 (last)** |
| live-SWE-agent + Gemini 3 Pro | 77.4 | 4 | 21.2% | 13.5–29.8 | 10 |
| EPAM AI/Run + Claude 4 Sonnet | 76.8 | 6 | 24.0% | 16.3–32.7 | 5 |
| Atlassian Rovo Dev (Sonnet 4 + GPT-5) | 76.8 | 6 | 25.0% | 16.3–33.7 | 3 |
| Claude 4.5 Opus (high), mini-SWE-agent | 76.8 | 6 | 14.4% | 7.7–21.2 | 30 |
| Gemini 3 Flash (high), mini-SWE-agent | 75.8 | 9 | 21.2% | 13.5–29.8 | 10 |
| Warp (GPT-5 + Sonnet 4) | 75.6 | 11 | 22.1% | 14.4–29.8 | 8 |

Contingency tables against A (both / A-only / backup-only / neither):
- OpenHands + Opus 4.5: 378 / 18 / **10** / 94 → oracle union 81.2%
- TRAE + Doubao: 362 / 34 / **32** / 72 → oracle union 85.6%

**The clean aha:** the #3 system is the *same model* as the default in a different scaffold. It fails on the same
tasks, so it rescues 10 of 104 failures. Systems built on a different model family, ranked 6th–11th, rescue 2–3×
more. Spearman(global, rescue) in the full pool drops to **0.46** for this default (0.5–0.8 for others).

Generalization: for all four Claude-4.5-Opus-based defaults in the top 10, same-family backups rescue less
(13–17%) than different-family ones (18–21%), and the best-by-leaderboard backup rescues 13–17 points less than
the best-by-fallback backup. For 5 of the top 10 defaults the gap is ≥ 10 points.

Stability: on 2,000 random 80% task subsets, "OpenHands + Opus rescues fewer than Gemini 3 Flash / TRAE Doubao"
holds in **100%** of draws, while "OpenHands + Opus is better globally than Gemini 3 Flash" holds in 98%.

## 4. Noise floor (important caveat for the demo)

The same model re-run under a different harness version rescues a surprising fraction of its own failures:

| Pair (same model) | Agreement | P(second run succeeds \| first fails) |
|---|---|---|
| Claude 4.5 Opus medium v1.16 vs high v2.0 | 90.0% | 24.2% |
| GPT-5.2 high v1.17.2 vs v2.0 | 87.8% | 23.4% |
| Claude 4.5 Sonnet v1.13 vs high v2.0 | 87.2% | 23.1% |

So a ~20% rescue rate is roughly what "just try again" buys. The demo should draw this as a reference line: a
backup has to beat *retrying the default*. (Caveat: these are not pure repeats; reasoning effort and harness version
differ.) OpenHands + Opus 4.5's 9.6% is *below* this floor, which is consistent with the #1 system already being a
near-superset of what Opus 4.5 can do.

## 5. Contamination / oddity check
Gemini 3.5 Flash (2026-09-01, newest) rescues 27% of A's failures at 71.8% global. It solves only 2 tasks no other
submission solves (Claude 4.5 Opus mini: 0), and its rescue rate rises smoothly with how many other systems solve the
task (9% on tasks ≤5 others solve, 78% on tasks ≥61 others solve). Nothing suspicious, but it also reports only
441/500 instances, so it is excluded from headline claims.

## 6. Phenomenon 2: a good failure predictor can be a bad router (real signals)

Default A' = Claude 4.5 Opus (high) via mini-SWE-agent (76.8%), the top system with per-instance trajectory
features. Three **real** failure predictors for A':

| Predictor | Kind | AUC for "A' fails" |
|---|---|---|
| Official difficulty annotation (<15 min / 15 min–1 h / 1–4 h / >4 h) | pre-run | 0.686 |
| A's own trajectory length (`api_calls`) | post-run (cascade) | 0.731 |
| "Difficulty oracle" = fraction of *other* systems that fail the task | oracle, not deployable | **0.946** |

Route flagged tasks to a backup; end-to-end success (A' alone = 76.8%):

| Backup (global, rescue) | oracle union | trajectory-length router (flag 30%) | difficulty-oracle router (flag 30%) | annotation router (flag "≥1–4 h", 45 tasks) |
|---|---|---|---|---|
| MiniMax M2.5 (75.8, 21.6%) | 81.8 | **78.0** | 76.4 | 77.0 |
| Gemini 3 Flash (75.8, 20.7%) | 81.6 | 76.6 | 76.6 | – |
| TRAE + Doubao (78.8, 34.5%) | 84.8 | **79.6** | – | 77.6 |
| OpenHands + Opus 4.5 (77.6, 19.8%) | 81.4 | 77.6 | 78.2 | 77.6 |

Observations that make the point without any synthetic data:
- The predictor with AUC 0.95 is a *worse* router than the one with AUC 0.73 for MiniMax (76.4 vs 78.0), because
  its true positives concentrate on tasks nobody solves (the "neither" quadrant) and its false positives land on
  tasks A' solves and the backup doesn't.
- The *same* trajectory-length router gains +1.2 with MiniMax but with Gemini 3.5 Flash it *loses* 1.0 (75.8):
  among flagged tasks, harm (A'-only: 17) exceeds gain (backup-only: 12). Routing value is a property of the
  (predictor × backup) pair, not of the predictor.
- With a strongly complementary backup, routing beats *both* systems' leaderboard scores (76.8 and 78.8 → 79.6).
- Break-even precision for a failure detector = P(backup fails | A ok) / (that + rescue rate): 32% for OpenHands+Opus
  (safe but low-yield) vs 38% for Gemini 3.5 Flash (high-yield but risky). Both numbers fall directly out of the 2×2.

Honest framing: absolute gains are +1 to +3 points on 500 tasks, against an oracle ceiling of +5 to +8. That is the
insight, not a weakness: near-peer routing has a small ceiling and a naive failure detector can spend it entirely.

## 7. Reasons to abandon / risks
- **None decisive.** The only real weakness is that the model-only (same-harness) version of the story is not
  supported; the demo must be framed at the system level, which is also what a production router actually chooses
  between.
- Single runs per system; rescue rates on ~100 failures have ±8 point CIs. The demo should show CIs and the
  retry-noise floor rather than hide them.
- A reviewer could ask "isn't this just ensemble diversity?" Answer: yes, and the non-obvious part is that leaderboard
  rank is actively misleading about it, and that failure-prediction metrics (AUC/precision/recall) don't measure it
  either.

## 8. Recommendation: build Idea 1. Minimum viable demo

A single static page (no backend, bundled 307 KB JSON), Theme 4 (Evaluation & Data Quality), two acts:

**Act 1 – "The fallback leaderboard" (must ship).** Choose a default system (preset: the #1 system). A slope chart
links each system's global rank (left) to its rescue-rate rank (right), with CI bars and a dashed "retry the
default" noise-floor line; systems coloured by model family. The crossing lines are the aha; hovering a system shows
the 2×2 contingency and oracle union. One sentence at the top: *"A leaderboard scores models on all tasks. A
fallback is only ever used on the tasks the default failed."*

**Act 2 – "Predicting failure ≠ predicting the value of switching" (ship if under budget).** Default fixed to the
mini-SWE-agent Opus 4.5 run. Pick a backup and a predictor (annotation / trajectory length / difficulty oracle) and
drag a flag-rate slider. Two side-by-side gauges: the classifier view (precision, recall, AUC) and the system view
(end-to-end success), with the flagged tasks split into harm / gain / neutral. The reviewer sees the AUC-0.95
predictor lose to the AUC-0.73 one. Closing line: *"Predict the value of switching, not the failure of the default."*

Estimated build: 3–5 hours including deploy (GitHub Pages / Vercel). Drop Act 2 first if over the 8-hour cap.
Explicitly out of scope: cost/latency, live model calls, training a real router.

## 9. Addendum (same-harness cohort only, cheap primaries) — 2026-09-07

Scope restricted to mini-SWE-agent v2.x (`analysis/cohort_v2.py`, `analysis/cheap_primaries.py`).

**Gemini 3.5 Flash coverage.** Its 59 missing tasks cluster by repo (33 matplotlib, 10 xarray, 9 pylint, 6 sklearn)
and 28 of them are solved by ≥9 of the other 12 models: environment failures, not hard tasks. On the 441 tasks it
ran it scores 81.4% (cohort #1), not 71.8%. Treating missing as failures understates both its global score and its
rescue rate by 6–10 points (e.g. Haiku default: 42.5% vs 50.7% on covered tasks). Its earlier appearance as a
"low-ranked model that rescues a lot" was an artifact. Correct handling: restrict to the 441 covered tasks whenever it
is included, or exclude it.

**Cheap primaries (Haiku 4.5, Sonnet 4.5, GPT-5 mini).** Both variants (11 models/500 tasks; 12 models/441 tasks):
Spearman(global, rescue) = 0.88–0.96. No backup moves more than 3–4 rank positions, every move is within
overlapping 95% CIs (≤5 rescued tasks apart), and no pair satisfies "≥2 points better globally but rescues fewer"
with bootstrap probability above 0.55. **Phenomenon 1 does not hold within a single harness.** The GPT-5.2 Codex
v2.0.0 submission has no per-instance file in the repo and could not be included.

## 10. Addendum: model × task-family interaction (same-harness cohort) — 2026-09-07

`analysis/task_families.py`, figure `data/interaction_heatmap.png`. Cohort = mini-SWE-agent v2.0.0, 11 models, 500 tasks.

**Repository as task family** (8 repos with n≥15; django 231, sympy 75, sphinx 44, matplotlib 34, sklearn 32,
astropy 22, xarray 22, pytest 19). After removing model and repo main effects, the remaining interaction
sum-of-squares is 1620 vs a permutation-null mean of 1733 (95th pct 2507), **p = 0.56**: the apparent per-repo
specialisation (GLM 5 best on matplotlib, GPT-5.2 best on pytest, MiniMax best on xarray) is exactly what shuffling
repo labels produces. In the rescue view, the global-best backup is within 1–3 rescued tasks of the top rescuer in
every repo for all three cheap defaults and for Opus 4.5.

**Difficulty tier as task family.** The interaction test rejects additivity (p<0.001), but the residual pattern is
monotone in model strength (weak models lose disproportionately on hard tasks: a ceiling/floor compression effect),
not specialisation. On the 42 "1–4 hour" tasks the best model is Claude 4.6 Opus with 18 solved vs 15 for Claude
4.5 Opus: 3 tasks, inside noise.

**Conclusion.** No model family outshines another on any task family in this cohort. Combined with §9, the
same-harness data supports only "success is a monotone function of task difficulty and model strength".
Idea 1 is not viable in the form the user wants; fall back to Idea 2.

## 11. Addendum: cross-domain tasks (LiveBench) — 2026-09-07, 30-minute time-box

Source: HF `livebench/model_judgment` (per-question scores, turn 1). Scored categories available without re-running
LiveBench's graders: **language** (typos, plot_unscrambling, connections, paraphrase; 240 q), **coding** (LCB_generation,
coding_completion; 128 q), **instruction_following** (50 q in the common set). Math, reasoning and data_analysis
answers exist in `livebench/model_answer` (368/150/150 q) but need re-scoring with the pure functions in
`LiveBench/livebench/process_results/*` against the HF ground truth: feasible, ~30 min, not done yet. 57 models cover
the common 418-question set; analysis uses the top 25 by mean score (models are Feb–Mar 2025 vintage).
Scripts: `analysis/livebench_families.py`, `analysis/livebench_headline.py`; data: `data/livebench_matrix.csv`,
`data/livebench_scores.json`.

**Model × category interaction is large and real.** Interaction sum-of-squares after removing model and category
main effects: 5,282 vs permutation-null mean 566 (95th pct 876), p < 0.001; at task level 17,686 vs 1,886. This is
the opposite of the SWE-bench same-harness result (§10, p = 0.56).

Examples of rank within category (of 25): o3-mini high is #2 coding / #12 language; qwq-32b #3 coding / #18 language;
Claude 3 Opus #25 coding / #8 language (#2 on typos); DeepSeek V3-0324 #1 instruction following / #15 language;
GPT-4.5 #4 coding / #21 instruction following. 29 pairs satisfy "X ≥2 pts better globally, Y better on a category
with bootstrap P ≥ 0.99".

**Conditional leaderboard flips by category.** Default Claude 3.5 Sonnet (59% on the binarised score ≥0.5, 173
failures: 44 coding, 106 language, 23 IF). Excluding Gemini 2.5 Pro, which dominates everywhere:
- coding failures: qwq-32b 61% [48–75], o3-mini high 57% [43–73] (global #9, #8) vs Claude 3.7 thinking / o1 43% (global #2, #3)
- language failures: o1 47% [38–57], Claude 3.7 thinking 46% vs o3-mini high 30%, qwq 28%
- IF failures: hunyuan-turbos 65% [43–83] (global #18) vs GPT-4.5 30% (global #4); n=23, wide CI.
Same pattern with default Gemini 2.0 Flash (coding: o3-mini high 68% vs o1 47%; language: reverse).

**Caveats.** A strictly stronger model (Gemini 2.5 Pro) is the best backup in every category, so the flips live
among near-peers and cheap models, which is where routing actually happens. IF has only 50 common questions. Models
are early-2025; the story is about the structure, not these specific models. Binarising at 0.5 is a choice; the
interaction test uses raw scores.

**Conclusion.** Phenomenon 1 holds on a cross-domain benchmark: the best fallback depends on the task family, and
global rank misleads within families. Idea 1 is back on the table with LiveBench as the data source.
