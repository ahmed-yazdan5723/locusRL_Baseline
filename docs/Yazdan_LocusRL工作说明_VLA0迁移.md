# Your Role in LocusRL

## 1. The project in one paragraph

LocusRL asks one central question:

> **Where should an LLM enter a reinforcement-learning system: as a reward designer, as the policy itself, or in both places?**

We compare three experimental arms on the same game environments:

- **Reward:** an LLM writes dense reward code; a conventional neural policy is trained with that reward.
- **Policy:** an LLM directly selects actions and is fine-tuned with SFT and multi-turn RL.
- **Hybrid:** the LLM policy is trained with both the true game outcome and an audited dense reward produced by the Reward arm.

The core environments are Connect Four, Leduc Poker, and Goofspiel. A continuous/high-resolution environment such as PettingZoo Simple Tag is a later extension.

---

## 2. Your role in one sentence

> **You own action-representation evaluation, reusable baselines, cost/generalization analysis, and the transfer of VLA-0 techniques into the Policy and Hybrid arms—especially in the later continuous-action environment.**

Your previous VLA-0 work is still useful. The project no longer treats VLA-0 fine-tuning as an independent branch; instead, we use it to answer a more general research question:

> **When an LLM acts as a policy, how should game actions be represented and decoded?**

This is directly related to VLA-0’s main philosophy: do not immediately add a specialized action head; first test whether actions can be represented as ordinary text tokens with suitable data augmentation and inference-time aggregation.

---

## 3. What changed from your previous assignment

### Before

Your previous work focused on applying VLA-0 techniques to one domain-specific fine-tuning pipeline:

- integer action encoding;
- Masked Action Augmentation;
- Ensemble Prediction;
- action-granularity ablations.

### Now

The new project is cross-game and compares different locations of LLM involvement. Therefore:

- VLA-0 is **not the whole project**;
- VLA-0 techniques are **not automatically useful in every game**;
- your first task is to build reliable evaluation and action-representation components;
- the full VLA-0 transfer becomes a focused ablation in the Policy/Hybrid arms, with the strongest test conducted in a continuous or high-resolution action environment.

This change makes your work more important to the paper’s central claim: if Policy performs differently from Reward, we need to know whether the difference is caused by the LLM-policy paradigm itself or merely by a poor action representation.

---

## 4. Your four work packages

## WP1 — Baseline runner and reproducible evaluation

### Goal

Make sure every learned agent is compared against the same opponents, seeds, metrics, and environment versions.

### Tasks

- [ ] Implement or organize one-command runners for:
  - Legal-Random;
  - rule/search agents;
  - CFR/MCTS/NFSP references when available;
  - Prompt-Only LLM;
  - SFT-Only LLM;
  - Outcome-Only GRPO.
- [ ] Use the opponent pool and `GameAdapter` provided by Yuxiang.
- [ ] Record win rate, average return, Elo, illegal-action rate, and worst-opponent performance.
- [ ] Add confidence intervals or standard deviations over seeds.
- [ ] Ensure that every result stores its config, commit, seed, opponent-pool version, and checkpoint path.

### Deliverable

A command such as:

```bash
python eval/run_eval.py --game leduc_poker --agent sft_only --seed 1
```

should run the evaluation and create a versioned JSON/CSV result that can be used directly in paper tables.

### Acceptance criteria

- The same evaluation command works across all three core games.
- Re-running the same checkpoint and seed produces the same result within expected stochastic variation.
- No paper number is manually copied from terminal output.

---

## WP2 — Action-as-text specification and VLA-0 transfer

### Goal

Define how an LLM policy expresses actions, then test whether VLA-0-style techniques improve validity, stability, and performance.

### 2.1 Canonical action serialization

Work with Yuxiang and Xinbo to define one structured schema per environment.

Examples:

```json
{"action": "DROP", "column": 4}
```

```json
{"action": "RAISE"}
```

```json
{"action": "BID", "card": 7}
```

The schema must be:

- easy for the LLM to generate;
- deterministic to parse;
- compatible with legal-action masking;
- stable across SFT, GRPO, and evaluation;
- free from multiple textual forms for the same action.

### 2.2 VLA-0 techniques in discrete games

The three core games mostly have categorical actions. VLA-0 techniques must be adapted carefully:

**Integer/action token encoding**

- Use a canonical token or integer ID for each legal action.
- Compare verbose JSON actions with compact action IDs only if the tokenizer or output length may affect training.
- Do not assume that a numerically larger action is “closer” to another action; action IDs are categorical.

**Ensemble Prediction**

- Sample several candidate actions from the same state.
- For categorical actions, use majority vote, probability aggregation, or value-based reranking.
- **Do not average categorical action IDs.** For example, averaging columns 1 and 7 into column 4 has no strategic meaning.

**Masked Action Augmentation**

- It is only meaningful when the target contains multiple informative tokens or when used as an auxiliary reconstruction objective.
- Do not mask a one-token action and claim it reproduces the VLA-0 setting.
- First run a small pilot: compare standard SFT with action-token corruption/masking and measure legal-action accuracy plus held-out performance.
- If it does not help in discrete games, report that boundary and reserve the full method for continuous actions.

### 2.3 Full VLA-0 transfer in a continuous/high-resolution game

Simple Tag is the planned extension because its action representation can expose a meaningful coarse-versus-fine question.

Possible comparison:

- **Coarse:** discrete macro-actions such as move-left, move-right, move-up, move-down, stay.
- **Fine:** normalized continuous action components encoded as integers, for example mapping each value from `[-1, 1]` to `[0, 1000]` and generating them as text.

Full VLA-0 ablations:

- [ ] Coarse vs. Fine action representation;
- [ ] with vs. without Masked Action Augmentation;
- [ ] single prediction vs. Ensemble Prediction;
- [ ] compact integer text vs. a more verbose structured representation.

### Deliverable

An `action_representation.md` specification plus reusable modules such as:

```text
policy/action_codec.py
policy/action_masking.py
policy/action_ensemble.py
```

and a small ablation report answering:

1. Does the representation reduce illegal actions?
2. Does it improve game performance?
3. Does it improve robustness to unseen states or opponents?
4. What additional inference cost does ensembling introduce?

---

## WP3 — Simple Tag feasibility study

### Goal

Decide whether the continuous-action extension is technically useful and affordable before the team commits to full experiments.

### Your responsibility

- [ ] Work with Yuxiang to confirm that PettingZoo Simple Tag can expose the required observations, legal/action bounds, outcome, seeds, and opponent configuration.
- [ ] Run random and conventional-RL baselines.
- [ ] Define Coarse and Fine text-action formats.
- [ ] Estimate rollout speed, sequence length, training cost, and expected experiment count.
- [ ] Produce a short Go/No-Go report.

### Go criteria

Proceed to full experiments only if:

- the environment can use the common evaluation protocol;
- actions can be encoded and decoded without ambiguity;
- rollout speed is sufficient for at least a small Policy/Hybrid experiment;
- the experiment adds a genuinely new property axis rather than merely another game;
- it does not delay the three core discrete environments.

### No-Go is a valid result

If Simple Tag is too costly or incompatible with the ICLR schedule, record the analysis and move it to the ICML extension plan. You are not expected to force it into the ICLR version.

---

## WP4 — Cost, transfer, and figure generation

### Goal

Quantify the main trade-off between Reward and Policy:

- Reward pays an up-front cost to generate/validate reward code, but deploys a small neural policy.
- Policy may adapt and reason better, but pays LLM inference cost at every action.

### Tasks

- [ ] Record environment interactions, training GPU hours, peak memory, model/API calls, and inference latency.
- [ ] Build three comparison views:
  - equal environment interactions;
  - equal training compute/cost;
  - deployment performance vs. inference cost.
- [ ] Evaluate unseen opponents and small rule changes.
- [ ] Generate paper-ready plots from registered result files.

### Main figures you may own

- performance vs. environment interactions;
- performance vs. training cost;
- performance vs. inference latency;
- cross-opponent win-rate matrix;
- Coarse/Fine × Mask/No-Mask × Ensemble/Single ablation;
- action-validity and policy-stability plots.

---

## 5. How you work with other members

| Person | What they provide to you | What you provide to them |
|---|---|---|
| Yuxiang | GameAdapter, opponent pool, fixed seeds, state/action API | action-codec requirements, adapter tests, Simple Tag feasibility feedback |
| Xinbo | Policy/SFT/GRPO trainer and checkpoints | action representation modules, VLA-0 ablations, Policy evaluation and diagnostics |
| Songyan | Reward and Hybrid reward versions | independent evaluation and cost/failure comparison |
| Guchong | paper structure and requested tables | verified result files, plots, captions, method notes |
| Chengyu | scope, priority, research questions | feasibility reports, evidence for claims, blockers requiring decisions |
| Yicheng | occasional high-level review | concise technical questions and milestone summaries |

### Important boundary

You do **not** need to:

- rebuild the common game environments from scratch;
- independently maintain a second GRPO training framework;
- run every Reward experiment;
- decide the final game suite alone;
- complete the continuous-action extension before the core discrete experiments;
- make VLA-0 techniques look successful if the evidence shows they do not help.

Negative results are useful when they define the boundary of action-as-text methods.

---

## 6. Timeline and concrete deliverables

| Deadline | Deliverable | Priority |
|---|---|---|
| July 20 | Read the LocusRL idea/management documents; submit questions and a one-page action-representation draft | P0 |
| July 28 | Baseline runner for at least one core game; canonical action schemas for all three core games | P0 |
| August 5 | Cross-game evaluation runner integrated with the experiment registry | P0 |
| August 8 | Cost and latency logging module | P0 |
| August 12 | Simple Tag + VLA-0 transfer feasibility report | P1 decision gate |
| August 22 | Cross-opponent and rule-transfer evaluation | P0 |
| August 28 | Discrete-game action-representation pilot and Policy-integrity analysis | P0 |
| August 30 | Main statistics and paper-ready figures | P0 |
| After ICLR Go/No-Go | Full continuous VLA-0 ablation, if approved | P1 / ICML extension |

---

## 7. Your first seven days

Please complete the following in order:

1. Read the LocusRL proposal and identify the difference among Reward, Policy, and Hybrid.
2. Meet briefly with Yuxiang to inspect the common `GameAdapter` interface.
3. Meet briefly with Xinbo to inspect the expected Policy input/output format.
4. Draft canonical action schemas for Connect Four, Leduc Poker, and Goofspiel.
5. Write down which VLA-0 techniques are directly applicable, require adaptation, or are not meaningful for each environment.
6. Run one simple baseline/evaluation path on Connect Four.
7. Report blockers and the estimated effort for the Simple Tag feasibility study.

Use the following table in your first report:

| Environment | Action type | Proposed text format | Masking useful? | Ensemble method | Main risk |
|---|---|---|---|---|---|
| Connect Four | categorical | TBD | TBD | vote/rerank | TBD |
| Leduc Poker | categorical | TBD | TBD | vote/rerank | TBD |
| Goofspiel | categorical/simultaneous | TBD | TBD | vote/rerank | TBD |
| Simple Tag | continuous or macro-action | TBD | likely | average/rerank | rollout cost |

---

## 8. Weekly update template

```text
[Yazdan / Date]

Completed:
-

Evidence / links:
-

Current result:
-

Blockers:
-

Next deliverable and deadline:
-

Decision needed from Chengyu / dependency needed from others:
-
```

---

## 9. Final explanation

Your previous VLA-0 work has not been discarded. It has been repositioned from a domain-specific trick into a controlled research question inside a broader paper:

> **Can an LLM represent actions as text across different game types, and when do VLA-0-style masking and ensembling actually help?**

The discrete games let us test legality, structured decoding, strategic stability, and categorical ensembling. The continuous extension lets us test the full VLA-0 hypothesis through integer action encoding, Masked Action Augmentation, and Ensemble Prediction.

Your work is therefore the bridge between the old action-as-text exploration and the new cross-game LLM-as-Policy study. Its success is not defined as “VLA-0 must improve every environment.” Its success is defined as producing a rigorous, reusable, and evidence-based answer about where those techniques work and where they do not.
