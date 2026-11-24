# Phase 5: Evaluation, the promotion gate and the learning loop

Until phase 5 every number came from the dev set, the same documents used for tuning. Phase 5
adds a frozen golden set, an evaluation harness, a promotion gate that lets a new version go
live only on a measured win, and the loop that turns reviewer corrections into candidates.

```
reviewer corrections (append-only) ─▶ correction pool (same tenant, golden-checked)
         │                                     │
         ▼                                     ▼
  learned scorer candidate             few-shot candidate (k similar layouts)
         └──────────────┬──────────────────────┘
                        ▼
   eval on frozen golden set (cached OCR + replies) ─▶ paired bootstrap ─▶ gate
                        │                                                   │
                  eval_runs + eval/runs/*.json          pass: promote, old retired
                                                        fail: candidate rejected + report
```

## 1. Three datasets, strictly separated

| Dataset | Used for | Never used for |
|---|---|---|
| golden (27 invoices) | the headline result, the gate | prompts, examples, thresholds, training |
| dev (41 invoices) | tuning OCR, calibration, scorers, thresholds | being reported as the result |
| correction pool | few-shot examples, training rows | evaluation |

The golden set covers every layout × noise combination three times (seed 2027, never used
elsewhere). The repo commits only `eval/golden/manifest.jsonl` (ids, hashes, labels):
`scripts/golden.py build` regenerates the documents and checks each SHA-256, and a deleted
and rebuilt copy matched byte for byte. `assert_disjoint` is called wherever data could cross
the line (the dev sets, the correction pool, scorer training, simulated review) and refuses to
continue if a golden hash shows up.

## 2. Metrics, kept per document

- **Field F1** after normalisation: a wrong value is both a false positive and a false negative.
- **Line items aligned before scoring** (description similarity, plus a bonus when amounts
  agree), so one missed row is one error, not a cascade (tested).
- **Straight-through rate**: documents with every required field and every row exactly right.
- **Routing**: auto-approve rate and precision, catch rate.
- **Calibration error (ECE)**, cost per page, p95 latency.

Metrics are stored per document, which is what the bootstrap needs.

## 3. The gate

Both versions run on the same golden documents, and every difference is judged by a **paired
bootstrap over documents**: 1,000 resamples of the documents with replacement (the same sample
for both versions), with the 95% interval of the difference. All must hold:

- **Accuracy**: macro F1 gains (interval above 0), or, for a change that only touches routing,
  the auto-approve rate gains while F1 does not fall. *(Extension of the design doc: a scorer
  change leaves every value unchanged, so it can never win on F1.)*
- **Safety**: auto-approved fields stay at or above 99.5% precision, and the catch rate
  does not fall.
- **No critical regression** on total, invoice_date, vendor_name, currency.
- **Cost** within budget; **p95 latency** within the active version's plus 3 s.

`python -m docflow_core.eval compare <active> <candidate> --decide` promotes on a pass (the old
version retired in the same transaction) or marks the candidate `rejected` with the report.

## 4. Fair and cheap evaluation: two caches

- **OCR cache** by file hash: OCR is not part of an extractor version.
- **Reply cache** keyed by the exact request (model, messages, schema, tools). Re-running costs
  nothing, and two versions that send identical prompts get identical replies, so a
  scorer-only candidate is compared on exactly the same extractions. Cached replies keep the
  original tokens, cost and latency, so those metrics stay honest. Empty replies are never
  cached.

## 5. Baseline: version 3 on the golden set

The first number in the project that nothing was tuned on:

| | Golden (27) | Dev (41), for comparison |
|---|---|---|
| straight-through documents | **26 / 27 (96.3%)** | 24 / 41 |
| macro field F1 | 0.965 | — |
| line-item rows exactly right | 83 / 84 | — |
| wrong fields sent to review | **11 / 11** | 18 / 18 |
| auto-approved fields that are right | **100%** | 100% |
| fields auto-approved | 87.9% | 85.3% |
| calibration error (ECE) | 0.020 | — |
| cost per page / p95 latency | $0.00066 / 13.6 s | |

The golden numbers sit close to the dev numbers, so the tuning did not overfit. **10 of the 11
wrong golden fields are tax IDs** misread by OCR. Every one is caught by the GSTIN check and
sent to review, most with a correct repair suggestion.

## 6. Candidate 1: a learned confidence scorer, rejected

Phase 4 showed the hand-made score sends provably correct fields to review. `fit_scorer.py`
fits a logistic regression on the dev set's evidence. The training label is whether the value
the extractor **originally** produced was right, recovered from each correction's `old_value`:
the append-only audit trail is what makes re-learning possible. Its weights are readable:

| Feature | Weight (log-odds) |
|---|---|
| a rule failed | **−3.67** |
| OCR confidence | +2.27 |
| rules passed | +1.38 |
| second model agrees | +0.59 |
| value absent (vs exact match) | +0.08, no longer penalised |

On the golden set it auto-approved **98.3%** of fields instead of 87.9%, a gain of
**+10.4 points with a 95% interval of [+5.6, +17.7]**. But it auto-approved one wrong
line-item quantity, so the catch rate fell from 100% to 90.9%:

```
[PASS] accuracy: macro F1 +0.0000; auto-approve rate +0.1036 [+0.0555, +0.1773] -> routing gain
[FAIL] safety: auto-approve precision 0.9982 (target 0.995), catch rate 0.909 vs 1.000
gate: FAILED -> candidate marked rejected; version 3 stays active
```

This is the gate doing its job: a large, statistically clear convenience gain was refused
because it would have let one error through. It was **not** refitted until it passed: every
retry against the golden set slowly turns it into a training set. The next scorer should be
fitted on a larger dev set (more of the rare wrong-but-confident cases) and judged once.

## 7. Candidate 2: few-shot examples from corrections

**Correction pool.** Documents a reviewer approved after touching at least one field, with
their OCR text and final values. Reviews here came from `simulate_review.py`, which clears the
queue through the real API using the dev labels: 38 documents, 129 accepts (positive labels)
and 18 edits, exactly the 18 wrong fields routing had caught.

**Selection.** A layout signature: OCR lines with digits replaced by 9, as character 3-grams,
compared by cosine similarity. It finds same-template documents without a model; leave-one-out
on the pool picked a same-template neighbour **39/39** times.

**Safeguards:**

- **tenant isolation**: examples come only from the same tenant's pool, so one company's
  invoices never appear in another company's prompt;
- **leakage guard** on every pool;
- the version's `prompt_hash` stays the base template, while the examples used are recorded per
  document;
- the second opinion keeps the plain zero-shot prompt, so it stays an independent check and
  costs nothing extra.

**Result: pending.** The golden run of this candidate (version `dfa375a1`, k=2, same
calibration and threshold as version 3) reached 14 of 27 documents, all straight-through,
including `syn-2027-0004`, the document whose wrong quantity sank the scorer candidate. It
then hit the free LLM tier's daily token limit. Completed replies are cached, so the run
resumes where it stopped. The gate verdict will be recorded here once it finishes.

## 8. Gaps found and fixed in this phase

- **A correction could quietly approve an unverified document.** Re-routing after a correction
  rebuilt issues from the rules and per-field checks and dropped the stored document-level
  "reference checks unavailable" issue. The regression test fails without the fix and passes
  with it.
- **Document-level issues could block approval forever.** Reviewers can now explicitly
  acknowledge them; the acknowledgement is stored in the corrections audit trail.

## Deviations from the design doc

- The gate accepts a routing-only gain with an extra safety condition (catch rate must not fall).
- Few-shot pools are per tenant (privacy), and evaluation borrows the dev tenant's pool.
- The second opinion always reads the zero-shot prompt.
- Optional LoRA fine-tuning is not built. On the golden set 10 of the 11 wrong fields are OCR
  misreads of tax IDs, not model mistakes, so a fine-tuned extractor would have little to fix.
