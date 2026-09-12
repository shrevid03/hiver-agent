# AI Customer Support Agent — AmazonHelp (Twitter Support)

**Author:** Shreya Vidyadhar Keshatti
**Assignment:** Hiver SDE Intern take-home
**Dataset:** Twitter Customer Support (`twcs.csv`), AmazonHelp brand

---

## 1. Problem and approach

The task is an AI agent that handles inbound Twitter customer-support messages for a brand (AmazonHelp) end to end: understand what the customer wants, decide whether the case needs a human, and draft a grounded, on-brand reply. Rather than a single monolithic prompt, the system is built as a small pipeline of focused components — intent classification, retrieval of similar historical cases, an escalation decision, and reply drafting — each independently testable and measurable. This mirrors how a production support agent is actually built and makes the behaviour auditable: when a reply is wrong, it is possible to say *which* stage failed.

The whole system runs on hosted LLM inference through a provider-agnostic layer, so no model is trained from scratch — the engineering value is in the architecture, the grounding, and the evaluation, not in model weights.

## 2. Data pipeline

The raw `twcs.csv` is a flat table of tweets with author IDs and reply pointers, not conversations. The pipeline reconstructs threads by linking inbound customer tweets to the brand's outbound responses, then filters to a single brand (AmazonHelp) so the retrieval corpus is coherent and the replies share one voice. This yields roughly **8,000 reconstructed customer→brand threads**, which become both the retrieval store and the source pool for evaluation.

Focusing on one brand was a deliberate choice: a mixed-brand RAG store would ground an Amazon reply in, say, a Spotify resolution, which is worse than useless. One brand keeps the grounding faithful.

## 3. Architecture

The agent is four cooperating components behind a shared inference layer.

**Intent classifier** (`src/agent/classifier.py`) — few-shot LLM classification into a 9-class taxonomy, returning structured JSON (`{intent, confidence}`) via the model's JSON mode. The taxonomy was derived by clustering the message corpus and manually naming the clusters, with an explicit `other` catch-all for non-English and out-of-scope messages. (A disambiguation guide targeting the hardest boundaries was later tried and reverted — see §7.)

**RAG store** (`src/agent/rag_store.py`) — sentence-transformer embeddings over the 8,000 historical threads, retrieved by cosine similarity with an *intent boost*: candidates matching the predicted intent are preferred, so a refund query retrieves past refund resolutions rather than superficially similar text. Persisted to disk (pickle) so it is built once.

**Escalation decider** (`src/agent/escalation.py`) — a hybrid: high-precision keyword rules catch the clear cases (legal threats, safety, repeated contact), and the LLM decides the remaining edge cases through the shared retry-backed client.

**Reply drafter** (`src/agent/reply_drafter.py`) — drafts a reply grounded in the retrieved historical resolutions, constrained to Twitter norms: ≤280 characters, warm but concise, no fabricated order details/refund amounts/timelines, and a polite move to DM when private details are needed.

**Provider-agnostic LLM layer** (`src/agent/llm.py`) — a thin factory that exposes an OpenAI-compatible `chat.completions` interface over either Groq or Gemini, selected by environment variable. Every component calls the same interface, so switching the entire pipeline between providers is a one-line config change (this mattered — see §8).

## 4. Evaluation methodology

Evaluation runs against a **200-example golden set**, hand-labelled and stratified across intent, difficulty (easy/medium/hard), and escalation, so the score is not dominated by easy majority-class cases. The harness measures three things independently:

- **Intent classification** — accuracy and macro-F1 (macro so rare intents count equally), plus a full confusion breakdown.
- **Escalation** — precision/recall/F1 against the golden escalation labels.
- **Reply quality** — an **LLM-as-judge** rubric scoring four dimensions (groundedness, tone, completeness, actionability) on a 1–3 scale, run on a 40-example sample. Crucially, the judge scores the agent *against two baselines* on the same inputs: a TF-IDF nearest-reply baseline and a trivial canned reply. This makes the quality result relative and defensible, not just an absolute number the model gave itself.

## 5. Results

### Intent classification (200 examples)

| Metric | Value |
|---|---|
| Accuracy | **84.5%** |
| Macro-F1 | **0.832** |
| Macro-Precision | 0.837 |
| Macro-Recall | 0.841 |

Per-intent F1 (best to worst):

| Intent | Precision | Recall | F1 |
|---|---|---|---|
| return_refund | 0.90 | 0.96 | 0.93 |
| tech_issue | 0.89 | 0.93 | 0.91 |
| delivery_wrong_or_damaged | 0.77 | 1.00 | 0.87 |
| other | 0.77 | 0.95 | 0.85 |
| order_status | 0.95 | 0.76 | 0.84 |
| billing_payment | 0.79 | 0.88 | 0.83 |
| account_access | 0.86 | 0.80 | 0.83 |
| delivery_not_received | 0.92 | 0.72 | 0.81 |
| prime_complaint | 0.69 | 0.56 | 0.62 |

### Escalation

The initial full-eval escalation numbers (P=0.640, R=0.679, F1=0.659; TP=57, FP=32, FN=27, TN=84) were later traced to a **bug**, not genuine behaviour: the escalation decider was still calling the exhausted Groq endpoint while the rest of the pipeline ran on Gemini, and its error handler defaulted every failed call to "escalate" (see §8). So the 32 false positives were mostly force-escalations from a dead API, not the model over-flagging.

After migrating the decider to the working LLM layer with retry-and-backoff and rebalancing the rules (emotional/emphasis cues became LLM *context* rather than instant-escalate triggers), an isolated re-measurement (golden intent as input, to separate the decider from classifier errors) gave:

| Metric | Before (buggy) | After (corrected decider) |
|---|---|---|
| Precision | 0.640 | **0.816** |
| Recall | 0.679 | 0.369 |
| F1 | 0.659 | 0.508 |
| FP / FN | 32 / 27 | 7 / 53 |

The correction eliminated the over-flagging (FP 32 → 7, precision 0.64 → 0.82) but exposed the real limit: the corrected decider is now conservative and *under*-escalates (recall 0.37). Multiple prompt rebalances (across two model backends) failed to lift recall past ~0.37, which points to a structural cause rather than a wording problem.

**Root cause — a definition/threshold mismatch.** The golden set labels **84 of 200 cases (42%) as "should escalate"** — a deliberately liberal threshold reflecting the annotator's policy. A general-purpose LLM asked "does this genuinely need a human?" escalates only the ~16% of clearly hard cases. That 42%-vs-16% gap cannot be closed by prompt wording, because the model has no way to infer one annotator's specific escalation threshold; closing it would require encoding the exact escalation rubric or fitting a decision threshold on labeled data (and hand-tuning to these 200 labels would simply be overfitting). This is the same lesson as the intent ceiling (§7): the metric is bounded by label definition, not model capability. The honest, defensible position is therefore a high-precision decider (0.82) plus explicit future work on recall calibration (§12), with the caveat that the "right" operating point is ultimately a business decision — in support, a missed escalation usually costs more than an unnecessary one.

### Reply quality (LLM judge, n=40, scale 1–3)

| Dimension | Agent | TF-IDF baseline | Trivial baseline |
|---|---|---|---|
| Groundedness | 2.93 | 2.26 | 2.71 |
| Tone | 2.93 | 1.81 | 1.71 |
| Completeness | 2.97 | 1.93 | 2.18 |
| Actionability | 2.97 | 2.11 | 2.54 |
| **Average** | **2.95** | **2.03** | **2.29** |

The agent wins on every dimension and by a wide margin overall (2.95 vs 2.03 and 2.29). This is the strongest evidence in the evaluation that RAG-grounded drafting does real work — the retrieved historical resolutions produce replies that are more grounded, warmer, and more actionable than either baseline.

### Difficulty breakdown

| Tier | n | Intent acc | Escalation acc |
|---|---|---|---|
| easy | 79 | 88.6% | 72.2% |
| medium | 70 | 78.6% | 67.1% |
| hard | 51 | 86.3% | 72.5% |

## 6. Environment / models

The pipeline was developed against Groq (`qwen/qwen3.8-27b`) and then migrated to Google Gemini when Groq's daily token quota was exhausted. The reported intent/escalation/reply numbers above are from a clean run on **Gemini 3.5 Flash-Lite**; the Banking77 generalization check (§9) was also run on Gemini 3.5 Flash-Lite. The LLM judge ran on the same Gemini family.

## 7. Error analysis and improvement

The 31 misclassifications are concentrated, not random. Five boundary confusions account for ~55% of all errors:

| Confusion (gold → predicted) | Count |
|---|---|
| delivery_not_received → delivery_wrong_or_damaged | 6 |
| prime_complaint → billing_payment | 4 |
| account_access → tech_issue | 3 |
| delivery_not_received → prime_complaint | 2 |
| prime_complaint → other | 2 |

The root cause in each case is semantic overlap in the intent definitions: "delivered" appears in both delivery intents; the prime-complaint definition literally mentions "paying," bleeding into billing; and "can't log in via the app" reads as both account and tech.

**Improvement attempted:** a disambiguation guide was added to the classifier prompt with explicit decision rules for each confused pair (e.g. *"never arrived / late / missing → not_received; arrived but damaged, wrong, or a failed delivery attempt → wrong_or_damaged"*) plus a set of **synthetic** contrastive examples, written by hand and deliberately *not* drawn from the golden set (to sharpen the model's reasoning, not to leak test answers).

**Result — an honest negative:** it did not improve accuracy. The refined prompt scored 82.0% (macro-F1 0.804) versus the 84.5% baseline, within run-to-run variance, and the dominant confusion (`delivery_not_received → delivery_wrong_or_damaged`) was completely unmoved at 6 errors. The conclusion is that these residual errors are driven by **genuine label ambiguity**, not prompt clarity: a late Prime order is defensibly both `prime_complaint` and `delivery_not_received`, and a failed delivery attempt is defensibly both `not_received` and `wrong_or_damaged` (the schema files failed attempts under the latter). No prompt fixes a boundary where a reasonable annotator would also disagree — forcing those cases to flip would mean memorizing the specific gold labels, i.e. overfitting the test set. The refinement was therefore **reverted**, and 84.5% is reported as the honest ceiling for this approach on this data. This is recorded as a decision, not hidden.

## 8. Reliability findings (engineering)

Three non-obvious reliability issues surfaced during evaluation, and each materially affected the validity of the numbers:

1. **Silent rate-limit contamination.** The original classifier wrapped its API call in a broad `except` that returned `"other"` on *any* error. Under free-tier rate limiting, every throttled call was therefore scored as a *wrong prediction* rather than a transient failure — accuracy on one run collapsed to ~11% purely from throttling, not model behaviour. The fix was retry-with-backoff that honours the server's `Retry-After` and only falls back after genuine exhaustion. Without this, the eval was measuring the rate limiter, not the model.

2. **Thinking-model token budget.** Gemini 2.5/3.5 models spend output tokens reasoning internally before emitting the answer. A `max_tokens=64` cap (fine for the non-thinking Groq model) truncated the JSON mid-string and every classification failed to parse. Raising the cap to 512 fixed it.

3. **The same silent-fallback bug in a second component.** The escalation decider had not been migrated off Groq, so during the Gemini eval its every LLM call hit the exhausted Groq quota and its `except` defaulted to "escalate" — silently inflating false positives (see §5). This is the same failure pattern as (1) in a different place, and it is why a broad `except: return <default>` is dangerous in an evaluated pipeline: a transient infra failure masquerades as a model decision. The fix was the same — route through the shared retry-backed client and make the hard-failure fallback conservative rather than blanket-escalate.

A further quota gotcha worth recording: on the free tier, `gemini-2.5-flash` allows only **20 requests/day**, versus 500/day for `gemini-3.5-flash-lite` — so model selection had to account for per-model daily caps, not just capability.

The provider-agnostic LLM layer (§3) is what made the Groq→Gemini migration a minutes-long change rather than a rewrite across four files.

## 9. Generalization check — Banking77 (optional)

To test whether the classification approach is domain-agnostic, the same LLM-classification method was pointed at **Banking77** (77 fine-grained banking intents) with zero code changes — only the candidate intent list swapped. On a stratified sample (2 examples per intent, 154 total, zero-shot, Gemini 3.5 Flash-Lite):

| Banking77 (zero-shot, n=154, 77 intents) | Value |
|---|---|
| Accuracy | 74.68% |
| Macro-F1 | 0.721 |
| Unmapped (returned outside the label set) | 2 |

The *same* architecture, aimed at a completely different 77-intent domain with no retraining, reaches **74.7% zero-shot** versus 84.5% on the 9-intent task it was designed for. The gap is reasonable and expected: 77 near-duplicate classes are far harder to separate than 9 broad ones. Every top confusion is a genuine near-synonym pair — `card_arrival → card_delivery_estimate`, `order_physical_card → get_physical_card`, `beneficiary_not_allowed → failed_transfer` — i.e. the model is failing only where the intent boundaries are genuinely fine, not on obvious cases. For reference, published *fine-tuned* models on Banking77 reach the low-90s; low-to-mid-70s from a zero-shot LLM with no training on the domain is a solid generalization result.

## 10. Decision log

1. **Reconstruct threads from flat CSV** — link inbound/outbound tweets into conversations rather than treating rows independently.
2. **Single brand (AmazonHelp)** — a coherent, single-voice RAG corpus; mixed brands would produce mis-grounded replies.
3. **9-intent taxonomy from clustering + manual naming**, with an explicit `other` catch-all for non-English / out-of-scope.
4. **JSON mode + confidence** — structured, parseable classifier output instead of free text.
5. **RAG with intent boost** — retrieve historical cases weighted toward the predicted intent, not just surface-similar text.
6. **Hybrid escalation** — cheap keyword rules first, LLM only for edge cases, to control cost and latency.
7. **Ground replies in historical brand responses** — avoids hallucinated policies, refund amounts, and timelines.
8. **≤280 chars + DM-for-details pattern** — Twitter-appropriate, and structurally prevents fabricating private order data.
9. **LLM-as-judge with two baselines** — quality measured *relatively* (agent vs TF-IDF vs trivial), not as a self-assigned absolute.
10. **Stratified 200-example golden set** — balanced across intent, difficulty, and escalation so the score isn't majority-class-driven.
11. **Provider-agnostic LLM layer** — enabled the Groq→Gemini migration as a one-line change.
12. **Retry-with-backoff** — after discovering rate-limit errors were being silently scored as wrong predictions (see §8).
13. **`max_tokens=512` for thinking models** — reasoning tokens were truncating JSON output.
14. **Error-driven prompt refinement** — a disambiguation guide targeting the measured top confusion pairs, using synthetic (non-leaking) examples (attempted, then reverted — see §7).
15. **`temperature=0`** — deterministic, reproducible classification.

## 11. Limitations

- **prime_complaint (F1 0.62)** is the weakest intent — it genuinely overlaps with both billing and delivery, and some golden labels are arguably ambiguous (a late Prime order is defensibly *both* prime_complaint and delivery_not_received).
- **Escalation is not yet at a good operating point.** After fixing the dead-API bug, the corrected decider is precise (0.82) but under-escalates (recall 0.37), bounded by the label-threshold mismatch in §5. Recall calibration is the clearest remaining improvement.
- **Judge self-preference risk** — the judge is a Gemini model scoring Gemini-generated replies, so the absolute judge scores may be biased upward; they are most trustworthy *relative to the baselines*, which is how they are presented.
- **Single-annotator golden set** — labelled by one person, so there is unmeasured label noise and no inter-annotator agreement.
- **Banking77 sample size** — evaluated on a 2-per-intent stratified sample (n=154), not the full 3,080-example test split.
- **Free-tier rate limits** constrained how fast the evaluation loop could iterate.

## 12. Future work

- Fine-tune a small dedicated classifier on the labelled data for lower cost and latency at inference time.
- **Tune escalation toward recall** — the corrected decider is too conservative (R=0.37); a decision threshold fit on labeled data, or the explicit escalation rubric encoded, should recover recall while keeping most of the precision gain.
- Confidence-thresholded human handoff — route low-confidence classifications to a human instead of guessing.
- Multi-annotator golden set with inter-annotator agreement to quantify label noise.
- Complete the full Banking77 test split and add a per-intent breakdown.
- Async / batched inference to raise throughput within rate limits.
