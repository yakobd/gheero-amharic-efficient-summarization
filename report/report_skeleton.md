# Quality-Filtered Diverse Sampling: Data-Efficient Amharic Summarization

**Gheero Applied AI & ML Residency — Individual Research Assignment**
**Research Area: Data and Compute-Efficient Generative AI**

---

## Abstract

*[TODO: write last, ~150 words. One sentence each on: problem, method,
result, implication. Cannot be written until results are in.]*

---

## 1. Research Overview

Recent progress in generative AI has largely come from scaling model size, dataset volume, and compute. This is expensive, hard to reproduce, and inaccessible for most researchers and organizations — particularly those working on low-resource languages, where large annotated datasets often don't exist at all. This project investigates the opposite direction: **can a small model reach most of its achievable performance using a fraction of the training data, if that data is chosen intelligently rather than randomly?**

We study this question through Amharic abstractive summarization — a task and language combination with genuine, practical data scarcity (the largest public dataset, XL-Sum, contains only ~5,700 Amharic article-summary pairs), making data efficiency not just an academic exercise but a real constraint researchers in this space face.

## 2. Research Question and Hypothesis

**Research Question:** Can a quality-and-diversity-based data selection method achieve summarization performance comparable to full-dataset fine-tuning, while training on a significantly smaller subset, for adapting a small language model to Amharic summarization — and does it meaningfully outperform random subset selection at the same data budget?

**Hypothesis:** A subset of Amharic document-summary training data, selected using combined quality (learnability + fidelity) and diversity (topical coverage) criteria, will achieve summarization performance closer to full-dataset fine-tuning than a randomly selected subset of equal size.

## 3. Literature Review Summary

*(Full review in `report/literature_review.md`; summarized here.)*

Data-efficient training is an active research area, with results such as Alpagasus (17% of data, comparable instruction-tuning performance) and S2L (11% of data for fine-tuning) establishing that intelligent subset selection can approach full-dataset performance in various domains. D2 Pruning (Maharana et al.) established the core principle motivating this project's method: quality and diversity are complementary, not substitutable, selection signals — optimizing for one alone systematically biases the resulting subset.

This principle has not previously been tested on low-resource-language summarization. Existing data curation practice in African-language NLP relies on quality-only filtering (e.g., similarity thresholds), without an explicit diversity-sampling mechanism, and without an ablation isolating each signal's individual contribution. One directly comparable applied project (Mekuriaw, 2024, IA3 PEFT fine-tuning of mT5-small for Amharic summarization) found that dataset refinement measurably affected output quality — but without a systematic, ablated selection methodology.

**Gap addressed by this project:** a controlled evaluation of combined quality-and-diversity data selection, on Amharic abstractive summarization, benchmarked against both a random-selection baseline and a full-dataset ceiling.

## 4. Methodology

### 4.1 Task, Dataset, and Model

- **Task:** Abstractive summarization (Amharic)
- **Dataset:** XL-Sum (Amharic subset) — Hasan et al., 2021. 5,761 raw article-summary pairs; after a length-outlier pre-filter (removing the top ~1% by word count, following inspection of the raw distribution), **5,704 training examples**, 719 validation, 719 test (XL-Sum's own predefined splits, unmodified).
- **Model:** mT5-small (`google/mt5-small`) — chosen for direct comparability to the XL-Sum paper's own published Amharic baseline (ROUGE-1/2/L: 15.33/5.12/13.85, low-resource single-language fine-tuning setting), and because it is small enough to run the full experiment matrix within the project's time constraints.

### 4.2 Proposed Method: Quality-Filtered Diverse Sampling

The proposed selection method combines two complementary signals:

1. **Quality filtering (two-stage):**
   - **Q1 — Learnability:** each training pair is scored by the loss a *pretrained, not fine-tuned* reference mT5-small assigns to the true summary given the article. Examples are kept if their loss falls within the [20th, 80th] percentile band — this excludes both near-trivial pairs (very low loss) and likely noisy/misaligned pairs (very high loss).
   - **Q2 — Fidelity:** a word-overlap ratio between summary and article, filtering out summaries with low grounding in their source article — addressing a known XL-Sum data-quality issue (a documented fraction of XL-Sum summaries contain information not inferable from the article).

2. **Diversity sampling (D1):** remaining examples are embedded using LaBSE (a multilingual sentence embedding model confirmed to support Amharic), clustered via k-means (20 clusters), and sampled proportionally across clusters — prioritizing higher quality-scored examples within each cluster — so the selected subset covers the topical space rather than concentrating in whatever topic is most frequent in the raw data.

### 4.3 Baselines

| Condition | Description | Purpose |
|---|---|---|
| Full dataset | All 5,704 training examples | Upper-bound ceiling |
| Random 25% | Uniform random sample, no scoring | Naive baseline |
| Combined 25% | Quality-Filtered Diverse Sampling (proposed method) | Test condition |

*[Scope note: the original design included quality-only and diversity-only ablation conditions at both 25% and 50% budgets, to isolate each signal's individual contribution. Given the project's time constraints, the core comparison (full dataset vs. random vs. combined, at 25%) was prioritized as the minimum viable test of the central hypothesis. Quality-only and diversity-only ablations remain a stretch goal / follow-up direction — see Section 7, Limitations.]*

### 4.4 Evaluation Metrics

- **Performance:** ROUGE-1, ROUGE-2, ROUGE-L (standard for summarization; matches the XL-Sum paper's own reported metrics for direct comparability)
- **Efficiency:** training data size, wall-clock training time, peak GPU memory usage

### 4.5 Training Configuration

All three conditions use identical hyperparameters (fixed seed = 42) to ensure the comparison isolates the effect of data selection, not confounding training differences:

- 6 epochs, batch size 2, gradient accumulation 2 (effective batch size 4)
- Learning rate 5e-5, weight decay 0.01, warmup ratio 0.1
- Full fp32 precision *(note: fp16 was initially used but found to cause a known T5/mT5 numerical instability issue — see Section 7)*
- Max input length 512 tokens, max target length 128 tokens

## 5. Results

*[TODO: populate once all 3 runs complete. Include:]*

*[TODO table: ROUGE-1/2/L for full_dataset, random_25pct, combined_25pct]*

*[TODO table: training time, peak GPU memory for each of the 3 runs]*

*[TODO: figure — bar chart comparing ROUGE scores across the 3 conditions]*

*[TODO: how do full_dataset's numbers compare to the XL-Sum paper's published baseline (15.33/5.12/13.85)? This is an important sanity check to report explicitly.]*

## 6. Analysis

*[TODO, written after Section 5 is populated. Should address:]*
- *[Does combined_25pct outperform random_25pct? By how much, on each metric?]*
- *[How close does combined_25pct get to full_dataset's ceiling, using only 25% of the data?]*
- *[Efficiency tradeoff: how much training time/compute was saved for how much (if any) performance cost?]*
- *[Honest discussion: does the result support, partially support, or fail to support the hypothesis?]*

## 7. Limitations

- **Scope reduction:** due to time constraints, the full planned ablation matrix (5 conditions × 2 data budgets = 9 runs) was reduced to the 3 core runs described in Section 4.3. This means the individual contributions of the quality-filtering and diversity-sampling components cannot be separated in this report — only their combined effect is measured against the two baselines.
- **Single data budget tested:** only the 25% budget was evaluated; the original design also planned a 50% budget to observe how the selection advantage changes as the budget grows. This would be valuable follow-up work.
- **fp16 instability discovered mid-project:** an initial training run using fp16 mixed precision produced a model with near-zero ROUGE scores despite normal-looking training loss — a known T5/mT5 architectural issue where internal layers overflow under fp16's limited numeric range. All reported results use full fp32 precision after this was identified and corrected.
- **Small dataset:** XL-Sum's Amharic subset (5,704 training examples) is itself small by NLP standards; results may not generalize to larger low-resource datasets without further testing.
- **Single reference model architecture:** all quality scoring uses mT5-small itself as the reference model; results might differ with a different or larger reference model for scoring.

## 8. Conclusion

*[TODO, written last, after Analysis. Should directly answer the research question and hypothesis stated in Section 2, using the actual evidence from Section 5-6.]*

## References

See `report/literature_review.md` for the full reference list.

---

## Appendix: Reproducibility

- Full code, config, and experiment logs: `github.com/yakobd/gheero-amharic-efficient-summarization`
- All 3 runs use a fixed random seed (42) and identical hyperparameters (Section 4.5)
- Raw per-run metrics: `results/metrics/all_runs_summary.csv`
