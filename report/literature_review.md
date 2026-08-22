# Literature Review: Data-Efficient Training via Quality-Filtered Diverse Sampling

## 1. Overview

This review situates the project's proposed method — combined quality-and-diversity subset selection for Amharic summarization fine-tuning — within three overlapping bodies of work: (1) data-efficient LLM training more broadly, (2) methods that explicitly combine quality and diversity signals for subset selection, and (3) data practices specific to low-resource and African-language NLP. The goal is to show precisely where existing work stops and where this project's contribution begins.

## 2. Data-Efficient LLM Training: The Broader Landscape

Data efficiency has become a central concern in LLM research as the field has recognized that "many training examples contribute little to final performance" (Market-Driven Subset Selection for Budgeted Training, 2025), motivating a shift from scaling data volume to curating data value. A recent systematic survey frames this as a "data value flywheel" spanning data selection, quality enhancement, synthetic generation, and distillation (A Survey on Efficient LLM Training: From Data-Centric Perspectives, 2025). Within data selection specifically, the survey distinguishes static filtering (offline scoring of inherent data properties) from dynamic selection (adapting during training) — this project's method falls in the static-filtering category, which is the more tractable and reproducible choice for a short, controlled research window.

Concrete results from this space are striking: Alpagasus achieves comparable instruction-tuning performance using only 17% of original data via complexity-based filtering; MASS reduces pretraining tokens by 50-70% while matching full-data performance using skill-graph-guided quality scoring; S2L (SmallToLarge) reduces fine-tuning data requirements to 11% of the original dataset for mathematical problem-solving, and, notably, also outperforms full-data training using only 50% of data on a clinical summarization task (MIMIC-III). These results establish the general plausibility of this project's hypothesis — that a well-selected subset can approach or match full-dataset performance — across multiple domains, though none apply this specifically to summarization in a low-resource language.

## 3. Combining Quality and Diversity Signals

The most directly relevant prior work is **D2 Pruning** (Maharana et al., 2023/2024, ICLR), which explicitly argues that "data diversity and importance scores are two complementary factors that need to be jointly considered during coreset selection" — optimizing for diversity alone biases a coreset toward easier samples, while optimizing for difficulty alone omits easy samples needed for stable training. D2 Pruning formalizes this via a dataset graph with message-passing to combine both signals, evaluated on vision and general NLP datasets.

This project's proposed method, "Quality-Filtered Diverse Sampling," is motivated by the same core insight — that quality and diversity are complementary, not substitutable, selection axes — but operationalizes it differently and more simply: sequential quality filtering (Q1 learnability + Q2 fidelity) followed by diversity-proportional cluster sampling (D1, via LaBSE embeddings), rather than D2 Pruning's joint graph-based formulation. This is a deliberate simplification suited to a short research window and small dataset, at the acknowledged cost of not capturing interaction effects between quality and diversity as elegantly as a joint graph method would.

Other related approaches include RL-Guided Data Selection (2025), which uses reinforcement learning to select subsets and reports matching full-dataset performance with just 5% of data on some tasks, and XMAS, which clusters examples by attention-trajectory similarity for vision-language model fine-tuning. Both reinforce that combining multiple signals (rather than relying on a single heuristic) is the emerging consensus approach in this literature — a consensus this project's method is designed to test in a new setting.

One important counter-finding worth flagging honestly: the Cowerage paper (NeurIPS 2023) found that "dataset pruning strategies used in vision tasks for sampling the most informative examples do not perform better than random subset selection" when applied to fine-tuning self-supervised ASR models. This is a useful caution — data selection methods do not universally transfer across modalities and tasks, which is itself part of the motivation for testing this specific combined method on this specific task (Amharic summarization) rather than assuming results from other domains automatically generalize.

## 4. Data Practices in Low-Resource and African-Language NLP

Work specific to African and low-resource languages relies heavily on transfer learning (used in ~44% of surveyed papers per "Charting the Landscape of African NLP," 2025) and, increasingly, on **audit and data filtering** as a recognized technique — the same survey notes that "auditing and filtering have become common and effective approaches — especially for low-resource languages" due to noisy web-scraped source data. However, the filtering described in this literature is typically simple quality-only filtering (e.g., AmaSQuAD's 0.6-similarity-threshold filter for Amharic QA pairs), not combined with an explicit diversity-sampling step, and not evaluated via a controlled ablation isolating each signal's individual contribution.

Directly relevant to this project's exact model and task: an independent applied project (Mekuriaw, 2024) fine-tuned mT5-small for Amharic summarization using IA3 parameter-efficient fine-tuning on a mixed Amharic/Arabic/English dataset, finding that "the disparity in performances between models fine-tuned with different versions of the Amharic dataset highlighted the effectiveness of fine-tuning on a refined dataset" — direct anecdotal evidence that data curation quality measurably affects Amharic summarization outcomes on this exact model family, though without a systematic, ablated methodology for *how* to construct that refined subset.

The original XL-Sum paper (Hasan et al., 2021) itself fine-tuned mT5 on Amharic in a single-GPU, low-resource setting, reporting baseline ROUGE-1/2/L scores of 15.33/5.12/13.85 for language-specific fine-tuning. This serves as the project's external validation reference point for the full-dataset ceiling condition.

## 5. The Research Gap

Synthesizing the above, three gaps converge to motivate this project:

1. **Modality/task gap**: combined quality-and-diversity data selection methods (e.g., D2 Pruning, S2L) are well-established for general-domain LLM fine-tuning and pretraining, but have not been evaluated on abstractive summarization for a low-resource language.
2. **Method gap within low-resource NLP**: data filtering practices in African-language NLP are real and effective but overwhelmingly quality-only (similarity thresholds, audit-based filtering); none of the reviewed work combines quality filtering with an explicit diversity-sampling mechanism, nor isolates each component's contribution via ablation.
3. **Evidence gap for Amharic specifically**: the one directly comparable applied project (IA3 PEFT on mT5-small for Amharic summarization) provides anecdotal support that data refinement matters, but no controlled, ablated methodology or quantified data-efficiency curve (i.e., how selection quality changes as the data budget varies).

This project's contribution is therefore best framed as: **the first controlled evaluation of a combined quality-and-diversity data selection method, tested via explicit ablation (random vs. quality-only vs. diversity-only vs. combined) across two data budgets (25%, 50%), on Amharic abstractive summarization** — a task, language, and combination of rigor (ablation + dual budgets + literature-grounded baseline) not previously reported together in the reviewed literature.

## 6. References

- Chen, L. et al. (2023). *Alpagasus: Training a Better Alpaca with Fewer Data.*
- Du, Q. et al. (2023). *MoDS: Model-oriented Data Selection for Instruction Tuning.*
- Maharana, A., Yadav, P., & Bansal, M. (2023/2024). *D2 Pruning: Message Passing for Balancing Diversity & Difficulty in Data Pruning.* ICLR 2024. arXiv:2310.07931.
- Yang, Y., Mishra, S., Chiang, J., & Mirzasoleiman, B. (2024). *SmallToLarge (S2L): Scalable Data Selection for Fine-tuning LLMs by Summarizing Training Trajectories of Small Models.*
- (2025). *A Survey on Efficient Large Language Model Training: From Data-Centric Perspectives.* arXiv:2510.25817.
- (2025). *Market-Driven Subset Selection for Budgeted Training.* arXiv:2510.02456.
- (2025). *RL-Guided Data Selection for Language Model Finetuning.* arXiv:2509.25850.
- (2023). *Representative Subset Selection for Efficient Fine-Tuning in Self-Supervised Speech Recognition Models (Cowerage).* NeurIPS 2023.
- (2025). *Charting the Landscape of African NLP: Mapping Progress and Shaping the Road Ahead.* arXiv:2505.21315.
- (2025). *AmaSQuAD: A Benchmark for Amharic Extractive Question Answering.* arXiv:2502.02047.
- Mekuriaw, D. (2024). *Amharic IA3 Parameter-Efficient Fine-Tuning.* Medium.
- Hasan, T. et al. (2021). *XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages.* Findings of ACL-IJCNLP 2021.
