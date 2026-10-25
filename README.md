# CRCRetinaAiAGE

> **Jovovic I, Popovic T, Rubezic Lukic I, Babic D, Popovic N.** Deep Learning Discrimination of Colorectal Cancer using Retinal Fundus Photographs: UK Biobank Case-Control Study. *Current Medical Imaging*. 2026;22:e15734056494970. DOI: [10.2174/0115734056494970260902110339](https://doi.org/10.2174/0115734056494970260902110339)


## Files

| File | Purpose |
|---|---|
| `config.json` | Experiment configuration. |
| `train_crc_b3.py` | Data audit, participant-level split, EfficientNet-B3 training, evaluation. |
| `covariates_baseline.py` | Age/vascular logistic-regression baseline and cohort statistics. |

## Experiment

- **Data:** UK Biobank (application 92804); 1,600 right-eye fundus photographs from 1,600 participants (800 CRC, 800 non-CRC) after manual quality review; 2048 × 1536 px.
- **Split:** participant-level stratified 70/15/15 (1,120 / 240 / 240), seed 42.
- **Preprocessing:** grayscale replicated to 3 channels, scaled to [0,1], resized to 1024 × 768; no ImageNet normalization.
- **Sampling:** 1,120 training images → 2,240 views (original + horizontal mirror) → 2,800 class-balanced draws per epoch with replacement.
- **Model:** `tf_efficientnet_b3.in1k` (ImageNet), all layers trainable, dropout 0.3, 2 output logits.
- **Training:** AdamW (lr 1e-4, wd 1e-4), cosine schedule with 2 warm-up epochs, cross-entropy with label smoothing 0.1, batch 8 × 4 gradient accumulation, FP16, max 30 epochs, early stopping on validation loss (patience 5).
- **Evaluation:** threshold by Youden index on validation, applied unchanged to test; AUROC (Hanley–McNeil CI), AP, accuracy, macro-F1, sensitivity, specificity, PPV, NPV; prevalence-adjusted PPV/NPV at 50%, 0.5% and 1.2%.
- **Baseline:** L2 logistic regression on age and vascular status, holdout on the imaging test set plus stratified 5-fold CV.
- **Statistics:** Mann–Whitney U, χ² / Fisher exact, Cliff's delta, odds ratio, Cramér's V, Benjamini–Hochberg FDR over seven comparisons.
- **Explainability:** Grad-CAM on `conv_head` for the predicted class; occlusion (patch 96, stride 48) on CRC probability; attribution within the retinal mask summarized over macular and peripapillary ROIs.

## Data availability

UK Biobank data are not redistributed; access is through the UK Biobank approved researcher process.

## Citation

```bibtex
@article{Jovovic2026CRCRetina,
  author  = {Jovovic, Ivan and Popovic, Tomo and Rubezic Lukic, Isidora and Babic, Dejan and Popovic, Natasa},
  title   = {Deep Learning Discrimination of Colorectal Cancer using Retinal Fundus Photographs: {UK Biobank} Case-Control Study},
  journal = {Current Medical Imaging},
  year    = {2026},
  volume  = {22},
  pages   = {e15734056494970},
  doi     = {10.2174/0115734056494970260902110339}
}
```
