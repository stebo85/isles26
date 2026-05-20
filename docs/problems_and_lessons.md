# Problems, Failure Modes, And Lessons

This file turns the ATLAS and ISLES'22 literature into concrete risks for ISLES'26.

## Problem 1: Dice Can Hide Missed Lesions

Dice is dominated by large lesions. If a scan contains one large infarct plus several punctate lesions, a model can get a good Dice score while missing clinically meaningful small lesions.

Both ATLAS and ISLES'22 therefore used lesion-wise metrics. The ISLES'22 dataset paper states that the evaluation should capture both large-lesion borders and detection of small punctiform infarcts. The DeepISLES paper repeats the same point: lesion-wise F1 reflects localization and detection of distinct lesions, while Dice is voxel-overlap biased toward larger lesions.

Action:

- Always report lesion-wise F1, precision, and recall by lesion-size bin.
- Keep Dice, but do not select checkpoints by Dice alone.
- Add connected-component visual reviews for cases with low F1 but acceptable Dice.

## Problem 2: Small Lesions Are Underrepresented By Voxel Count

ATLAS v2.0 has many small connected components but few small-lesion voxels. Shang et al. report that 517 training lesions had volume under 100 voxels, yet their total voxel count was smaller than a single very large lesion can be.

Action:

- Oversample patches containing tiny lesions.
- Stratify folds by lesion volume and lesion count.
- Test MSL/DBL or a similar auxiliary labeling strategy.
- Consider lesion-wise sampling weights instead of purely voxel-wise loss.

## Problem 3: T1w Lesion Appearance Is Subtle And Phase-Dependent

The ISLES'22 dataset paper contrasts the multimodal acute task with ATLAS: chronic T1w lesions are reduced to glial scars, smaller infarcts are difficult to visualize, and etiology can be ambiguous. ISLES'26 expands this T1w problem across acute, sub-acute, and chronic stages, so appearance will vary more than in ATLAS v2.0.

Action:

- Use `DAYS_POST_STROKE` and `CHRONICITY` for stratified validation and subgroup reporting.
- Do not assume a single intensity signature for lesions.
- Build intensity augmentation that can mimic phase and scanner variation.

**Empirically confirmed (2026-05-20).** Our first nnU-Net v2 baseline trained
on ATLAS R2.1 (chronic T1w) scored 0.640 Dice / 0.654 lesion-F1 on the ATLAS
fold-0 held-out set (in-distribution, ~MAPPING-level) but only **0.188 Dice /
0.201 lesion-F1 on soop_bench** (out-of-distribution). The soop_bench
per-chronicity breakdown is monotonic — **chronic 0.53 > mixed 0.33 >
acute 0.12** — and the model massively under-segments acute cases (e.g. 0.48 mL
predicted vs 79.7 mL GT). Mechanism: ATLAS lesions are chronic encephalomalacia
on T1w; soop_bench GT is DWI/TRACE-defined acute infarcts that are bright on DWI
but subtle/invisible on T1w. **A chronic-T1w-trained model is the wrong tool for
an acute DWI-defined benchmark, and the gap is modality+phase, not model
quality.** Implication for ISLES'26: a T1w-only model must be validated on a
T1w-defined test set across all phases; do not read a low score on a
DWI-defined benchmark as a model-quality verdict. Full numbers + caveats in
`baselines/reports/nnunet/PIPELINE_STATUS.md`.

## Problem 4: Center And Scanner Shift Matter

ATLAS v2.0 was designed with a hidden generalizability set from separate cohorts. ISLES'22 multimodal data included a test center unseen during training, and DeepISLES analyzed performance by center. ISLES'26 has 60+ centers, so center shift is not a side issue.

Action:

- Create leave-one-center-out or grouped-center validation if enough center labels exist.
- Track per-center metrics and worst-center performance, not only mean performance.
- Avoid preprocessing choices that depend on hidden-test distributions.

## Problem 5: Public Leaderboards Encourage Overfitting

The ATLAS ranking weighted the hidden Docker test result 4x more heavily than public-test ranking. That is a direct warning: public test submissions can be gamed, even unintentionally.

Action:

- Keep an internal validation set untouched until final model selection.
- Limit public leaderboard-driven tuning.
- Treat a public leaderboard improvement without internal confirmation as suspect.

## Problem 6: Post-Processing Has Metric Tradeoffs

MAPPING's post-processing improved public Dice from 0.664 to 0.667 and lesion-wise F1 from 0.555 to 0.564, but post-processing rules are brittle if tuned to one dataset. Component removal helps false positives but can delete true tiny lesions.

Action:

- Tune thresholds on internal validation only.
- Report precision/recall tradeoffs before and after post-processing.
- Keep a no-postprocessing baseline for sanity.

## Problem 7: Human Ground Truth Is Imperfect

The ISLES'22 dataset paper reports that prior annotations often underestimated infarct borders and missed small embolic infarcts. The authors used a hybrid human-algorithm workflow plus expert review because manual delineation is expensive and variable.

Action:

- Expect label noise, especially at lesion boundaries and tiny lesions.
- Use robust losses and evaluate boundary tolerance where possible.
- Review apparent false positives manually; some may be plausible lesions absent from the label.

## Problem 8: Published "Strong T1w" Weights Are Effectively Unreachable

The strongest published T1w stroke segmenters all gate their weights behind
interactive auth that a compute cluster cannot script through (verified
2026-05-20):

- **MAPPING / CTRL** (ATLAS R2 winner): weights on a KCL SharePoint folder —
  returns HTTP 403 `Forms_Based_Auth_Required` from the cluster.
- **Halai et al. 2026** (`AjayHalai/Lesion-Segmentation-nnUNet`): weights on a
  Cambridge OneDrive share.
- **BrainSegFounder** (`lab-smile/BrainSegFounder`): weights on Google Drive.
- **qSynth** (`liamchalcroft/qsynth`): weights "coming soon", not yet released.

Action:

- Budget for **training our own** baseline rather than assuming a drop-in
  pretrained model; this is why the canonical move is "train nnU-Net on the
  data we have" (we have ATLAS R2.1 decrypted locally).
- If a published model is essential, ask a human to fetch the weights via
  browser and stage them on `/scratch`; do not burn cycles scripting around
  SharePoint/OneDrive auth.

## What We Should Avoid

- Selecting models by Dice only.
- Random k-fold validation that leaks center/chronicity structure.
- Building a browser-small model before having a strong teacher.
- Trusting a novel architecture without beating nnU-Net under identical folds.
- Removing tiny components blindly in post-processing.
- Ignoring metadata that the challenge explicitly provides.

## Sources

- ISLES'22 dataset paper: https://www.nature.com/articles/s41597-022-01875-5
- ATLAS v2.0 dataset paper: https://www.nature.com/articles/s41597-022-01401-7
- MAPPING winner report: https://arxiv.org/abs/2211.15486
- DeepISLES paper: https://www.nature.com/articles/s41467-025-62373-x
- MSL/DBL small-lesion paper: https://arxiv.org/abs/2408.02929
