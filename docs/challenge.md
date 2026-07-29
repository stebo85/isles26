Participants are challenged to develop automated segmentation algorithms that accurately delineate ischemic stroke lesions in T1-weighted MRI, covering acute, sub-acute, and chronic lesions.

Inputs
Native space T1w MRI
Metadata (.json): Includes stroke lesion covariates such as DAYS_POST_STROKE, CHRONICITY, and CENTER code.

> CORRECTION (2026-07-28): an earlier version of this line said CHRONICITY is
> "1 if < 180 days". That is backwards. Measured over the 26 public training
> cases where both fields are present and valid, CHRONICITY=1 occurs only when
> DAYS_POST_STROKE >= 180 — i.e. **1 means CHRONIC**. Follow
> [isles26_data_status.md](isles26_data_status.md). Getting this backwards
> would invert the chronicity label on 270 training cases.
Outputs
Binary infarct segmentation mask.


Goal: build a small and highly capable model for segmenting this data that could run in the browser
- the model should be able to segment other contrasts as well, so we need to use an extensive dataset augmentation based on synthseg
- start with nnUnet model
- look at literature from previous challenge - what worked well? What didn't work?
- build the evaluation pipeline first
- download winner models from last challenge and run - then evaluate if their performance matches what was published last time
- try out to use Shakes SAM base model as a starting point