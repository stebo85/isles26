Participants are challenged to develop automated segmentation algorithms that accurately delineate ischemic stroke lesions in T1-weighted MRI, covering acute, sub-acute, and chronic lesions.

Inputs
Native space T1w MRI
Metadata (.json): Includes stroke lesion covariates such as DAYS_POST_STROKE, CHRONICITY (1 if < 180 days), and CENTER code.
Outputs
Binary infarct segmentation mask.


Goal: build a small and highly capable model for segmenting this data that could run in the browser
- the model should be able to segment other contrasts as well, so we need to use an extensive dataset augmentation based on synthseg
- start with nnUnet model
- look at literature from previous challenge - what worked well? What didn't work?
- build the evaluation pipeline first
- download winner models from last challenge and run - then evaluate if their performance matches what was published last time
- try out to use Shakes SAM base model as a starting point