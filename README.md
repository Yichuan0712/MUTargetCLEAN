# MUTarget + CLEAN 240223

The primary objective of this branch is to integrate MUTarget with contrastive learning while ensuring minimal modifications to the original MUTarget codebase.

## Modifications

Here's a summary of the changes made:

1. Added new configurations related to SupCon in configs.yaml.
2. Updated data.py: LocalizationDataset can now fetch both positive and negative samples from the dataset.
3. Introduced LayerNormNet from the CLEAN methodology into model.py as the projection head for SupCon.
4. Implemented the projection head after the ESM layer. This is activated when configs.supcon.apply is set to True; otherwise, the standard MUTarget process is followed.
5. Introduced a new loss.py incorporating SupConHardLoss from the CLEAN method.
6. Fixed a minor bug in prepare_samples that prevented handling datasets including the deprecated label 'dual'.

## To-Do List

1. **Combining Losses**: Is it feasible to simply aggregate three different losses?
   
   - Recommendation: Avoid direct aggregation. Temporarily apply a small weight to the SupCon loss to prevent gradient explosion issues.

2. **Distance Map Efficiency**: How can we enhance the efficiency of the distance map calculation?
   
   - Current challenge: The hard-mine function has significantly slowed down, making inference times unacceptable due to the updating ESM2.

3. **Evaluation & Testing for SupCon**: MaxSep and Pvalue... later.
