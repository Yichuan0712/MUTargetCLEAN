# MUTarget + CLEAN 240302

- `apply` - 是否使用SupCon，选择`False`以不使用。
- `device` - 指定使用的设备。
- `drop_out` - Dropout比率。
- `n_pos` - 正样本的数量。
- `n_neg` - 负样本的数量。
- `temperature` - 温度参数，用于调整损失函数中的缩放。
- `hard_neg` - 是否选择较难的negative sample计算loss。
- `weight` - 暂时使用这个参数来避免NaN。
- `warm_start` - Warm start结束的epoch。


~~The primary objective of this branch is to integrate MUTarget with contrastive learning while ensuring minimal modifications to the original MUTarget codebase.~~

~~## Modifications~~

~~Here's a summary of the changes made:~~

~~1. Added new configurations related to SupCon in configs.yaml.~~
~~2. Updated data.py: LocalizationDataset can now fetch both positive and negative samples from the dataset.~~
~~3. Introduced LayerNormNet from the CLEAN methodology into model.py as the projection head for SupCon.~~
~~4. Implemented the projection head after the ESM layer. This is activated when configs.supcon.apply is set to True; otherwise, the standard MUTarget process is followed.~~
~~5. Introduced a new loss.py incorporating SupConHardLoss from the CLEAN method.~~
~~6. Fixed a minor bug in prepare_samples that prevented handling datasets including the deprecated label 'dual'.~~

~~## To-Do List~~

~~1. **Combining Losses**: Is it feasible to simply aggregate three different losses?~~
   
   ~~- Recommendation: Avoid direct aggregation. Temporarily apply a small weight to the SupCon loss to prevent gradient explosion issues.~~

~~2. **Distance Map Efficiency**: How can we enhance the efficiency of the distance map calculation?~~
   
   ~~- Current challenge: The hard-mine function will significantly slow down, making inference times unacceptable due to the updating ESM2.~~

~~3. **Evaluation & Testing for SupCon**: MaxSep and Pvalue, I will do this later.~~


~~## To-Do 0226~~
~~two step 热启动~~
~~lr调节~~
~~distance map~~
