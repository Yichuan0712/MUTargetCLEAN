conda activate pytorch1.13.0
python train.py --config_path ./config.yaml
python train.py --config_path ./config_supcon.yaml

python train.py --config_path ./config_supcon.yaml --predict 1

python train.py --config_path ./config_supcon.yaml
#test   n_pos: 2 n_neg: 6 and pft 1


python train.py --config_path ./configs/config_supcon_onlysampling.yaml --result_path ./results_supcon_hardTrue_pft2
#this was wrong! use supcon's data sampling method but didn't calculate supcon loss becuase in train.py set 
#to calculate supcon loss only if configs.supcon.apply and warm_starting:  already changed this but still need to see this performance!
#in config add one additional parameter if configs.supcon.apply and configs.apply_supcon_loss: will apply the supcon loss
#this provide apply supcon sampling without supcon loss for previous old mutarget loss!
equal to run config_supcon_onlysampling.yaml

python train.py --config_path ./configs/config_supcon.yaml --result_path ./results_supcon_hardTrue_pft2_supconloss
#on 3.26.2024 changed data.py using clean dataloader where anchor are from different class


python train.py --config_path ./configs/config_supcon.yaml --result_path ./debug --predict 1 --resume_path /data/duolin/MUTargetCLEAN/results_supcon_hardTrue_pft2_supconloss/2024-03-27__00-02-01/checkpoints/best_model.pth
=======
python train.py --config_path ./config.yaml
python train.py --config_path ./config_supcon.yaml

python train.py --config_path ./config_supcon.yaml --predict 1

python train.py --config_path ./config_supcon.yaml
#test   n_pos: 2 n_neg: 6 and pft 1


python train.py --config_path ./configs/config_supcon_onlysampling.yaml --result_path ./results_supcon_hardTrue_pft2
#this was wrong! use supcon's data sampling method but didn't calculate supcon loss becuase in train.py set 
#to calculate supcon loss only if configs.supcon.apply and warm_starting:  already changed this but still need to see this performance!
#in config add one additional parameter if configs.supcon.apply and configs.apply_supcon_loss: will apply the supcon loss
#this provide apply supcon sampling without supcon loss for previous old mutarget loss!
equal to run config_supcon_onlysampling.yaml

python train.py --config_path ./configs/config_supcon.yaml --result_path ./results_supcon_hardTrue_pft2_supconloss
#on 3.26.2024 changed data.py using clean dataloader where anchor are from different class


python train.py --config_path ./configs/config_supcon.yaml --result_path ./debug --predict 1 --resume_path /data/duolin/MUTargetCLEAN/results_supcon_hardTrue_pft2_supconloss/2024-03-27__00-02-01/checkpoints/best_model.pth
python train.py --config_path ./configs/config_supcon.yaml --result_path ./debug 

python train.py --config_path ./configs/config_nosupcon2.yaml --result_path ./result_nosupcon/2/ 

python train.py --config_path ./configs/config_nosupcon_dataaug.yaml --result_path ./result_nosupcon/dataaug/time10_0.3_mask0.1
#cannot use MLM enable


python train.py --config_path ./configs/config_nosupcon_dataaug.yaml --result_path ./result_nosupcon/dataaug/time10_0.3_repeat_nooriginal
python train.py --config_path ./configs/config_nosupcon_dataaug.yaml --result_path ./result_nosupcon/dataaug/time5_0.3_repeat_nooriginal


python train.py --config_path ./configs/config_nosupcon_dataaug.yaml \
--result_path ./result_nosupcon/dataaug_MLM/time5_0.3_repeat_nooriginal


python train.py --config_path ./configs/config_nosupcon_dataaug.yaml \
--result_path ./result_nosupcon/dataaug_MLM/time10smallestclass_0.3_MLM0.2_repeat_nooriginal


python train.py --config_path ./configs/config_nosupcon_CNNdecoder.yaml \
--result_path ./result_nosupcon/cnndecoder/cnn_c8_k14

python train.py --config_path ./configs/config_nosupcon_CNNdecoder.yaml \
--result_path ./result_nosupcon/cnndecoder/cnn2_c32_k7_drop0.15

python train.py --config_path ./configs/config_nosupcon_CNNdecoder.yaml \
--result_path ./result_nosupcon/combine_cnnlineardecoder/cnn2linear_c32_k7_drop0.15