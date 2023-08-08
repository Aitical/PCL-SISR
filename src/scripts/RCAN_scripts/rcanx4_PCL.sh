EXP_NAME=RCANx4_PCL_New
python main.py --model RCAN --scale 4 --n_resgroups 10 --n_resblocks 20 --n_feats 64 \
	--dir_data ../data/DIV2K_EDSR --batch_size 16 --data_train DIV2K --data_test Set5+Set14 \
	--patch_size 192 --data_range 1-800/801-900 --res_scale 1 --rgb_range 255 \
	--save "$EXP_NAME" --save_models --gclip 1 --save_results \
	--exp_name "$EXP_NAME" --proj_name RCANx4_PCL \
    --before_relu --cl_layer 4 --use_wavelet --contras_D_train \
	--loss "1*L1+5*CL-patchD" --cl_loss_type "InfoNCE" --no_ad_loss \
    --scheduler CosineRestart --cosine_tmax 200 --cosine_etamin 1e-8 --cosine_restart_weight 1.0 \
    --lr 5e-5 --epochs 600 \
    --pre_train "$1" --resume 0 \
	--use_aug \
    --random_neg --only_blur --gpu_blur 
    --gpu_sharp --sharp_hr --sharp_value 0.5 --sharp_range 2.5 \
 	