EXP_NAME=NLSNx4_PCL_FT
python main.py --model NLSN --scale 4 --dir_data ../data/DIV2K_EDSR \
--batch_size 16 --data_train DIV2K --rgb_range 1 --nlsn_n_hashes 4 --nlsn_chunk_size 144 \
--data_test Set5+Set14+B100+Urban100+Manga109 --patch_size 192 --data_range 1-800/801-900 --save_models \
--n_resblocks 32 --n_feats 256 --res_scale 0.1 --rgb_range 1 \
--save "$EXP_NAME" --save_results \
--exp_name "$EXP_NAME" --proj_name NLSNx4_PCL \
--loss 1*L1+3*CL-patchD --no_ad_loss \
--cl_layer 4 --use_wavelet --before_relu \
--use_aug --random_neg --only_blur --gpu_blur --sharp_hr --gpu_sharp --sharp_value 0.5 --sharp_range 2.5 \
--scheduler CosineRestart --cosine_tmax 200 --cosine_etamin 1e-9 --cosine_restart_weight 0.5 \
--lr 5e-6 --epochs 600 \
--pre_train ../model_zoo/NLSN/model_x4.pt --resume 0
