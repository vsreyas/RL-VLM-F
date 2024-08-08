export CUDA_VISIBLE_DEVICES=1
python agent/iql.py --data_set_path "/home/sreyas/RL-VLM-F/RL-VLM-F/exp/datagen_drawer_open-Random/metaworld_drawer-open-v2/2024-08-01-21-33-30/vlm_1gemini_free_form_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/datagen_PassWater_init1000_unsup9000_inter4000_maxfeed20000_seg1_acttanh_Rlr0.0003_Rbatch40_Rupdate10_en3_sample0_large_batch10_seed0/data.pkl"\
 --seed 42 --vlm_reward False --const_reward 0.0 --name "drawer_open-random-"

python agent/iql.py --data_set_path "/home/sreyas/RL-VLM-F/RL-VLM-F/exp/datagen_drawer_open-Random/metaworld_drawer-open-v2/2024-08-01-21-33-30/vlm_1gemini_free_form_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/datagen_PassWater_init1000_unsup9000_inter4000_maxfeed20000_seg1_acttanh_Rlr0.0003_Rbatch40_Rupdate10_en3_sample0_large_batch10_seed0/data.pkl"\
 --seed 42 --vlm_reward True --name "drawer_open-random-"

python agent/iql.py --data_set_path "/home/sreyas/RL-VLM-F/RL-VLM-F/exp/datagen_drawer_open-Random/metaworld_drawer-open-v2/2024-08-01-21-33-30/vlm_1gemini_free_form_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/datagen_PassWater_init1000_unsup9000_inter4000_maxfeed20000_seg1_acttanh_Rlr0.0003_Rbatch40_Rupdate10_en3_sample0_large_batch10_seed0/data.pkl"\
 --seed 42 --name "drawer_open-random-"