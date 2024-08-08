# # PassWater
# python dummy/collect_dataset_softgym.py \
#     env=softgym_PassWater \
#     seed=0 \
#     reward=learn_from_preference \
#     vlm_label=1 \
#     vlm=gemini_free_form \
#     image_reward=1 \
#     reward_batch=100 \
#     segment=1 \
#     teacher_eps_mistake=0 \
#     reward_update=30 \
#     num_interact=5000 \
#     max_feedback=20000 \
#     reward_lr=1e-4 \
#     agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
#     num_train_steps=600000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
#     diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
#     feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
#     resnet=1 \
#     cached_label_path=data/cached_labels/PassWater/seed_0/\
#     reward_model_load_dir=/home/sreyas/RL-VLM-F/RL-VLM-F/exp/reproduce/softgym_PassWater/2024-07-13-01-26-24/vlm_1gemini_free_form_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup9000_inter5000_maxfeed20000_seg1_acttanh_Rlr0.0001_Rbatch100_Rupdate30_en3_sample0_large_batch10_seed0/models\
#     reward_model_load_step=520000 \
#     agent_model_load_dir=/home/sreyas/RL-VLM-F/RL-VLM-F/exp/reproduce/softgym_PassWater/2024-07-13-01-26-24/vlm_1gemini_free_form_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup9000_inter5000_maxfeed20000_seg1_acttanh_Rlr0.0001_Rbatch100_Rupdate30_en3_sample0_large_batch10_seed0/models\
#     agent_load_step=520000 \
#     exp_name=datagen_PassWater


# ### ClothFold
# python dummy/collect_dataset_softgym.py \
#     env=softgym_ClothFoldDiagonal \
#     seed=0 \
#     vlm_label=1 \
#     vlm=gpt4v_two_image \
#     reward=learn_from_preference \
#     image_reward=1 \
#     num_train_steps=15000 \
#     agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=250 \
#     num_interact=1000 max_feedback=500 \
#     reward_batch=50 reward_update=25 \
#     resnet=1 \
#     feed_type=0 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 \
#     teacher_eps_skip=0 teacher_eps_equal=0 segment=1 num_seed_steps=250 \
#     eval_frequency=250 \
#     cached_label_path=data/cached_labels/ClothFold/seed_0/ \
#     reward_model_load_dir=/home/sreyas/RL-VLM-F/RL-VLM-F/exp/test/softgym_ClothFoldDiagonal/2024-07-13-01-29-55/vlm_1gpt4v_two_image_rewardlearn_from_preference_H1024_L2_lr0.0005/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init250_unsup250_inter1000_maxfeed500_seg1_acttanh_Rlr0.0003_Rbatch50_Rupdate25_en3_sample0_large_batch10_seed0/models\
#     load_step=15000 \
#     agent_model_load_dir=/home/sreyas/RL-VLM-F/RL-VLM-F/exp/test/softgym_ClothFoldDiagonal/2024-07-13-01-29-55/vlm_1gpt4v_two_image_rewardlearn_from_preference_H1024_L2_lr0.0005/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init250_unsup250_inter1000_maxfeed500_seg1_acttanh_Rlr0.0003_Rbatch50_Rupdate25_en3_sample0_large_batch10_seed0/models\
#     exp_name=datagen_ClothFoldDiagonal

# # # RopeFlattenEasy
# python dummy/collect_dataset_softgym.py \
#     env=softgym_RopeFlattenEasy \
#     seed=0 \
#     reward=learn_from_preference \
#     vlm_label=1 \
#     vlm=gemini_free_form \
#     image_reward=1 \
#     reward_batch=100 \
#     segment=1 \
#     teacher_eps_mistake=0 \
#     reward_update=30 \
#     num_interact=5000 \
#     max_feedback=20000 \
#     reward_lr=1e-4 \
#     agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
#     num_train_steps=600000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
#     diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
#     feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
#     resnet=1 \
#     cached_label_path=data/cached_labels/RopeFlattenEasy/seed_0/ \
#     reward_model_load_dir=/data/sreyas/RL_VLM_F-exp/reproduce/softgym_RopeFlattenEasy/2024-07-12-02-12-40/vlm_1gemini_free_form_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup9000_inter5000_maxfeed20000_seg1_acttanh_Rlr0.0001_Rbatch100_Rupdate30_en3_sample0_large_batch10_seed0/models\
#     reward_model_load_step=600000 \
#     epsilon=1\
#     exp_name=datagen_RopeFlattenEasy-Random
#     # agent_model_load_dir=/data/sreyas/RL_VLM_F-exp/reproduce/softgym_RopeFlattenEasy/2024-07-12-02-12-40/vlm_1gemini_free_form_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup9000_inter5000_maxfeed20000_seg1_acttanh_Rlr0.0001_Rbatch100_Rupdate30_en3_sample0_large_batch10_seed0/models\
#     # agent_load_step=0 \
    


# python dummy/collect_dataset_softgym.py \
#     env=softgym_RopeFlattenEasy \
#     seed=0 \
#     reward=learn_from_preference \
#     vlm_label=1 \
#     vlm=gemini_free_form \
#     image_reward=1 \
#     reward_batch=100 \
#     segment=1 \
#     teacher_eps_mistake=0 \
#     reward_update=30 \
#     num_interact=5000 \
#     max_feedback=20000 \
#     reward_lr=1e-4 \
#     agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
#     num_train_steps=600000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
#     diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
#     feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
#     resnet=1 \
#     cached_label_path=data/cached_labels/RopeFlattenEasy/seed_0/ \
#     reward_model_load_dir=/data/sreyas/RL_VLM_F-exp/reproduce/softgym_RopeFlattenEasy/2024-07-12-02-12-40/vlm_1gemini_free_form_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup9000_inter5000_maxfeed20000_seg1_acttanh_Rlr0.0001_Rbatch100_Rupdate30_en3_sample0_large_batch10_seed0/models\
#     reward_model_load_step=600000 \
#     agent_model_load_dir=/data/sreyas/RL_VLM_F-exp/reproduce/softgym_RopeFlattenEasy/2024-07-12-02-12-40/vlm_1gemini_free_form_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup9000_inter5000_maxfeed20000_seg1_acttanh_Rlr0.0001_Rbatch100_Rupdate30_en3_sample0_large_batch10_seed0/models\
#     agent_load_step=60000 \
#     exp_name=datagen_RopeFlattenEasy-Medium
export CUDA_VISIBLE_DEVICES=1
python dummy/collect_dataset_softgym.py \
    env=metaworld_drawer-open-v2 \
    seed=0 \
    exp_name=reproduce \
    reward=learn_from_preference \
    vlm_label=1 \
    vlm=gemini_free_form \
    image_reward=1 \
    reward_batch=40 \
    segment=1 \
    teacher_eps_mistake=0 \
    reward_update=10 \
    num_interact=4000 \
    max_feedback=20000 \
    agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
    num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
    diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
    feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
    num_eval_episodes=5000 \
    reward_model_load_dir=/data/sreyas/RL_VLM_F-exp/reproduce/metaworld_drawer-open-v2/complete_seed0_cached/vlm_1gemini_free_form_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup9000_inter4000_maxfeed20000_seg1_acttanh_Rlr0.0003_Rbatch40_Rupdate10_en3_sample0_large_batch10_seed0/models\
    reward_model_load_step=1000000 \
    agent_model_load_dir=/data/sreyas/RL_VLM_F-exp/reproduce/metaworld_drawer-open-v2/complete_seed0_cached/vlm_1gemini_free_form_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup9000_inter4000_maxfeed20000_seg1_acttanh_Rlr0.0003_Rbatch40_Rupdate10_en3_sample0_large_batch10_seed0/models\
    agent_load_step=200000 \
    exp_name=datagen_drawer_open-Medium

python dummy/collect_dataset_softgym.py \
    env=metaworld_drawer-open-v2 \
    seed=0 \
    exp_name=reproduce \
    reward=learn_from_preference \
    vlm_label=1 \
    vlm=gemini_free_form \
    image_reward=1 \
    reward_batch=40 \
    segment=1 \
    teacher_eps_mistake=0 \
    reward_update=10 \
    num_interact=4000 \
    max_feedback=20000 \
    agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
    num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
    diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
    feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
    num_eval_episodes=5000 \
    reward_model_load_dir=/data/sreyas/RL_VLM_F-exp/reproduce/metaworld_drawer-open-v2/complete_seed0_cached/vlm_1gemini_free_form_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup9000_inter4000_maxfeed20000_seg1_acttanh_Rlr0.0003_Rbatch40_Rupdate10_en3_sample0_large_batch10_seed0/models\
    reward_model_load_step=1000000 \
    exp_name=datagen_drawer_open-Random
    # agent_model_load_dir=/data/sreyas/RL_VLM_F-exp/reproduce/metaworld_drawer-open-v2/complete_seed0_cached/vlm_1gemini_free_form_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup9000_inter4000_maxfeed20000_seg1_acttanh_Rlr0.0003_Rbatch40_Rupdate10_en3_sample0_large_batch10_seed0/models\
    # agent_load_step=200000 \





