# python custom/collect_dataset_softgym.py \
#     env=metaworld_drawer-open-v2 \
#     seed=0 \
#     exp_name=reproduce \
#     reward=learn_from_preference \
#     vlm_label=1 \
#     vlm=gemini_free_form \
#     image_reward=1 \
#     reward_batch=40 \
#     segment=1 \
#     teacher_eps_mistake=0 \
#     reward_update=10 \
#     num_interact=4000 \
#     max_feedback=20000 \
#     agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
#     num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
#     diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
#     feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
#     num_eval_episodes=1000 \
#     reward_model_load_dir=/project_data/held/sreyas/RL-VLM-F/reward_models/Drawer_open\
#     reward_model_load_step=999999 \
#     dataset_dir=/project_data/held/sreyas/sim_data/datagen/drawer/datagen_drawer_open-Medium/old\
#     exp_name=datagen_drawer_open-Medium

# python custom/collect_dataset_softgym.py \
#     env=metaworld_drawer-open-v2 \
#     seed=0 \
#     exp_name=reproduce \
#     reward=learn_from_preference \
#     vlm_label=1 \
#     vlm=gemini_free_form \
#     image_reward=1 \
#     reward_batch=40 \
#     segment=1 \
#     teacher_eps_mistake=0 \
#     reward_update=10 \
#     num_interact=4000 \
#     max_feedback=20000 \
#     agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
#     num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
#     diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
#     feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
#     num_eval_episodes=1000 \
#     reward_model_load_dir=/project_data/held/sreyas/RL-VLM-F/reward_models/Drawer_open\
#     reward_model_load_step=999999 \
#     epsilon=2.0\
#     dataset_dir=/project_data/held/sreyas/sim_data/datagen/drawer/datagen_drawer_open-random/old\
#     exp_name=datagen_drawer_open-random


# python custom/collect_dataset_softgym.py \
#     env=metaworld_drawer-open-v2 \
#     seed=0 \
#     exp_name=reproduce \
#     reward=learn_from_preference \
#     vlm_label=1 \
#     vlm=gemini_free_form \
#     image_reward=1 \
#     reward_batch=40 \
#     segment=1 \
#     teacher_eps_mistake=0 \
#     reward_update=10 \
#     num_interact=4000 \
#     max_feedback=20000 \
#     agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
#     num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
#     diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
#     feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
#     num_eval_episodes=1000 \
#     reward_model_load_dir=/project_data/held/sreyas/RL-VLM-F/reward_models/Drawer_open\
#     reward_model_load_step=999999 \
#     dataset_dir=/project_data/held/sreyas/sim_data/datagen/drawer/datagen_drawer_open-expert/old\
#     exp_name=datagen_drawer_open-expert

# python custom/collect_dataset_softgym.py \
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
#     reward_model_load_dir=/project_data/held/sreyas/RL-VLM-F/reward_models/Rope\
#     reward_model_load_step=599999 \
#     dataset_dir=/project_data/held/sreyas/sim_data/datagen/Rope_flatten_easy/datagen_RopeFlattenEasy-Expert/old\
#     exp_name=datagen_RopeFlattenEasy-Expert

# python custom/collect_dataset_softgym.py \
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
#     reward_model_load_dir=/project_data/held/sreyas/RL-VLM-F/reward_models/Rope\
#     reward_model_load_step=599999 \
#     epsilon=2.0\
#     dataset_dir=/project_data/held/sreyas/sim_data/datagen/Rope_flatten_easy/datagen_RopeFlattenEasy-Random/old\
#     exp_name=datagen_RopeFlattenEasy-Random

# python custom/collect_dataset_softgym.py \
#     env=CartPole-v1 \
#     seed=0 \
#     exp_name=reproduce \
#     reward=learn_from_preference \
#     vlm=gemini_free_form \
#     vlm_label=1 \
#     exp_name=2024-3-24-icml-rebuttal-more-seeds \
#     segment=1 \
#     image_reward=1 \
#     max_feedback=10000 reward_batch=50 reward_update=50 \
#     num_interact=5000 \
#     num_train_steps=500000 \
#     agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=1000 num_train_steps=500000   \
#     feed_type=0 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0  \
#     agent.params.actor_lr=0.0005 \
#     reward_model_load_dir=/project_data/held/sreyas/RL-VLM-F/reward_models/CartPole\
#     reward_model_load_step=499999 \
#     dataset_dir=/project_data/held/sreyas/sim_data/datagen/Cartpole/datagen_Cartpole-Expert/old\
#     exp_name=datagen_Cartpole-Expert


# python custom/collect_dataset_softgym.py \
#     env=CartPole-v1 \
#     seed=0 \
#     exp_name=reproduce \
#     reward=learn_from_preference \
#     vlm=gemini_free_form \
#     vlm_label=1 \
#     exp_name=2024-3-24-icml-rebuttal-more-seeds \
#     segment=1 \
#     image_reward=1 \
#     max_feedback=10000 reward_batch=50 reward_update=50 \
#     num_interact=5000 \
#     num_train_steps=500000 \
#     agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=1000 num_train_steps=500000   \
#     feed_type=0 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0  \
#     agent.params.actor_lr=0.0005 \
#     reward_model_load_dir=/project_data/held/sreyas/RL-VLM-F/reward_models/CartPole\
#     reward_model_load_step=499999 \
#     epsilon=2.0\
#     dataset_dir=/project_data/held/sreyas/sim_data/datagen/Cartpole/datagen_Cartpole-Random/old\
#     exp_name=datagen_Cartpole-Random


# python custom/collect_dataset_softgym.py \
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
#     reward_model_load_dir=/project_data/held/sreyas/RL-VLM-F/reward_models/Rope\
#     reward_model_load_step=599999 \
#     dataset_dir=/project_data/held/sreyas/sim_data/datagen/Rope_flatten_easy/datagen_RopeFlattenEasy-Medium-10000/old\
#     exp_name=datagen_RopeFlattenEasy-Medium-10000

# python custom/collect_dataset_softgym.py \
#     env=CartPole-v1 \
#     seed=0 \
#     exp_name=reproduce \
#     reward=learn_from_preference \
#     vlm=gemini_free_form \
#     vlm_label=1 \
#     exp_name=2024-3-24-icml-rebuttal-more-seeds \
#     segment=1 \
#     image_reward=1 \
#     max_feedback=10000 reward_batch=50 reward_update=50 \
#     num_interact=5000 \
#     num_train_steps=500000 \
#     agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=1000 num_train_steps=500000   \
#     feed_type=0 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0  \
#     agent.params.actor_lr=0.0005 \
#     reward_model_load_dir=/project_data/held/sreyas/RL-VLM-F/reward_models/CartPole\
#     reward_model_load_step=499999 \
#     dataset_dir=/project_data/held/sreyas/sim_data/datagen/Cartpole/datagen_Cartpole-Medium-5000/old\
#     exp_name=datagen_Cartpole-Medium-5000

# ### soccer
# python custom/collect_dataset_softgym.py \
#     env=metaworld_soccer-v2 \
#     seed=0 \
#     reward=learn_from_preference \
#     vlm_label=1 \
#     vlm=gemini_free_form \
#     image_reward=1 \
#     reward_batch=40 \
#     segment=1 \
#     teacher_eps_mistake=0 \
#     reward_update=5 \
#     num_interact=4000 \
#     max_feedback=20000 \
#     reward_lr=1e-4 \
#     agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
#     num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
#     diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
#     feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
#     reward_model_load_dir=/project_data/held/sreyas/RL-VLM-F/reward_models/Soccer\
#     reward_model_load_step=999999 \
#     dataset_dir=/project_data/held/sreyas/sim_data/datagen/soccer/soccer-expert/old\
#     exp_name=datagen_soccer-expert

python custom/collect_dataset_softgym.py \
    env=metaworld_soccer-v2 \
    seed=0 \
    reward=learn_from_preference \
    vlm_label=1 \
    vlm=gemini_free_form \
    image_reward=1 \
    reward_batch=40 \
    segment=1 \
    teacher_eps_mistake=0 \
    reward_update=5 \
    num_interact=4000 \
    max_feedback=20000 \
    reward_lr=1e-4 \
    agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
    num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
    diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
    feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
    reward_model_load_dir=/project_data/held/sreyas/RL-VLM-F/reward_models/Soccer\
    reward_model_load_step=999999 \
    dataset_dir=/project_data/held/sreyas/sim_data/datagen/soccer/soccer-medium-12000/old\
    exp_name=datagen_soccer-medium-12000

python custom/collect_dataset_softgym.py \
    env=metaworld_soccer-v2 \
    seed=0 \
    reward=learn_from_preference \
    vlm_label=1 \
    vlm=gemini_free_form \
    image_reward=1 \
    reward_batch=40 \
    segment=1 \
    teacher_eps_mistake=0 \
    reward_update=5 \
    num_interact=4000 \
    max_feedback=20000 \
    reward_lr=1e-4 \
    agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
    num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
    diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
    feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
    reward_model_load_dir=/project_data/held/sreyas/RL-VLM-F/reward_models/Soccer\
    reward_model_load_step=999999 \
    epsilon=2.0\
    dataset_dir=/project_data/held/sreyas/sim_data/datagen/soccer/soccer-random/old\
    exp_name=datagen_soccer-random