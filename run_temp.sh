python relabel.py \
    env=CartPole-v1 \
    seed=0 \
    exp_name=reproduce \
    reward=learn_from_preference \
    vlm=gemini_free_form \
    vlm_label=1 \
    exp_name=2024-3-24-icml-rebuttal-more-seeds \
    segment=1 \
    image_reward=1 \
    max_feedback=10000 reward_batch=50 reward_update=50 \
    num_interact=5000 \
    num_train_steps=500000 \
    agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=1000 num_train_steps=500000   \
    feed_type=0 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0  \
    agent.params.actor_lr=0.0005 \
    cached_label_path=data/cached_labels/CartPole/seed_0/ \
    reward_model_load_dir=/home/venky/Desktop/RL-VLM-F/test_dummy/cartpole

