from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

# -------------------------
# Agent configuration
# -------------------------
@dataclass
class AgentParams:
    obs_dim: Optional[int] = None           # to be specified after env creation
    action_dim: Optional[int] = None          # to be specified after env creation
    action_range: Optional[List[float]] = None  # to be specified after env creation
    device: str = "cuda:0"
    critic_cfg: Dict[str, Any] = field(default_factory=dict)
    actor_cfg: Dict[str, Any] = field(default_factory=dict)
    discount: float = 0.99
    init_temperature: float = 0.1
    alpha_lr: float = 1e-4
    alpha_betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    actor_lr: float = 1e-4
    actor_betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    actor_update_frequency: int = 1
    critic_lr: float = 1e-4
    critic_betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    critic_tau: float = 0.005
    critic_target_update_frequency: int = 2
    batch_size: int = 1024
    learnable_temperature: bool = True

@dataclass
class AgentConfig:
    name: str = "sac"
    # Use 'class_' to avoid conflicting with Python's reserved keyword "class"
    class_: str = "agent.sac.SACAgent"
    params: AgentParams = AgentParams()

# -------------------------
# Critic configuration
# -------------------------
@dataclass
class DoubleQCriticParams:
    obs_dim: Optional[int] = None           # will be set from agent.params.obs_dim
    action_dim: Optional[int] = None          # will be set from agent.params.action_dim
    hidden_dim: int = 1024
    hidden_depth: int = 2

@dataclass
class DoubleQCriticConfig:
    class_: str = "agent.critic.DoubleQCritic"
    params: DoubleQCriticParams = DoubleQCriticParams()

# -------------------------
# Actor configuration
# -------------------------
@dataclass
class DiagGaussianActorParams:
    obs_dim: Optional[int] = None           # will be set from agent.params.obs_dim
    action_dim: Optional[int] = None          # will be set from agent.params.action_dim
    hidden_depth: int = 2
    hidden_dim: int = 1024
    log_std_bounds: List[float] = field(default_factory=lambda: [-5, 2])

@dataclass
class DiagGaussianActorConfig:
    class_: str = "agent.actor.DiagGaussianActor"
    params: DiagGaussianActorParams = DiagGaussianActorParams()

# -------------------------
# Hydra runtime configuration
# -------------------------
@dataclass
class HydraConfig:
    name: str = "${env}"
    run: Dict[str, Any] = field(default_factory=lambda: {
        "dir": "./exp/${exp_name}/${env}/${now:%Y-%m-%d}-${now:%H-%M-%S}/"
               "vlm_${vlm_label}${vlm}_reward${reward}_H${diag_gaussian_actor.params.hidden_dim}_"
               "L${diag_gaussian_actor.params.hidden_depth}_lr${agent.params.actor_lr}/"
               "teacher_b${teacher_beta}_g${teacher_gamma}_m${teacher_eps_mistake}_"
               "s${teacher_eps_skip}_e${teacher_eps_equal}/label_smooth_${label_margin}/"
               "schedule_${reward_schedule}/${experiment}_init${num_seed_steps}_unsup${num_unsup_steps}_"
               "inter${num_interact}_maxfeed${max_feedback}_seg${segment}_act${activation}_"
               "Rlr${reward_lr}_Rbatch${reward_batch}_Rupdate${reward_update}_"
               "en${ensemble_size}_sample${feed_type}_large_batch${large_batch}_seed${seed}"
    })

# -------------------------
# Overall configuration
# -------------------------
@dataclass
class Config:
    # Basic defaults and experiment settings
    defaults: List[Any] = field(default_factory=lambda: [{"agent": "sac"}])
    experiment: str = "datagen_PassWater"
    
    # Reward learning parameters
    segment: int = 1
    activation: str = "tanh"
    num_seed_steps: int = 1000
    num_unsup_steps: int = 5000
    num_interact: int = 5000
    reward_lr: float = 0.0003
    reward_batch: int = 100
    reward_update: int = 30  # for soccer, as per your config
    feed_type: int = 0
    reset_update: int = 100
    topK: int = 5
    ensemble_size: int = 3
    max_feedback: int = 20000
    large_batch: int = 10
    label_margin: float = 0.0
    teacher_beta: int = -1
    teacher_gamma: int = 1
    teacher_eps_mistake: int = 0
    teacher_eps_skip: int = 0
    teacher_eps_equal: int = 0

    # Scheduling and training steps
    reward_schedule: int = 0
    num_train_steps: float = 1e6
    # Often, replay_buffer_capacity is set equal to num_train_steps
    replay_buffer_capacity: float = 1e6

    # Evaluation configuration
    eval_frequency: int = 10000
    num_eval_episodes: int = 1000
    device: str = "cuda:0"

    # Logger and saving options
    log_frequency: int = 10000
    log_save_tb: bool = False
    save_interval: int = 5000
    save_video: bool = False

    # Experiment setups
    seed: int = 0

    # Environment and related settings
    env: str = "softgym_PassWater"
    gradient_update: int = 1

    # VLM label and parameters
    vlm_label: int = 1
    vlm: str = "gemini_free_form"
    flip_vlm_label: int = 0
    sum_segment_score: bool = False
    collect_data_interval: int = 0
    max_image_difference: int = 0
    use_first_and_last: int = 0
    image_reward: int = 1
    resnet: int = 0
    conv_kernel_sizes: List[int] = field(default_factory=lambda: [5, 3, 3, 3])
    conv_n_channels: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    conv_strides: List[int] = field(default_factory=lambda: [3, 2, 2, 2])
    image_size: int = 300
    cached_label_path: Optional[str] = None

    # Experiment naming and prompt settings
    exp_name: str = "datagen"
    prompt: str = "???"
    clip_prompt: str = "The green drawer is completely opened."
    reward: str = "learn_from_preference"

    # Paths for pretrained models
    reward_model_load_dir: str = ""
    reward_model_score_load_dir: Optional[str] = None
    agent_model_load_dir: Optional[str] = None
    reward_model_load_step: int = 525000
    agent_load_step: int = 525000
    mode: str = "eval"
    save_images: bool = False
    epsilon: float = 0.01
    dataset_dir: str = ""

    # Hydra-specific settings
    hydra: HydraConfig = HydraConfig()

    # Nested configurations for agent, critic, and actor
    agent: AgentConfig = AgentConfig()
    double_q_critic: DoubleQCriticConfig = DoubleQCriticConfig()
    diag_gaussian_actor: DiagGaussianActorConfig = DiagGaussianActorConfig()
