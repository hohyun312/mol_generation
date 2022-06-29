import hydra
from omegaconf import DictConfig

import logging

from src.layers import GGNN
from src.ppo_agent import GCPNPolicy, PPOAlgorithm
from src.rl_env import GCPNMoleculeEnv, AtomReward
import src.utils as utils


@hydra.main(version_base=None, config_path="configs", config_name="atom_rl")
def main(cfg: DictConfig):

    ggnn = GGNN(
        cfg.graph_input_dim, 
        cfg.graph_feature_dim, 
        cfg.graph_hidden_dims
    )
    node_feature_extractor = utils.AtomFeature(cfg.atom_features).transform
    policy = GCPNPolicy(
        ggnn, 
        node_feature_extractor,
        cfg.actor_hidden_dims, 
        cfg.critic_hidden_dims,
        cfg.name,
        cfg.save_dir
    )
    reward_fn = AtomReward(cfg.rewarding_atom).get_reward
    venv = GCPNMoleculeEnv(
        n_envs=cfg.n_envs, 
        reward_fn=reward_fn
    )
    logger = logging.getLogger(cfg.name)
    ppo_algo = PPOAlgorithm(
        env=venv, 
        policy=policy,
        logger=logger,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        entropy_loss_coef=cfg.entropy_loss_coef,
        value_loss_coef=cfg.value_loss_coef,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        n_steps=cfg.n_steps,
        learning_rate=cfg.learning_rate,
        max_grad_norm=cfg.max_grad_norm,
        normalize_advantage=cfg.normalize_advantage,
        clip_range_vf=cfg.clip_range_vf
    )

    ppo_algo.learn(cfg.total_timesteps)

    policy.save_model()

if __name__ == "__main__":
    main()