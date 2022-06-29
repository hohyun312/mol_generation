import os
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter.composite import scatter_log_softmax
from torch_scatter import scatter_max

from src.layers import MultiLayerPerceptron
from src.utils import explained_variance
from src.mgraph import PackedMolecularGraph

    
class GCPNPolicy(nn.Module):
    def __init__(self,
                 graph_emb,
                 feature_extractor,
                 actor_hidden_dims,
                 critic_hidden_dims,
                 name='policy',
                 checkpoint_dir='./',
                 num_node_type=9,
                 num_edge_type=3,
                ):
        super().__init__()
        self.checkpoint_path = os.path.join(checkpoint_dir, name)
        self.graph_emb = graph_emb
        self.feature_extractor = feature_extractor
        
        self.num_node_type = num_node_type
        self.num_edge_type = num_edge_type
        
        self.atom_embeddings = nn.Parameter(
            torch.zeros(num_node_type, self.graph_emb.node_dim))
        nn.init.normal_(self.atom_embeddings, mean=0, std=0.1)
        
        node1_input_dim = self.graph_emb.node_dim + self.graph_emb.graph_dim
        node2_input_dim = node1_input_dim + self.graph_emb.node_dim
        edge_input_dim = 2 * self.graph_emb.node_dim

        self.mlp_node1 = MultiLayerPerceptron([node1_input_dim, *actor_hidden_dims, 1],
                                             activation="relu")
        self.mlp_node2 = MultiLayerPerceptron([node2_input_dim, *actor_hidden_dims, 1],
                                             activation="relu")
        self.mlp_edge = MultiLayerPerceptron([edge_input_dim, *actor_hidden_dims, num_edge_type],
                                             activation="relu")
        self.mlp_stop = MultiLayerPerceptron([self.graph_emb.graph_dim, *actor_hidden_dims, 2],
                                             activation="relu")
        self.mlp_critic = MultiLayerPerceptron([self.graph_emb.graph_dim, *critic_hidden_dims, 1],
                                             activation="relu")
        
        
    def predict_values(self, obs):
        graph = PackedMolecularGraph(obs)
        input_feature = self.feature_extractor(graph)
        graph_feature, _ = self.graph_emb(graph, input_feature)
        return self.mlp_critic(graph_feature).flatten()
    
        
    def evaluate_actions(self, obs, actions):
        action1, action2, action3, action4 = actions.t()
        graph = self._prepare_actions(obs)
        
        # node1
        node1_log_probs = self._node1_log_probs(graph)
        max_num_node = graph.num_nodes.max()
        packed_index = graph.node2graph * max_num_node + graph.node_ids_per_graph
        node1_dist = self._get_action_distribution(node1_log_probs, 
                                       packed_index,
                                       size=(graph.batch_size, max_num_node))
        node1_index = action1 + graph.offsets
        node1_log_probs = node1_dist.log_prob(action1)
        
        # node2
        node2_log_probs = self._node2_log_probs(graph, node1_index)
        max_num_node = graph.num_nodes.max() + self.num_node_type
        packed_index = graph.extended_node2graph * max_num_node + graph.extended_node_ids_per_graph
        node2_dist = self._get_action_distribution(node2_log_probs, 
                                       packed_index,
                                       size=(graph.batch_size, max_num_node))
        node2_index = self._get_packed_node_indices(action2, 
                                        packed_index, 
                                        size=(graph.batch_size, max_num_node))                        
        node2_log_probs = node2_dist.log_prob(action2)
        
        # edge
        edge_log_probs = self._edge_log_probs(graph, node1_index, node2_index)
        edge_dist = torch.distributions.Categorical(probs=edge_log_probs.exp())
        edge_log_probs = edge_dist.log_prob(action3)
        
        # stop
        stop_log_probs = self._stop_log_probs(graph)
        stop_dist = torch.distributions.Categorical(probs=stop_log_probs.exp())
        stop_log_probs = stop_dist.log_prob(action4)
        
        # values
        values = self.mlp_critic(graph.graph_feature).flatten()
        log_probs = torch.vstack(
            (node1_log_probs, node2_log_probs, edge_log_probs, stop_log_probs)).sum(dim=0)
        
        #entropy
        entropy = node1_dist.entropy() + node2_dist.entropy() + edge_dist.entropy() + stop_dist.entropy()
        
        return values, log_probs, entropy
        
    
    def predict(self, obs, deterministic=False):
        graph = self._prepare_actions(obs)
        
        node1_log_probs = self._node1_log_probs(graph)
        if deterministic:
            node1_log_probs, node1_index = scatter_max(node1_log_probs, graph.node2graph)
            action1 = graph.node_ids_per_graph[node1_index]
        else:
            max_num_node = graph.num_nodes.max()
            packed_index = graph.node2graph * max_num_node + graph.node_ids_per_graph
            node1_dist = self._get_action_distribution(node1_log_probs, 
                                                       packed_index,
                                                       size=(graph.batch_size, max_num_node))
            action1 = node1_dist.sample()
            node1_index = action1 + graph.offsets
            node1_log_probs = node1_dist.log_prob(action1)
            
        node2_log_probs = self._node2_log_probs(graph, node1_index)
        if deterministic:
            node2_log_probs, node2_index = scatter_max(node2_log_probs, graph.extended_node2graph)
            action2 = graph.extended_node_ids_per_graph[node2_index]
        else:
            max_num_node = graph.num_nodes.max() + self.num_node_type
            packed_index = graph.extended_node2graph * max_num_node + graph.extended_node_ids_per_graph
            node2_dist = self._get_action_distribution(node2_log_probs, 
                                                       packed_index,
                                                       size=(graph.batch_size, max_num_node))
            action2 = node2_dist.sample()
            node2_index = self._get_packed_node_indices(action2, 
                                      packed_index, 
                                      size=(graph.batch_size, max_num_node))                        
            node2_log_probs = node2_dist.log_prob(action2)
            
        edge_log_probs = self._edge_log_probs(graph, node1_index, node2_index)
        stop_log_probs = self._stop_log_probs(graph)
        
        if deterministic:
            action3 = edge_log_probs.argmax(dim=1)
            action4 = stop_log_probs.argmax(dim=1)
            edge_log_probs = edge_log_probs.gather(dim=1, index=action3.unsqueeze(1)).flatten()
            stop_log_probs = stop_log_probs.gather(dim=1, index=action4.unsqueeze(1)).flatten()
        else:
            edge_dist = torch.distributions.Categorical(probs=edge_log_probs.exp())
            stop_dist = torch.distributions.Categorical(probs=stop_log_probs.exp())
            action3 = edge_dist.sample()
            action4 = stop_dist.sample()
            edge_log_probs = edge_dist.log_prob(action3)
            stop_log_probs = stop_dist.log_prob(action4)
        
        actions = torch.vstack((action1, action2, action3, action4)).t()
        values = self.mlp_critic(graph.graph_feature).flatten()
        log_probs = torch.vstack(
            (node1_log_probs, node2_log_probs, edge_log_probs, stop_log_probs)).sum(dim=0)
        return actions, values, log_probs
        
        
    def _prepare_actions(self, obs):
        ''' 
        Merge list of graphs into a single large graph, and variables that are needed for action predictions
            *extended_node_feature, extended_node2graph, node_ids_per_graph, extended_node_ids_per_graph
        '''
        graph = PackedMolecularGraph(obs)
        input_feature = self.feature_extractor(graph)
        graph.graph_feature, graph.node_feature = self.graph_emb(graph, input_feature)

        #   [node_embedding    ] 
        #  [new_atom_embedding ]
        graph.extended_node_feature = torch.vstack(
            (
                graph.node_feature,          
                self.atom_embeddings.repeat((graph.batch_size, 1)), 
            )
        )
        atom2graph = torch.arange(graph.batch_size).repeat_interleave(self.num_node_type)
        graph.extended_node2graph = torch.cat((graph.node2graph, atom2graph), dim=0)
        
        graph.node_ids_per_graph = torch.cat([torch.arange(n) for n in graph.num_nodes])
        atom_id = torch.arange(self.num_node_type).repeat(graph.batch_size, 1)
        atom_id = atom_id + graph.num_nodes[:, None]
        graph.extended_node_ids_per_graph = torch.cat((graph.node_ids_per_graph, atom_id.flatten()))
        return graph
    
        
    def _get_action_distribution(self, log_probs, packed_index, size):
        probs = torch.zeros(*size, dtype=torch.float32)
        probs.view(-1)[packed_index] = log_probs.exp()
        dist = torch.distributions.Categorical(probs=probs)
        return dist
    
    
    def _get_packed_node_indices(self, actions, packed_index, size):        
        node_id_per_batch = torch.zeros(*size, dtype=torch.int64)
        node_id_per_batch.view(-1)[packed_index] = torch.arange(len(packed_index))
        node_id_per_batch = node_id_per_batch.gather(1, actions.view(-1, 1)).view(-1)
        return node_id_per_batch
        
        
    def _node1_log_probs(self, graph):
        #   [node_embedding     graph_embedding]
        node1_input_feature = torch.cat(
            (
                graph.node_feature, 
                graph.graph_feature[graph.node2graph]
            ), dim=1)
        node1_logits = self.mlp_node1(node1_input_feature).view(-1)
        node1_log_probs = scatter_log_softmax(node1_logits, graph.node2graph)
        return node1_log_probs
    
    
    def _node2_log_probs(self, graph, selected_node1_index):
        #   [node_embedding     graph_embedding, seleted_node1_feature] 
        #  [new_atom_embedding  graph_embedding, seleted_node1_feature]
        node2_input_feature = torch.cat(
            (
                graph.extended_node_feature,
                graph.graph_feature[graph.extended_node2graph],
                graph.node_feature[selected_node1_index][graph.extended_node2graph]
            ), dim=1)
            
        node2_logits = self.mlp_node2(node2_input_feature).flatten()
        node2_logits[selected_node1_index] = -torch.inf  # exclude already selected node
        node1_nei = torch.isin(graph.edge_index[0], selected_node1_index) 
        node2_logits[graph.edge_index[1][node1_nei]] = -torch.inf # exclude existing bonds(neighbors of node1)
        node2_log_probs = scatter_log_softmax(node2_logits, graph.extended_node2graph)
        return node2_log_probs
    
    
    def _edge_log_probs(self, graph, selected_node1_index, selected_node2_index):
        edge_feature_node1 = graph.extended_node_feature[selected_node1_index]
        edge_feature_node2 = graph.extended_node_feature[selected_node2_index]
        edge_feature = torch.cat((edge_feature_node1, edge_feature_node2), dim=1)
        edge_logits = self.mlp_edge(edge_feature)
        return F.log_softmax(edge_logits, dim=1)
    
    
    def _stop_log_probs(self, graph):
        stop_logits = self.mlp_stop(graph.graph_feature)
        return F.log_softmax(stop_logits, dim=1)

    def save_model(self, suffix=''):
        torch.save(self.state_dict(), self.checkpoint_path + "%s.pt" %suffix)
        
    def load_model(self, suffix=''):
        state_dict = torch.load(self.checkpoint_path + "%s.pt" %suffix)
        return self.load_state_dict(state_dict)



class RolloutBuffer:
    def __init__(self, buffer_size, n_envs, gamma=0.99, gae_lambda=0.95):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        
        
    def add(self, observations, actions, rewards, dones, values, log_probs):
        self.observations[self.pos] = observations
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.values[self.pos] = values
        self.log_probs[self.pos] = log_probs
        self.pos += 1
            
    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.n_envs), dtype=object)
        self.actions = np.zeros((self.buffer_size, self.n_envs, 4), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pos = 0
        
    def compute_returns_and_advantage(self, last_values):
        
        next_values = last_values
        advantage_t = 0
        for t in reversed(range(self.buffer_size)):
            non_terminal = 1.0 - self.dones[t]
            delta_t = self.rewards[t] + non_terminal * self.gamma * next_values - self.values[t]
            advantage_t = delta_t + non_terminal * self.gamma * self.gae_lambda * advantage_t
            self.advantages[t] = advantage_t   
            next_values = self.values[t]
        self.returns = self.advantages + self.values
    

    def get(self, batch_size):
        n_data = self.buffer_size * self.n_envs
        random_indices = np.random.permutation(n_data)
        batch_range = range(0, 
                            n_data-n_data%batch_size, 
                            batch_size
                           )
        
        observations = self.observations.ravel()
        actions = self.actions.reshape(n_data, 4)
        log_probs = self.log_probs.ravel()
        values = self.values.ravel()
        advantages = self.advantages.ravel()
        returns = self.returns.ravel()

        for i in batch_range:
            indices = random_indices[i:i+batch_size]
            yield (
                observations[indices],
                torch.from_numpy(actions[indices]),
                torch.from_numpy(log_probs[indices]),
                torch.from_numpy(values[indices]),
                torch.from_numpy(advantages[indices]),
                torch.from_numpy(returns[indices])
            )
    
    def __len__(self):
        return len(self.observations)
    
    def __repr__(self):
        return "RolloutBuffer(size=%s/%s)" %(len(self), self.buffer_size)
    


class PPOAlgorithm():
    def __init__(self, env, policy, logger=None,
                 gamma=0.99, gae_lambda=0.95, clip_range=0.2, 
                 entropy_loss_coef=0.0, value_loss_coef=0.5, max_grad_norm=0.5,
                 normalize_advantage=True, clip_range_vf=None,
                 batch_size=64, n_epochs=1, n_steps=40, 
                 learning_rate=0.0003):
        self.env = env
        self.policy = policy
        self.logger = logger
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_loss_coef = entropy_loss_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.clip_range_vf = clip_range_vf
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        
        
        self._last_obs = None
        self.total_timesteps = None
        
        self.buffer = RolloutBuffer(n_steps, env.n_envs, gamma, gae_lambda)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        

    @torch.no_grad()
    def collect_rollouts(self, n_steps):
        self.buffer.reset()
        obs = self._last_obs

        for _ in range(n_steps):
            actions, values, log_probs = self.policy.predict(obs)

            actions = actions.cpu().numpy()
            values = values.cpu().numpy()
            log_probs = log_probs.cpu().numpy()

            obs, rewards, dones, _ = self.env.step(actions)
            self.buffer.add(self._last_obs, actions, rewards, dones, values, log_probs)
            self._last_obs = obs

        values = self.policy.predict_values(self._last_obs)
        self.buffer.compute_returns_and_advantage(values.cpu().numpy())
            
            
    def train(self):
        self.collect_rollouts(self.n_steps)
        
        running_policy_loss = 0
        running_value_loss = 0
        running_entropy_loss = 0
        running_total_loss = 0
        
        for epoch in range(self.n_epochs):
            
            for rollout_data in self.buffer.get(self.batch_size):
                obs, actions, old_log_probs, old_values, advantages, returns = rollout_data
                
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8)

                values, log_probs, entropy = self.policy.evaluate_actions(obs, actions)
                
                prob_ratio = torch.exp(log_probs - old_log_probs)
                
                policy_loss_1 = advantages * prob_ratio
                policy_loss_2 = advantages * torch.clip(prob_ratio, 1-self.clip_range, 1+self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf:
                    values_pred = old_values + \
                        (values - old_values).clip(-self.clip_range_vf, self.clip_range_vf)
                else:
                    values_pred = values
                
                value_loss = self.value_loss_coef * F.mse_loss(returns, values_pred)
                
                entropy_loss = - self.entropy_loss_coef * torch.mean(entropy)
                
                loss = policy_loss + value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 
                                               self.max_grad_norm)

                self.optimizer.step()
                
                # used for logging
                running_policy_loss += policy_loss.item()
                running_value_loss += value_loss.item()
                running_entropy_loss += entropy_loss.item()
                running_total_loss += loss.item()

        running_policy_loss /= self.n_epochs * self.batch_size
        running_value_loss /= self.n_epochs * self.batch_size
        running_entropy_loss /= self.n_epochs * self.batch_size
        running_total_loss /= self.n_epochs * self.batch_size
        explained_var = explained_variance(self.buffer.values, self.buffer.returns)

        return {
            'policy_loss': running_policy_loss,
            'value_loss': running_value_loss,
            'entropy_loss': running_entropy_loss,
            'total_loss': running_total_loss,
            'explained_var': explained_var
        }


    def learn(self, timesteps=1, consecutive=False, verbose=False):
        assert self.env.n_envs * self.n_steps % self.batch_size == 0
        if verbose:
            iterator = tqdm(range(timesteps))
        else:
            iterator = range(timesteps)

        if not consecutive or self._last_obs is None:
            self._last_obs = self.env.reset()
            self.total_timesteps = 0
        
        for timestep in iterator:
            losses = self.train()

            self.total_timesteps += 1

            if self.logger:
                self.logger.info("policy_loss: %s"   %losses['policy_loss'])
                self.logger.info("value_loss: %s"    %losses['value_loss'])
                self.logger.info("entropy_loss: %s"  %losses['entropy_loss'])
                self.logger.info("total_loss: %s"    %losses['total_loss'])
                self.logger.info("explained_var: %s" %losses['explained_var'])
            
            if self.total_timesteps % 500 == 0:
                self.policy("-%s" %self.total_timesteps)
    
