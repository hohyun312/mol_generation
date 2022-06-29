import numpy as np


class BufferingWrapper:
    def __init__(self, venv, buffer_size):
        self.venv = venv
        self.buffer_size = buffer_size
        self.n_envs = venv.n_envs
        self.n_transitions = None
        self._last_obs = None
        self._buffer_reset()
        
    def reset(self):
        self._last_obs = self.venv.reset()
        return self._last_obs
    
    def step(self, actions):
        obs, rews, dones, infos = self.venv.step(actions)
        next_obs = obs.copy()
        next_obs[dones] = [info['terminal_observation'] for info in infos[dones]]
        self.add(self._last_obs, actions, next_obs, dones)
        self._last_obs = obs
        return obs, rews, dones, infos
        
    def add(self, obs, acts, next_obs, dones):
        self.obs[self.n_transitions] = obs
        self.acts[self.n_transitions] = acts
        self.next_obs[self.n_transitions] = next_obs
        self.dones[self.n_transitions] = dones
        self.n_transitions += 1
            
    def _buffer_reset(self):
        self.obs = np.zeros((self.buffer_size, self.n_envs), dtype=object)
        self.acts = np.zeros((self.buffer_size, self.n_envs, 4), dtype=np.int64)
        self.next_obs = np.zeros((self.buffer_size, self.n_envs), dtype=object)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.n_transitions = 0

    def pop_trainsitions(self):        
        obs = self.obs.ravel()
        acts = self.acts.reshape(self.buffer_size*self.n_envs, 4)
        next_obs = self.next_obs.ravel()
        dones = self.dones.ravel()
        self._buffer_reset()
        return obs, acts, next_obs, dones
    
    def __len__(self):
        return self.n_transitions
    
    def __repr__(self):
        return "BufferingWrapper(size=%s/%s)" %(self.n_transitions, self.buffer_size)
    
    
    
def get_transitions_from_trajectory(trajectory):
    n = len(trajectory)-1
    obs = trajectory[:-1]
    next_obs = trajectory[1:]

    node = np.zeros((n, 2), dtype=np.int64)
    for i, (o, no) in enumerate(zip(obs, next_obs)):
        n1, n2 = no.edge_index[:, -2]
        if o.num_node == n2:
            n2 = o.num_node + no.node_type[-1]
        node[i] = n1, n2

    edge = np.array([o.edge_type[-1] for o in next_obs], dtype=np.int64)
    stop = np.zeros(n, dtype=np.int64)
    stop[-1] = 1

    obs = np.array(obs, dtype=object)
    actions = np.column_stack([node, edge, stop])
    next_obs = np.array(next_obs, dtype=object)
    return obs, actions, next_obs