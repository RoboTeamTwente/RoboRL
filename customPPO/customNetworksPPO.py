# customNetworksPPO.py
# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO networks."""

from typing import Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax.numpy as jnp
import jax


@flax.struct.dataclass
class PPONetworks:
  policy_network: networks.FeedForwardNetwork
  value_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(ppo_networks: PPONetworks, num_agents: int, obs_size_per_agent: int):
  """Creates params and inference function for the PPO agent.

    Args:
    ppo_networks: The PPO networks.
    num_agents: Number of agents in the environment.
    obs_size_per_agent: Observation size for each agent. 
    """

  def make_policy(
      params: types.Params, deterministic: bool = False
  ) -> types.Policy:
    policy_network = ppo_networks.policy_network
    parametric_action_distribution = ppo_networks.parametric_action_distribution

    action_size_per_agent = 4
    expected_obs_dim = obs_size_per_agent * num_agents

    def policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      param_subset = (params[0], params[1])  # normalizer and policy params

      obs_dim = observations.shape[-1]
      if obs_dim != expected_obs_dim:
        raise ValueError(
            f"Expected observation dimension {expected_obs_dim} "
            f"(num_agents={num_agents} * obs_size_per_agent={obs_size_per_agent}), "
            f"but got {obs_dim}"
        )
      
      obs_list = []
      for agent_idx in range(num_agents):
        start_idx = agent_idx * obs_size_per_agent
        agent_obs = observations[:, start_idx:start_idx + obs_size_per_agent]
        obs_list.append(agent_obs)

      logits_list = []
      for agent_obs in obs_list:
        agent_logits = policy_network.apply(*param_subset, agent_obs)
        logits_list.append(agent_logits)

      if deterministic:
        actions_list = []
        for agent_logits in logits_list:
            agent_actions = parametric_action_distribution.mode(agent_logits)
            actions_list.append(agent_actions)

        actions = jnp.concatenate(actions_list, axis=-1)
        return actions, {}
    
      # Stochastic policy
      keys = jax.random.split(key_sample, num_agents)
      raw_actions_list = []
      log_probs_list = []
      actions_list = []

      for agent_idx, agent_logit in enumerate(logits_list):
        agent_key = keys[agent_idx]
        agent_raw_actions = parametric_action_distribution.sample_no_postprocessing(
            agent_logit, agent_key
        )
        raw_actions_list.append(agent_raw_actions)

        log_prob = parametric_action_distribution.log_prob(agent_logit, agent_raw_actions)
        log_probs_list.append(log_prob)

        postprocessed_actions = parametric_action_distribution.postprocess(agent_raw_actions)
        actions_list.append(postprocessed_actions)

      actions = jnp.concatenate(actions_list, axis=-1)
      raw_actions = jnp.concatenate(raw_actions_list, axis=-1)
      # print("Actions shape: {}", actions.shape)

      # Sum log probabilities
      log_prob = sum(log_probs_list)
      
      return actions, {
          'log_prob': log_prob,
          'raw_action': raw_actions,
      }

    return policy
  return make_policy


def make_ppo_networks( # Has also been modified!
    num_agents: int = 1,
    agent_observation_size: int = 16,
    agent_action_size: int = 4,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
) -> PPONetworks:
  """Make PPO networks with preprocessor."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=agent_action_size
  )
  policy_network = networks.make_policy_network(
      parametric_action_distribution.param_size,
      agent_observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      obs_key=policy_obs_key,
  )

  total_observation_size = num_agents * agent_observation_size
  value_network = networks.make_value_network(
      total_observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
      obs_key=value_obs_key,
  )

  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution,
  )