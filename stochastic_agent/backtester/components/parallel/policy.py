from ray.rllib.utils.policy_client import PolicyClient

policy_client = PolicyClient("http://localhost:9900")

def start_episode():
    return policy_client.start_episode()

def get_action(episode, obs):
    return policy_client.get_action(episode_id=episode, observation=obs)

def log_reward(episode, reward):
    policy_client.log_returns(episode, reward)