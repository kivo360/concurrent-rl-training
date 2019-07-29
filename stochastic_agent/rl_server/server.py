from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Example of running a policy server. Copy this file for your use case.

To try this out, in two separate shells run:
    $ python cartpole_server.py
    $ python cartpole_client.py
"""

import os
from gym import spaces
import numpy as np

import ray
from ray.rllib.agents.ppo import PPOAgent
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.utils.policy_server import PolicyServer
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

SERVER_ADDRESS = "localhost"
SERVER_PORT = 9900
CHECKPOINT_FILE = "last_checkpoint.out"


class TradingServing(ExternalEnv):
    def __init__(self):
        ExternalEnv.__init__(
            self, spaces.Discrete(3), #discrete state == [0, 1, 2] or sell, hold, buy
            spaces.Box(low=-100, high=100, shape=(10, ), dtype=np.float32)) # we'll take an NP array of 10 positions between -100, and 100 to train

    def run(self):
        print("Starting policy server at {}:{}".format(SERVER_ADDRESS,
                                                       SERVER_PORT))
        server = PolicyServer(self, SERVER_ADDRESS, SERVER_PORT)
        server.serve_forever()


if __name__ == "__main__":
    ray.init()
    register_env("srv", lambda _: TradingServing())

    # We use PPO since it supports off-policy actions, but you can choose and
    # configure any agent.
    ppo = PPOAgent(
        env="srv",
        config={
            # Use a single process to avoid needing to set up a load balancer
            "num_workers": 0,
            # "num_gpus": 1
            # We can set GPUs later ... dog ... keep it real.
            # Configure the agent to run short iterations for debugging
            # "exploration_fraction": 0.01,
            # "learning_starts": 100,
            "model":{
                "use_lstm": True
            },
            "timesteps_per_iteration": 200,
        })

    # Attempt to restore from checkpoint if possible.
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint_path = open(CHECKPOINT_FILE).read()
        print("Restoring from checkpoint path", checkpoint_path)
        ppo.restore(checkpoint_path)

    # Serving and training loop
    while True:
        print(pretty_print(ppo.train()))
        checkpoint_path = ppo.save()
        print("Last checkpoint", checkpoint_path)
        with open(CHECKPOINT_FILE, "w") as f:
            f.write(checkpoint_path)