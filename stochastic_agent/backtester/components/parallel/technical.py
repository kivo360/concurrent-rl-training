import time
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

process_pool = ProcessPoolExecutor()



def calculate_reward(episode, variables: dict):
    return {
        "reward": np.random.uniform(0, 100000),
        "episode": episode
    }

def simple_techincal_analysis(episode, dataframe: pd.DataFrame):
    """ We do technical analysis here """
    # time.sleep(random.uniform(1, 5))
    return {
        "episode": episode,
        "technicals": np.random.uniform(-95, 95, size=(10,))
    }

def get_technical_analysis(episode, dataframe: pd.DataFrame):
    return process_pool.submit(simple_techincal_analysis, episode, dataframe)

def reward_calculation(episode: str, variables: dict):
    return process_pool.submit(calculate_reward, episode, variables)
