import threading
from crayons import yellow
from funguauniverse import PNode
from stochastic_agent.backtester.components.events import CheckNext
from stochastic_agent.backtester.components.parallel.policy import log_reward
from stochastic_agent.backtester.components.parallel.technical import reward_calculation


class RewardWorker(PNode):
    def __init__(self, event_queue, *args, **kwargs):
        super().__init__(event_queue, *args, **kwargs)
        # This is watching all futures to check if they're done
        self.future_list = []
        self.future_watcher = threading.Thread(target=self.signal_watch, daemon=True).start()

    def process(self):
        event = self.get_action_from_queue()
        if event is None:
            return
        

        episode = event.__dict__["episode"]
        # print(yellow(episode))
        # Get from episode data
        # Check for the changes using the exact same ideas from before.
        future = reward_calculation(episode, {})
        self.future_list.append(future)
    
    def create_next_event(self, episode=None):
        
        return CheckNext(episode=episode)
    
    def act_on_results(self, reward_obs):
        episode = reward_obs['episode']
        reward = reward_obs['reward']
        print(yellow(f"Reward: {reward}"))
        log_reward(episode, reward)
        # Do something about the action here
        # Create a new event
        next_event = self.create_next_event(episode)
        self.event_queue.put(next_event)

    def signal_watch(self):
        while True:
            # print(self.future_list)
            for index, future in enumerate(self.future_list):
                if future.done():
                    final = self.future_list.pop(index)
                    # print(final)
                    self.act_on_results(final.result())