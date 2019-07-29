import threading
from crayons import green
from funguauniverse import PNode
from stochastic_agent.backtester.components.events import RewardEvent
from stochastic_agent.backtester.components.parallel.technical import get_technical_analysis
from stochastic_agent.backtester.components.parallel.policy import get_action

class SignalWorker(PNode):
    def __init__(self, event_queue, *args, **kwargs):
        super().__init__(event_queue, *args, **kwargs)
        self.future_list = []

        # This is watching all futures to check if they're done
        self.future_watcher = threading.Thread(target=self.signal_watch, daemon=True).start()

    def process(self):
        event = self.get_action_from_queue()
        if event is None:
            return
        

        episode = event.__dict__["episode"]
        prices = event.__dict__["prices"]
        future = get_technical_analysis(episode, prices)
        self.future_list.append(future)

    
    def create_reward_event(self, episode=None):
        print(episode)
        print("\n")        
        return RewardEvent(episode=episode)
    
    def act_on_results(self, ta_obs):
        episode = ta_obs['episode']
        ta_observation = ta_obs['technicals']
        action = get_action(episode, ta_observation)
        print(green(f"Action: {action}", bold=True))
        # Do something about the action here
        # Create a new event
        next_event = self.create_reward_event(episode)
        self.event_queue.put(next_event)

    def signal_watch(self):
        while True:
            for index, future in enumerate(self.future_list):
                if future.done():
                    self.future_list.pop(index)
                    self.act_on_results(future.result())
