from loguru import logger
from funguauniverse import PNode
from stochastic_agent.backtester.components.events import CheckNext, SignalEvent

class MarketWorker(PNode):
    def __init__(self, event_queue, *args, **kwargs):
        super().__init__(event_queue, *args, **kwargs)
        self.handler = kwargs.get("handler", None)
        if self.handler is None:
            raise Exception("There's no handler")

    def process(self):
        event = self.get_action_from_queue()
        if event is None:
            return
        

        episode = event.__dict__["episode"]
        # Get the coins for an episode here.
        coin_dataframe = self.handler.pop_coin(
            eid=episode, 
            coin_name="general", 
            limit=100
        )


        next_event = self.create_signal_event(coin_dataframe, episode=episode)
        self.event_queue.put(next_event)
    
    def create_signal_event(self, dataframe, episode=None):
        logger.debug(dataframe)
        return SignalEvent(episode=episode, prices=dataframe)