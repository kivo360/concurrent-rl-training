from funguauniverse import PNode
from stochastic_agent.backtester.components.events import MarketEvent

class NextWorker(PNode):
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
        

        if self.handler.is_price_in_coin(episode, "general") == False:
                return
        else:
            market_event = MarketEvent(episode=episode)
            self.event_queue.put(market_event)
        

    
    def create_signal_event(self, dataframe):
        pass