class Event(object):
    def __init__(self, *args, **kwargs):
        self.episode = kwargs.get("episode", None)
    def set_episode(self, episode):
        self.episode = episode



class MarketEvent(Event):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_name = "MARKET"


class SignalEvent(Event):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_name = "SIGNAL"
        self.prices = kwargs.get("prices", None)


class RewardEvent(Event):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_name = "REWARD"


class CheckNext(Event):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_name = "NEXT"