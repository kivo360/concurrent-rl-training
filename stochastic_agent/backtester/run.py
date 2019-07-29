import random
import time
from queue import Queue
from random import choice

from crayons import blue

from stochastic_agent.backtester.components.parallel.policy import \
    start_episode
from stochastic_agent.backtester.components.events import MarketEvent
from stochastic_agent.backtester.components.generators.coins import \
    create_coins
from stochastic_agent.backtester.components.workers import (MarketWorker,
                                                            NextWorker,
                                                            RewardWorker,
                                                            SignalWorker)
from funhandler import InMemoryHandler


def main():
	# Create episodes to run iterate on. 100 episodes
	# It gets the ID from the client
	episodes = [start_episode() for x in range(100)]


	# we handle the stochastic episodes using the memory bits here
	inmem = InMemoryHandler() 


	# We create the coins here (stochastic variables)
	create_coins(inmem, episodes)
	



	# Create an event_queue here
	event_queue = Queue()

	# We create concurrent processing pools here to iterate on.
	market_pool = [MarketWorker(event_queue, handler=inmem, interval=0) for _ in range(2)]
	signal_pool = [SignalWorker(event_queue, interval=0) for _ in range(2)]
	reward_pool = [RewardWorker(event_queue, interval=0) for _ in range(2)]
	next_pool 	= [NextWorker(event_queue, handler=inmem, interval=0) for _ in range(3)]

	# Create a market event for each episode
	for episode in episodes:
		market_event = MarketEvent(episode=episode)
		event_queue.put(market_event)

	
	
	start = time.time()
	since_last_event = 0
	while True:
		q_size = event_queue.qsize()
		
		while q_size != 0:
			# print(blue(event_queue.qsize()))

			event = event_queue.get()
			if event is not None:
				end = time.time()
				since_last_event = end - start
				start = end
			if event.event_name == "MARKET":
				choice(market_pool).push(event)
			elif event.event_name == "SIGNAL":
				choice(signal_pool).push(event)
			elif event.event_name == "REWARD":
				choice(reward_pool).push(event)
			elif event.event_name == "NEXT":
				choice(next_pool).push(event)
		

		end = time.time()
		since_last_event = end - start
		# If it's been 5 seconds since we saw an event we quit
		# This is because we presume that 
		if since_last_event > 5.0:
			print("It's been 5 seconds since the last recieved event. QUITTING")
			return

if __name__ == "__main__":
	main()
