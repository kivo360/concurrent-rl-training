def create_coins(handler, episodes):

	coin_creation_list = []
	for episode_id in episodes:
		handler.generate_coin_in_episode(episode_id, "general")
	return "COINS CREATED"