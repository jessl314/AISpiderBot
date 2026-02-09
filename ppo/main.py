def training_loop():
    agent = Agent()
    agent.clear_optimizer()
    agent.learn()
    observation = None
    pred = agent.choose_action(observation)