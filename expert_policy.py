import random


def expert_episodes_continuous(env, replay_buffer, nb_trajs, render=False):
    """Stores sample from a expert policy in the given replay buffer.

    Args:
        env: The simulated environment.
        replay_buffer: The replay buffer to store each sample.
        nb_trajs: The number of trajectories to generate.
        render: Whether to open rendering window or not.
    """
    for _ in range(nb_trajs):
        state = env.reset()
        is_done = False
        for _ in range(50):
            variation = random.random() / 20
            action = [-1.0 + variation]
            if render:
                env.render(mode='rgb_array')
            next_state, reward, is_done, _ = env.step(action)
            replay_buffer.put((state, action[0], reward, next_state, is_done))
            state = next_state
        while not is_done:
            variation = random.random() / 10
            action = [1.0 + variation]
            if render:
                env.render(mode='rgb_array')
            next_state, reward, is_done, _ = env.step(action)
            replay_buffer.put((state, action[0], reward, next_state, is_done))
            state = next_state
