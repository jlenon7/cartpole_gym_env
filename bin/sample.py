from src.env import env

env.reset()

while True:
    action = env.action_space.sample()

    env.render()

    observation, reward, terminated, info = env.step(action)

    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Info: {info}")

    if terminated:
        break
