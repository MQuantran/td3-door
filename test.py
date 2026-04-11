"""
Visual test script — loads saved checkpoints and renders the robot.
Run this separately from main.py (it does NOT save or update any models).
"""
import time
import robosuite as suite
from robosuite.wrappers import GymWrapper

from td3_torch import Agent


def _joint_velocity_config():
    return {
        "type": "BASIC",
        "body_parts": {
            "right": {
                "type": "JOINT_VELOCITY",
                "input_max": 1.0,
                "input_min": -1.0,
                "output_max": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                "output_min": [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                "kp": 100,
                "velocity_limits": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            },
            "gripper": {"type": "GRIP"},
        },
    }


def make_env():
    env = suite.make(
        'Door',
        robots='Panda',
        controller_configs=_joint_velocity_config(),
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
        render_camera='frontview',
    )
    return GymWrapper(env)


def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        obs, reward, done, info = result
    return obs, reward, done, info


if __name__ == '__main__':
    env = make_env()

    agent = Agent(
        actor_learning_rate=0.001,
        critic_learning_rate=0.001,
        tau=0.005,
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=256,
        layer2_size=128,
        batch_size=128,
    )

    agent.load_models()

    n_games = 3

    for episode in range(n_games):
        observation = reset_env(env)
        done = False
        score = 0.0

        while not done:
            env.render()
            # validation=True skips warm-up random actions
            action = agent.choose_action(observation, validation=True)
            next_observation, reward, done, _ = step_env(env, action)
            score += reward
            observation = next_observation
            time.sleep(0.03)   # slow down for human viewing

        print(f'Test episode {episode}  Score: {score:.2f}')

    env.close()
