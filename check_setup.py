"""
Run this before training to verify every component works.
Each check prints PASS or FAIL with a reason.
"""
import sys

PASS = '\u2713 PASS'
FAIL = '\u2717 FAIL'


def check(label, fn):
    try:
        result = fn()
        msg = f'  ({result})' if result else ''
        print(f'{PASS}  {label}{msg}')
        return True
    except Exception as e:
        print(f'{FAIL}  {label}')
        print(f'       {type(e).__name__}: {e}')
        return False


print('\n========== check_setup.py ==========\n')

# ── 1. Imports ────────────────────────────────────────────────────────────────
print('-- Imports --')
check('numpy',       lambda: __import__('numpy').__version__)
check('torch',       lambda: __import__('torch').__version__)
check('tensorboard', lambda: __import__('tensorboard').__version__)
check('robosuite',   lambda: __import__('robosuite').__version__)
gym_ok = check('gymnasium (or gym)',
               lambda: __import__('gymnasium').__version__
               if __import__('importlib').util.find_spec('gymnasium')
               else __import__('gym').__version__)

# ── 2. CUDA ───────────────────────────────────────────────────────────────────
print('\n-- CUDA --')
import torch as T
cuda_ok = check('CUDA available', lambda: (
    f'{T.cuda.get_device_name(0)}, CUDA {T.version.cuda}'
    if T.cuda.is_available() else (_ for _ in ()).throw(
        RuntimeError('No CUDA device found — training will run on CPU (slow)'))
))
if not cuda_ok:
    print('       Training will still work on CPU, just much slower.')

# ── 3. Local modules ──────────────────────────────────────────────────────────
print('\n-- Local modules --')
buf_ok  = check('buffer.py imports',    lambda: __import__('buffer') and 'ok')
net_ok  = check('networks.py imports',  lambda: __import__('networks') and 'ok')
td3_ok  = check('td3_torch.py imports', lambda: __import__('td3_torch') and 'ok')

# ── 4. Environment ────────────────────────────────────────────────────────────
print('\n-- Robosuite environment --')
env = None

def make_env():
    import robosuite as suite
    from robosuite.wrappers import GymWrapper
    from robosuite.controllers.composite.composite_controller_factory import (
        refactor_composite_controller_config,
    )
    part_cfg = suite.load_part_controller_config(default_controller='JOINT_VELOCITY')
    ctrl_cfg = refactor_composite_controller_config(part_cfg, 'Panda', ['right'])
    e = suite.make(
        'Door',
        robots='Panda',
        controller_configs=ctrl_cfg,
        has_renderer=False,
        use_camera_obs=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )
    return GymWrapper(e)

env_ok = check('create Door + GymWrapper', make_env)

if env_ok:
    env = make_env()
    check('observation_space.shape', lambda: str(env.observation_space.shape))
    check('action_space.shape',      lambda: str(env.action_space.shape))

    def do_reset():
        r = env.reset()
        obs = r[0] if isinstance(r, tuple) else r
        return f'obs shape {obs.shape}'
    check('env.reset()', do_reset)

    def do_step():
        import numpy as np
        r = env.reset()
        obs = r[0] if isinstance(r, tuple) else r
        action = env.action_space.sample()
        result = env.step(action)
        n = len(result)
        return f'{n}-tuple step return'
    check('env.step(random action)', do_step)

# ── 5. Networks ───────────────────────────────────────────────────────────────
print('\n-- Networks --')
if env_ok and net_ok:
    from networks import ActorNetwork, CriticNetwork
    obs_shape = env.observation_space.shape
    n_act = env.action_space.shape[0]

    check('CriticNetwork creates', lambda: CriticNetwork(obs_shape, n_act) and 'ok')
    check('ActorNetwork creates',  lambda: ActorNetwork(obs_shape, n_actions=n_act) and 'ok')

# ── 6. Agent (full trial run) ─────────────────────────────────────────────────
print('\n-- Agent trial (10 steps) --')
if env_ok and td3_ok:
    from td3_torch import Agent

    def trial_run():
        import numpy as np
        e = make_env()
        agent = Agent(
            actor_learning_rate=0.001,
            critic_learning_rate=0.001,
            tau=0.005,
            input_dims=e.observation_space.shape,
            env=e,
            n_actions=e.action_space.shape[0],
            batch_size=128,
        )
        r = e.reset()
        obs = r[0] if isinstance(r, tuple) else r
        for _ in range(10):
            action = agent.choose_action(obs)
            result = e.step(action)
            if len(result) == 5:
                next_obs, reward, term, trunc, _ = result
                done = term or trunc
            else:
                next_obs, reward, done, _ = result
            agent.remember(obs, action, reward, next_obs, done)
            agent.learn()   # won't learn yet (buffer too small), but shouldn't crash
            obs = next_obs
            if done:
                break
        e.close()
        return '10 steps ok'

    check('Agent: 10 env steps + learn()', trial_run)

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n=====================================')
print('If all checks show PASS, run:  python main.py')
print('Then monitor:                  tensorboard --logdir logs --port 6006')
print('To watch the robot:            python test.py  (after some training)')
print('=====================================\n')

if env and hasattr(env, 'close'):
    try:
        env.close()
    except Exception:
        pass
