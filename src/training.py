"""Training loop for the DDQN agent.

This module exposes a `train` function that takes a config dict and an
ExperimentLogger and runs training, saving models and logs to the logger.
"""
import time
from collections import deque
import os
from typing import Dict, Any

from .agent import DDQNAgent
from .evaluation import run_episode


def train(config: Dict[str, Any], logger):
    """Run training according to `config` and record outputs with `logger`.

    Args:
        config: Dictionary of hyperparameters and environment settings.
        logger: An `ExperimentLogger` instance used to save models and logs.

    Returns:
        The filesystem path where the final model was saved.
    """
    env_name = config.get('env_name', 'CartPole-v1')
    env_threshold = config.get('env_threshold')
    try:
        env_threshold = float(env_threshold) if env_threshold is not None else None
    except (TypeError, ValueError):
        env_threshold = None
    if env_threshold is None and env_name.startswith('CartPole'):
        env_threshold = 195.0
    episodes = int(config.get('episodes', 1000))
    max_steps = int(config.get('max_steps', 1000))
    batch_size = int(config.get('batch_size', 64))
    try:
        avg_calc = int(config.get('AVG_CALC', 40))
    except (TypeError, ValueError):
        avg_calc = 40
    avg_calc = max(1, avg_calc)

    env = __import__('gymnasium').make(env_name)
    obs, _ = env.reset()
    obs_shape = obs.shape
    action_size = env.action_space.n

    agent = DDQNAgent(
        observation_shape=obs_shape,
        action_size=action_size,
        hidden_sizes=tuple(config.get('hidden_sizes', (128, 128))),
        lr=float(config.get('lr', 1e-3)),
        gamma=float(config.get('gamma', 0.99)),
        replay_capacity=int(config.get('replay_capacity', 10000)),
    )

    # Report device selection (GPU/CPU) based on agent's torch/device flags
    try:
        if getattr(agent, '_torch_available', False):
            dev = getattr(agent, '_device', None)
            if dev is not None and str(dev).lower().startswith('cuda'):
                device_msg = f'GPU ({dev})'
            else:
                device_msg = f'CPU ({dev})' if dev is not None else 'CPU (torch available)'
        else:
            device_msg = 'CPU (torch not available)'
    except Exception:
        device_msg = 'CPU (device check failed)'
    print('Using device:', device_msg)
    try:
        logger.write_log('training.log', f'Using device: {device_msg}')
    except Exception:
        # logger may not be available in some dry-run contexts; ignore
        pass

    # training hyperparameters
    learn_start = int(config.get('learn_start', 500))
    train_frequency = int(config.get('train_frequency', 1))
    target_update = int(config.get('target_update', 500))
    train_iterations = int(config.get('train_iterations', 4))
    enable_soft_update = bool(config.get('enable_soft_update', True))
    soft_tau = float(config.get('soft_tau', 0.005))
    n_step = int(config.get('n_step', 1))
    n_step = max(1, n_step)

    total_steps = 0
    train_steps = 0
    rewards_window = []

    start_time = time.time()
    logger.write_log('training.log', f'AVG_CALC (average window): {avg_calc}')
    print(f'AVG_CALC (average window): {avg_calc}')
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        n_step_queue = deque()
        for t in range(max_steps):
            action = agent.select_action(obs, eval_mode=False)
            next_obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            n_step_queue.append((obs, action, reward, next_obs, done))
            # when we have n steps (or hit done), compute n-step return for the oldest transition
            if len(n_step_queue) >= n_step or done:
                ret = 0.0
                for idx, (_, _, r, _, dflag) in enumerate(n_step_queue):
                    ret += (agent.gamma ** idx) * float(r)
                    if dflag:  # stop accumulating past terminal within the window
                        break
                first_obs, first_action, _, _, _ = n_step_queue[0]
                last_next_obs = n_step_queue[-1][3]
                last_done = n_step_queue[-1][4]
                effective_n = min(len(n_step_queue), n_step)
                agent.store_transition(first_obs, first_action, ret, last_next_obs, last_done, n_steps=effective_n)
                n_step_queue.popleft()
            obs = next_obs
            ep_reward += float(reward)
            total_steps += 1

            if total_steps >= learn_start and total_steps % train_frequency == 0:
                for _ in range(train_iterations):
                    try:
                        loss = agent.train_step(batch_size=batch_size, gamma=agent.gamma)
                    except RuntimeError:
                        print("Training failed: torch not available or other issue")
                        return
                    train_steps += 1
                    if enable_soft_update and hasattr(agent, 'update_target'):
                        agent.update_target(tau=soft_tau)
                    if train_steps % target_update == 0:
                        agent.update_target(tau=1.0)

            if done:
                break

        # flush any remaining n-step partials at episode end
        while n_step_queue:
            ret = 0.0
            for idx, (_, _, r, _, dflag) in enumerate(n_step_queue):
                ret += (agent.gamma ** idx) * float(r)
                if dflag:
                    break
            first_obs, first_action, _, _, _ = n_step_queue[0]
            last_next_obs = n_step_queue[-1][3]
            last_done = n_step_queue[-1][4]
            effective_n = len(n_step_queue)
            agent.store_transition(first_obs, first_action, ret, last_next_obs, last_done, n_steps=effective_n)
            n_step_queue.popleft()

        rewards_window.append(ep_reward)
        if len(rewards_window) > avg_calc:
            rewards_window.pop(0)
        avg_reward = sum(rewards_window) / len(rewards_window)
        logger.write_log('training.log', f'Episode {ep} reward={ep_reward:.2f} AVG-{avg_calc}={avg_reward:.2f} steps={total_steps}')
        print(f'Episode {ep}/{episodes} reward={ep_reward:.2f} AVG-{avg_calc}={avg_reward:.2f}')

        if env_threshold is not None and len(rewards_window) == avg_calc and avg_reward >= env_threshold:
            # compute elapsed time and report
            elapsed_s = time.time() - start_time
            elapsed_min = elapsed_s / 60.0
            msg = (
                f'Environment solved in {ep} episodes AVG-{avg_calc}={avg_reward:.2f} '
                f'threshold={env_threshold:.2f} Elapsed time: {elapsed_min:.2f} minutes '
                f'(AVG_CALC={avg_calc})'
            )
            print(msg)
            try:
                logger.write_log('training.log', msg)
            except Exception:
                pass
            break

    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60.0
    try:
        logger.write_log(
            'training.log',
            f'Training finished in {elapsed:.1f}s ({elapsed_min:.2f} minutes) AVG_CALC={avg_calc}',
        )
    except Exception:
        pass
    print(f'Training finished. AVG_CALC={avg_calc}. Elapsed time: {elapsed_min:.2f} minutes')
    # final save
    model_path = logger.save_model(agent, name='ddqn_agent')
    logger.save_meta({
        'config': config,
        'elapsed_s': elapsed,
        'total_steps': total_steps,
    }, name='run')
    env.close()
    return model_path
