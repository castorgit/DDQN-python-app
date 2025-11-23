
"""Evaluation helpers for running and recording episodes.

This module contains simple helpers used by the training and evaluation
scripts to run a single episode and optionally collect frames for video
recording.
"""
from typing import Tuple, List
import gymnasium as gym
import numpy as np


def run_episode(env, agent, render: bool = False, max_steps: int = 1000) -> Tuple[float, int, List]:
    """Run a single episode using `agent` on `env`.

    Args:
        env: An environment implementing the Gymnasium API.
        agent: Agent object exposing `select_action(obs, eval_mode=True)`.
        render: If True, collect rendered frames (if supported).
        max_steps: Maximum number of steps to run before terminating.

    Returns:
        A tuple of (total_reward, episode_length, frames) where `frames` is a
        list of rendered frames (may be empty if `render` is False or the
        environment does not support rendering to arrays).
    """
    obs, _ = env.reset()
    total = 0.0
    frames = []
    for t in range(max_steps):
        a = agent.select_action(obs, eval_mode=True)
        obs, reward, terminated, truncated, info = env.step(int(a))
        done = terminated or truncated
        total += float(reward)
        if render:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        if done:
            return total, t + 1, frames
    return total, max_steps, frames


def record_episode_video(env_name: str, agent, max_steps: int = 1000):
    """Run one episode on `env_name` and return frames for video writing.

    Args:
        env_name: Environment id string (e.g., 'CartPole-v1').
        agent: Agent instance used to act deterministically.
        max_steps: Max steps for the episode.

    Returns:
        Tuple (total_reward, episode_length, frames) matching `run_episode`.
    """
    env = gym.make(env_name, render_mode='rgb_array')
    total, length, frames = run_episode(env, agent, render=True, max_steps=max_steps)
    env.close()
    return total, length, frames
