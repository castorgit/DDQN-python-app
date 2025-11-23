"""Simple experiment logger to save models, metadata, and logs.

This helper centralizes where experiment outputs are saved and provides
convenient methods for saving model parameters, JSON metadata, simple
text logs, and optional videos. The layout produced is:

<base_dir>/<config_name>/<timestamp>/
    agent/            # saved model parameters and metadata
    training_logs/    # text logs such as training progress

The `base_dir` is supplied by the caller (runner) and may be relative
or absolute.
"""
import os
import time
import json
from typing import Dict, Any, Optional


class ExperimentLogger:
    """Create and manage an experiment output directory.

    Args:
        base_dir: Base directory under which experiment folders will be created.
        config_name: Short name identifying the configuration (used as a
            subdirectory).
    """

    def __init__(self, base_dir: str, config_name: str):
        # Create a clear, consistent layout under base_dir:
        # <base_dir>/<config_name>/<timestamp>/
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        self.experiment_dir = os.path.join(base_dir, config_name, timestamp)
        os.makedirs(self.experiment_dir, exist_ok=True)
        # keep models and logs as explicit subdirectories
        self.models_dir = os.path.join(self.experiment_dir, 'agent')
        os.makedirs(self.models_dir, exist_ok=True)
        self.logs_dir = os.path.join(self.experiment_dir, 'training_logs')
        os.makedirs(self.logs_dir, exist_ok=True)

    def save_meta(self, meta: Dict[str, Any], name: str = 'agent') -> str:
        """Save a JSON metadata file describing an object (e.g., agent).

        Returns the path written.
        """
        path = os.path.join(self.models_dir, f"{name}.meta.json")
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2)
        return path

    def save_model(self, agent, name: str = 'agent') -> str:
        """Ask `agent` to save its weights/state under the models directory.

        The agent is expected to implement a `save(path)` method which will
        create files such as `<path>.meta.json` and `<path>.pt` as needed.
        """
        path = os.path.join(self.models_dir, f"{name}")
        agent.save(path)
        return path

    def save_video(self, frames, fps: int = 30, name: str = 'episode') -> Optional[str]:
        """Write a sequence of frames to an MP4 file using imageio.

        Returns the path written or `None` on failure.
        """
        try:
            import imageio
            out_path = os.path.join(self.experiment_dir, f"{name}.mp4")
            imageio.mimwrite(out_path, frames, fps=fps)
            return out_path
        except Exception:
            return None

    def write_log(self, filename: str, text: str) -> str:
        """Append a single line to a textual log file under `training_logs`.

        Returns the path written.
        """
        path = os.path.join(self.logs_dir, filename)
        with open(path, 'a') as f:
            f.write(text + '\n')
        return path
