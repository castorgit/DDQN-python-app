"""Entry point for running experiments using a YAML config."""
import argparse
import os
import sys
import yaml

# Make the project root importable so we can import modules from `src/`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from src.config import load_config
from src.utils.logger import ExperimentLogger
from src.training import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--out', default=os.path.join('..','logs'), help='Base output directory')
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    logger = ExperimentLogger(base_dir=os.path.abspath(args.out), config_name=config_name)

    # Write metadata.log in training_logs
    import datetime
    meta_lines = [
        f"Algorithm: DDQN",
        f"Environment: {cfg.get('env_name', 'CartPole-v1')}",
        f"Run Date: {datetime.datetime.now().strftime('%Y-%m-%d')}",
        f"Seed: {cfg.get('seed', 42)}"
    ]
    meta_path = os.path.join(logger.logs_dir, 'metadata.log')
    meta_content = '\n'.join(meta_lines)
    meta_content += '\n\n# Full configuration\n'
    meta_content += yaml.safe_dump(cfg, sort_keys=False)
    with open(meta_path, 'w') as f:
        f.write(meta_content)

    print('Starting experiment', args.config)
    model_path = train(cfg, logger)
    print('Model saved to', model_path)
    # Show metadata contents for quick inspection
    try:
        print("\n=== metadata.log ===")
        with open(meta_path, 'r') as f:
            print(f.read())
    except Exception as e:
        print("Could not display metadata.log:", e)


if __name__ == '__main__':
    main()
