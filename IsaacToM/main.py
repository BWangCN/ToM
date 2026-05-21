"""Top-level entry point. Kept thin so that the Isaac Sim runner controls app
lifecycle (SimulationApp must be the first heavy import).

Usage (inside the Isaac Sim python env):
    conda activate env_isaacsim
    set OPENAI_API_KEY=sk-...
    python -m isaac_env.runner -c config/split_decision.yaml

Or for a silent dry run without the VLM:
    python -m isaac_env.runner -c config/split_decision.yaml --no-vlm
"""
from isaac_env.runner import main

if __name__ == "__main__":
    main()
