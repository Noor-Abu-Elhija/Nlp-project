import wandb
YOUR_WANDB_USERNAME = "noor25"
project = "NLP2024_PROJECT_kinan-02"

command = [
        "${ENVIRONMENT_VARIABLE}",
        "${interpreter}",
        "NLP/StrategyTransfer.py",
        "${project}",
        "${args}"
    ]

sweep_config = {
    "name": "Final Model",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        "seed": {"values": list(range(1, 10))},
        "ENV_LEARNING_RATE": {"values": [0.001, 0.0001, 1e-05, 4e-05, 0.0004]}
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
