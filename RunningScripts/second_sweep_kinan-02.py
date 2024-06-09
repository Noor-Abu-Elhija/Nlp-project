import wandb
YOUR_WANDB_USERNAME = "kinan"
project = "NLP2024_PROJECT_kinan-02"

command = [
        "${ENVIRONMENT_VARIABLE}",
        "${interpreter}",
        "StrategyTransfer.py",
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
        "architecture": {"values": ["LSTM", "transformer", "Final_Model", "LogReg", "TransformerAttention",
                                    "TransformerLSTM", "AttentionLSTM", "Attention"]}
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
