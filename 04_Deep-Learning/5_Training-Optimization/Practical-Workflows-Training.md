# Practical Workflows - Training

> **Professional training pipelines** - Experiment tracking and Hyperparameter tuning

---

## 📊 Experiment Tracking (Weights & Biases)

Stop using `print()`. Use a proper tracker.

```python
import wandb

# 1. Initialize
wandb.init(project="my-project", config={
    "learning_rate": 0.001,
    "architecture": "CNN",
    "dataset": "CIFAR-10",
    "epochs": 10,
})

config = wandb.config

# 2. Log metrics
for epoch in range(config.epochs):
    train_loss = train_one_epoch()
    val_acc = validate()
    
    wandb.log({
        "epoch": epoch, 
        "loss": train_loss, 
        "val_acc": val_acc
    })

# 3. Finish
wandb.finish()
```

---

## 🎯 Hyperparameter Tuning (Optuna)

Automated search for best parameters.

```python
import optuna

def objective(trial):
    # Suggest parameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    
    model = create_model(dropout)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    # Train and return validation accuracy
    accuracy = train_and_evaluate(model, optimizer)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
```

---

## 🛑 Early Stopping

Stop training when validation loss stops improving.

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
```

---

**Professional workflows: Reproducible and optimized!**
