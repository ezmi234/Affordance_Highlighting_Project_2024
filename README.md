# Affordance Highlighting project
## Part 3 - Affordance Benchmark on the AffordanceNet Dataset

### Fine-Tuning
The code for fine-tuning the hyperparameters is located in the `validation.ipynb` file. It uses Optuna to optimize the hyperparameters on the validation set (`val_set`).

### Testing
For testing, the same code in `validation.ipynb` is used, but the `val_set` is replaced with the `test_set`. The testing is performed using the three models mentioned in the paper report.
