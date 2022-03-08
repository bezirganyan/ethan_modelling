
# Ethan Spatio-temporal modelling

A framework written for experiments in scope of my Master's thesis in
"Ethical AI for pandemic control using Spatio-temporal features" 
## Installation

Clone the project

```bash
  git clone https://github.com/bezirganyan/ethan_modelling
```

Go to the project directory

```bash
  cd ethan_modelling
```

Install dependencies

```bash
  pip3 install -r requirements.txt
```
## Run Locally

### Preprocessing

For Preprocessing the data you need to run the `preprocessing/encounters.py` file
from the command line, or `include the preprocessing.encounters.EncounterProcessor` 
in your code.

The command line interface looks like this:
```
python3 encounters.py [-h] --encounters_dir ENCOUNTERS_DIR [--district_graph_file DISTRICT_GRAPH_FILE]
                      [--data_output_dir DATA_OUTPUT_DIR] [--start_day START_DAY]
                      [--end_day END_DAY] [--verbose] [--mode MODE]
```

here's the interface of using the preprocessing module from inside of your Python code:

```python
from preprocessing.encounters import EncounterProcessor

if __name__ == "__main__":
    ep = EncounterProcessor(encounters_dir=ENCOUNTERS_DIR,
                            district_graph_file=DISTRICT_GRAPH_FILE,
                            data_output_dir=DATA_OUTPUT_DIR,
                            start_day=START_DAY,
                            end_day=END_DAY,
                            verbose=verbose,
                            mode=MODE)
    ep.prepare()
```

### Mobility-based learning

To run the mobility-based learning, you need to preprocess the data with `graph_learning`
mode. After the preprocessing is done, you can run the learning with:

```bash
python3 run_graph_learning.py [-h] [--max_epochs MAX_EPOCHS] [--lr LR] [--hidden HIDDEN]
                              [--batch-size BATCH_SIZE] [--dropout DROPOUT]
                              [--window WINDOW] [--graph-window GRAPH_WINDOW]
                              [--early-stop EARLY_STOP] [--start-exp START_EXP]
                              [--ahead AHEAD] [--sep SEP] [--smooth_window SMOOTH_WINDOW]
                              --labels_path LABELS_PATH --graph_dir GRAPH_DIR
                              [--country COUNTRY] [--data_output_dir DATA_OUTPUT_DIR]

```

### Visit-based learning

To run the visit-based learning, you need to preprocess the data with `tabular_visit`
mode. After the preprocessing is done, you can run the learning with the following models.

#### XGBoost

```bash
python3 run_xgboost_learning.py [-h] --data_path DATA_PATH [--output_dir OUTPUT_DIR]
                                --xgboost_config_file XGBOOST_CONFIG_FILE
                                [--random_seed RANDOM_SEED]

```

#### H2O AutoML

```bash
python3 run_h2o_learning.py [-h] --data_path DATA_PATH [--output_dir OUTPUT_DIR]
                            [--random_seed RANDOM_SEED] [--verbose]

```
#### TabNet

```bash
python3 run_tabnet_learner.py [-h] --data_path DATA_PATH [--output_dir OUTPUT_DIR]
                              --tabnet_config_file TABNET_CONFIG_FILE
                              [--random_seed RANDOM_SEED]
```
### Viewing logs

At this moment the logs are kept in several places.

For the **XGBoost** and **TabNet** models the experiments are loged using **MLFlow**.
To see the logs you need to run the MLFlow server with 
```bash
mlflow ui
```

For the **H2O** model, the logs are kept in **H2O Flows**. The server can be 
started from python shell with
```python
>>> import h2o
>>> h2o.init()
```

For the mobility-based learning, the logs are kept using PyTorch-Lightning and
can be viewed using TensBoard. You can start the TensBoard server with

```bash
tensorboard --logdir lightning_logs
```
