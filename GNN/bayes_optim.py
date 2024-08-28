import os
import yaml
import json
import subprocess
import re
from datetime import datetime
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger, ScreenLogger, _get_default_logger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


class BayesianOptimizationExperiment:
    def __init__(self, base_dir='./results', base_name='bay_optim_', config_file='configs/active-speaker-detection/ava_active-speaker/SPELL_lstm.yaml', log_path='./logs.log.json'):
        self.base_dir = base_dir
        self.base_name = base_name
        self.config_file = config_file
        self.log_path = log_path
        self.dir_name = self.get_next_dir_name()
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def get_next_dir_name(self):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        directories = [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]
        indices = []
        for d in directories:
            if d.startswith(self.base_name):
                match = re.search(r'\d+', d[len(self.base_name):])
                if match:
                    indices.append(int(match.group(0)))
        next_index = max(indices) + 1 if indices else 0
        return os.path.join(self.base_dir, f"{self.base_name}{next_index}")

    def train_and_evaluate(self, channel1, channel2, proj_dim, lstm_proj_dim, dropout, lr, wd):
        with open(self.config_file, 'r') as file:
            config = yaml.safe_load(file)

        exp_name = f"SPELL_ASD_lstm_conc_{int(channel1)}_{int(channel2)}_{int(proj_dim)}_{int(lstm_proj_dim)}_{dropout:.1e}_{lr:.1e}_{wd:.1e}"
        config.update({
            'exp_name': exp_name,
            'channel1': int(channel1),
            'channel2': int(channel2),
            'proj_dim': int(proj_dim),
            'lstm_proj_dim': int(lstm_proj_dim),
            'dropout': float(dropout),
            'lr': float(lr),
            'wd': float(wd),
            'num_epoch': 70,
            'use_spf': True,
            'use_ref': False,
            'batch_size': 12,
            'sch_param': 10
        })

        with open(self.config_file, 'w') as file:
            yaml.safe_dump(config, file)

        subprocess.run(['python', 'tools/train_context_reasoning.py', '--cfg', self.config_file, '--root_result', self.dir_name], capture_output=True, text=True)
        return self.run_evaluation(exp_name)

    def run_evaluation(self, exp_name):
        eval_result = subprocess.run(['python', 'tools/evaluate.py', '--exp_name', exp_name, '--eval_type', 'AVA_ASD', '--root_result', self.dir_name], capture_output=True, text=True)
        for line in eval_result.stderr.split('\n'):
            if "evaluation finished:" in line:
                return float(line.split(':')[-1].strip().replace('%', ''))
        return 92.00

    def optimize(self, init_points=10, n_iter=100):
        pbounds = {
            'channel1': (16, 128),
            'channel2': (4, 32),
            'proj_dim': (32, 128),
            'lstm_proj_dim': (32, 128),
            'dropout': (0.0, 0.3),
            'lr': (1e-5, 1e-3),
            'wd': (1e-6, 1e-4),
        }
        optimizer = BayesianOptimization(
            f=lambda **kwargs: self.train_and_evaluate(**kwargs),
            pbounds=pbounds,
            random_state=1,
            allow_duplicate_points=True,
        )

        if os.path.exists(self.log_path):
            print("Loading previous logs")
            load_logs(optimizer, logs=[self.log_path])
            init_points = 0
            n_iter=1000
        
        logger = JSONLogger(path=self.log_path, reset=False)
        def_logger = _get_default_logger(2, False) # Verbose, Constraint
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.subscribe(Events.OPTIMIZATION_START, def_logger)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, def_logger)
        optimizer.subscribe(Events.OPTIMIZATION_END, def_logger)
        
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        return optimizer.max

# Usage
experiment = BayesianOptimizationExperiment()
best_parameters = experiment.optimize()
print("Best parameters:", best_parameters)
