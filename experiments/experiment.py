import os

class Experiment:
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        self.experiment_name = cfg.get("name", None)

    def setup(self, log_dir):
        self.log_dir = log_dir
        self.exp_dir = os.path.join(log_dir, "experiments", self.experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)

    def run(self, model, data_loader):
        # Run the experiment
        pass