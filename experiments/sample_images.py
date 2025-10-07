from experiments.experiment import Experiment
import torch
import os
import numpy as np
import torchvision.utils as vutils

class SampleImages(Experiment):
    def __init__(self, cfg):
        super().__init__(cfg)


    def run(self, model, loader):
        """
        Runs image logging after training
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        print("Running Image Logger Experiment...")
                
        with torch.no_grad():
            output = model.sample(**self.cfg)

            # If return_steps is True, sample from the model
            if self.cfg.get('return_steps', False):
                samples = output[0]  # Assuming the first element is the samples
                steps = output[-1] # (bs, steps, C, H, W)
            else:
                samples = output
                steps = None

            # Move to [0,1]
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, min=0, max=1)

            # If steps are provided, log them as well
            if steps is not None:
                steps = (steps + 1) / 2
                steps = torch.clamp(steps, min=0, max=1)

        samples_grid = vutils.make_grid(samples, nrow=int(np.sqrt(self.cfg.nsamples)), padding=2)
        # Save image
        save_path = os.path.join(self.exp_dir, f"samples_{len(self.cfg.delta_steps)}steps_{self.cfg.temperature}temp.png")
        vutils.save_image(samples_grid, save_path)
        print(f"Samples saved to {save_path}")

        if steps is not None:   
            sequence_grid = vutils.make_grid(steps.flatten(0, 1), nrow=steps.shape[1], padding=2)
            # Save image
            save_path = os.path.join(self.exp_dir, f"steps_{len(self.cfg.delta_steps)}steps_{self.cfg.temperature}temp.png")
            vutils.save_image(sequence_grid, save_path)
            print(f"Steps saved to {save_path}")
