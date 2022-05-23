import numpy as np
import torch
from torch import nn



class simulation_forecast:
    def __init__(
        self,
        arr,
        new_model,
        
        inputs_mean,
        inputs_std,
        updates_mean,
        updates_std,
        timesteps=500
    ):
        self.arr = arr
      
        self.inputs_mean = inputs_mean
        self.inputs_std = inputs_std
        self.updates_mean = updates_mean
        self.updates_std = updates_std
        self.model = new_model
        self.moment_preds = []
        self.updates_prev = None
        self.real_updates = []
        self.timesteps=timesteps

    def setup(self):

        arr = self.arr.astype(np.float32)
   
    def test(self):
        self.setup()
        self.model_params = self.sim_data[-3:]
        predictions_updates = self.model.test_step(torch.from_numpy(self.inputs))
        self.moment_calc(predictions_updates)
      
        for i in range(
            1, self.timesteps):

            self.create_input()
            predictions_updates = self.model.test_step(torch.from_numpy(self.inputs))
            self.moment_calc(predictions_updates)

    # For Calculation of Moments
    def calc_mean(self, no_norm, means, stds):
        return (no_norm - means.reshape(-1,)) / stds.reshape(
            -1,
        )

    # For creation of inputs
    def create_input(self):
        tau = self.sim_data[2] / (self.sim_data[2] + self.sim_data[0])

        xc = self.sim_data[0] / (self.sim_data[1] + 1e-8)

        inputs = np.concatenate(
            (
                self.sim_data[0:4].reshape(1, -1),
                tau.reshape(1, -1),
                xc.reshape(1, -1),
                self.model_params.reshape(1, -1),
            ),
            axis=1,
        )
        # new_input_=np.concatenate((predictions_orig_[:,0:],self.model_params.reshape(1,-1),tau.reshape(1,-1),xc.reshape(1,-1)),axis=1)

        self.inputs = self.calc_mean(inputs, self.inputs_mean, self.inputs_std)
        self.inputs = np.float32(self.inputs)

    # For checking updates
    def check_updates(self):
        
        if self.updates[0, 0] > 0:
            self.updates[0, 0] = 0

        if self.updates[0, 2] < 0:
            self.updates[0, 2] = 0

        if self.updates[0, 1] > 0:
            self.updates[0, 1] = 0

    def check_preds(self):

        if self.preds[0, 0] < 0:
            self.preds[0, 0] = 0

        if self.preds[0, 2] < 0:
            self.preds[0, 2] = 0

        if self.preds[0, 2] > self.model_params[0]:
            self.preds[0, 2] = self.model_params[0]

        if self.preds[0, 1] < 0:
            self.preds[0, 1] = 0

        if self.preds[0, 3] < 0:
            self.preds[0, 3] = 0

        self.preds[:, 0] = self.model_params[0] - self.preds[:, 2]

    def moment_calc(self, predictions_updates):
        self.updates = (
            predictions_updates.detach().numpy() * self.updates_std
        ) + self.updates_mean
        self.check_updates()

        self.preds = self.sim_data[0:4] + (self.updates * 20)
        self.check_preds()
        # print(self.updates)
        self.moment_preds.append(self.preds)
        self.sim_data = self.preds.reshape(
            -1,
        )
        self.updates_prev = self.updates


