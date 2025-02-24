import importlib
import inspect
import os

import lightning.pytorch as L
from lightning.pytorch import Trainer
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
import matplotlib.pyplot as plt
from . import ltsf_lossfunc

class LTSFRunner(L.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

        # Load the scaler info which should include 'min' and 'max'
        #stat = np.load(os.path.join(self.hparams.data_root, self.hparams.dataset_name, 'var_scaler_info.npz'))
        stat = np.load(os.path.join(self.hparams.data_root, 'var_scaler_info.npz'))

        # Assuming 'min' and 'max' are stored in the file instead of 'mean' and 'std'
        self.register_buffer('min', torch.tensor(stat['min']).float())
        self.register_buffer('max', torch.tensor(stat['max']).float())

        # Use the closing price channel (index 3) for output de-normalization.
        # close_index = 3  # Index of Close price in OHLC
        # min_close = self.min[close_index].view(1, 1, 1).cuda()   # Shape: [1, 1, 1]
        # max_close = self.max[close_index].view(1, 1, 1).cuda()   # Shape: [1, 1, 1]
        # self.model.rev_output.set_statistics(min_close, max_close)

        # To record the train and val loss for each epoch 
        self.train_losses = []

    def evaluate_trading_strategy(self, predictions_tomorrow, true_prices_tomorrow, true_prices_today):
        """
        Evaluates the trading strategy based on the model's predictions.
        
        :param predictions: List of predicted next day closing prices (T+1) for the entire period
        :param true_prices: List of actual closing prices for the current day (T+1) for the entire period
        
        :return: a dictionary with average daily return, cumulative return, and the number of days with losses
        """
        # Initialize variables
        daily_returns = []
        cumulative_return = 1.0  # Start with a base value of 1 (100% initial investment)
        total_profits = 0
        loss_days = 0

        print(f"Number of testing trading days : {len(true_prices_today)}")
        
        # Loop through predictions and actual prices
        for i in range(len(predictions_tomorrow)):  # Loop till second last day to avoid out of range on true_prices[i+1]
            predicted_price = predictions_tomorrow[i]
            true_price_tomorrow = true_prices_tomorrow[i]  # Actual price for the next day (i+1)
            true_price_today = true_prices_today[i]

            # If predicted price is higher, long strategy
            if predicted_price > true_price_today:
                profit = true_price_tomorrow - true_price_today # Long position
            # If predicted price is lower, short strategy
            elif predicted_price < true_price_today:
                profit = true_price_today - true_price_tomorrow  # Short position
            else:
                profit = 0  # No profit or loss if predicted = actual price
            
            # Calculate daily return and track loss days
            daily_return = profit / true_price_today  # Return for the day
            daily_returns.append(daily_return)
            
            if daily_return < 0:
                loss_days += 1  # Count the day if there's a loss
            
            # Update cumulative return
            cumulative_return *= (1 + daily_return)
            total_profits += profit

        # Calculate average daily return
        avg_daily_return = np.mean(daily_returns)
        
        # Compile the results into a dictionary
        evaluation_metrics = {
            'average_daily_return': avg_daily_return,
            'cumulative_return': cumulative_return - 1,  # subtract 1 to get the net return
            'loss_days': loss_days,
            'total_profits': total_profits
        }
        
        return evaluation_metrics
    
    def on_test_epoch_end(self):
        """
        After all test steps, evaluate the trading strategy using accumulated predictions and actual prices.
        """
        if hasattr(self, 'predictions_tomorrow') and hasattr(self, 'true_prices_tomorrow') and hasattr(self, 'true_prices_today'):
            # Evaluate the trading strategy using the full predictions and actual prices
            evaluation_metrics = self.evaluate_trading_strategy(self.predictions_tomorrow, self.true_prices_tomorrow, self.true_prices_today)
            
            # Log the trading strategy evaluation metrics
            self.log('test/average_daily_return', evaluation_metrics['average_daily_return'], on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/cumulative_return', evaluation_metrics['cumulative_return'], on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/loss_days', evaluation_metrics['loss_days'], on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/total_profits', evaluation_metrics['total_profits'], on_step=False, on_epoch=True, sync_dist=True)
            
    def forward(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        # Extract label from var_y.
        # (Note: var_y is already constructed to carry only the closing price information.)
        label = var_y[:, -self.hparams.pred_len:, :, 0]
        # Now, call the model and keep all output channels (which is only 1 channel now).
        prediction = self.model(var_x, marker_x)[:, -self.hparams.pred_len:, :]
        # true_price_today is now directly taken from the closing price, which is at index 3 in the original var_x.
        true_price_today = var_x[:, -1, 3]
        # print(f"prediction shape: {prediction.shape} and label shape {label.shape}")
        # print(f"prediction value: \n {prediction}")
        # print(f"label value: \n {label}")
        return prediction, label, true_price_today
        # return prediction, label

    def training_step(self, batch, batch_idx):
        loss = self.loss_function(*self.forward(batch, batch_idx))
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss_function(*self.forward(batch, batch_idx))
        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        # prediction shape: torch.Size([1, 1, 1]), label shape: torch.Size([1, 1, 1])
        # prediction, label = self.forward(batch, batch_idx)
        prediction, label, true_price_today = self.forward(batch, batch_idx)
        mae = torch.nn.functional.l1_loss(prediction, label)
        mse = torch.nn.functional.mse_loss(prediction, label)
        mean_error_percentage = torch.mean(torch.abs((label - prediction) / label) * 100)
        self.log('test/mae', mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/mse', mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/error_percentage', mean_error_percentage, on_step=False, on_epoch=True, sync_dist=True)

        predicted_price_tomorrow = prediction.item()
        true_price_tomorrow = label.item()
        true_price_today = true_price_today.item()
        
        # Track predictions and actual prices for the entire testing period
        if not hasattr(self, 'predictions_tomorrow'):
            self.predictions_tomorrow = []
            self.true_prices_tomorrow = []
            self.true_prices_today = []
        
        # Accumulate predictions and actual prices
        self.predictions_tomorrow.append(predicted_price_tomorrow)
        self.true_prices_tomorrow.append(true_price_tomorrow)
        self.true_prices_today.append(true_price_today)

    def configure_loss(self):
        #self.loss_function = ltsf_lossfunc.MSELossWrapper(reduction='mean')
        self.loss_function = ltsf_lossfunc.MSEPenaltyLoss(penalty_factor=1.0)
        
    def configure_optimizers(self):
        if self.hparams.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.optimizer_weight_decay)
        elif self.hparams.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.95), weight_decay=1e-5)
        elif self.hparams.optimizer == 'LBFGS':
            optimizer = torch.optim.LBFGS(self.parameters(), lr=self.hparams.lr, max_iter=self.hparams.lr_max_iter)
        else:
            raise ValueError('Invalid optimizer type!')

        if self.hparams.lr_scheduler == 'StepLR':
            lr_scheduler = {
                "scheduler": lrs.StepLR(
                    optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.lr_gamma)
            }
        elif self.hparams.lr_scheduler == 'MultiStepLR':
            lr_scheduler = {
                "scheduler": lrs.MultiStepLR(
                    optimizer, milestones=self.hparams.milestones, gamma=self.hparams.gamma)
            }
        elif self.hparams.lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = {
                "scheduler": lrs.ReduceLROnPlateau(
                    optimizer, mode='min', factor=self.hparams.lrs_factor, patience=self.hparams.lrs_patience),
                "monitor": self.hparams.val_metric
            }
        elif self.hparams.lr_scheduler == 'WSD':
            assert self.hparams.lr_warmup_end_epochs < self.hparams.lr_stable_end_epochs < self.hparams.max_epochs

            def wsd_lr_lambda(epoch):
                if epoch < self.hparams.lr_warmup_end_epochs:
                    return (epoch + 1) / self.hparams.lr_warmup_end_epochs
                if self.hparams.lr_warmup_end_epochs <= epoch < self.hparams.lr_stable_end_epochs:
                    return 1.0
                if self.hparams.lr_stable_end_epochs <= epoch <= self.hparams.max_epochs:
                    return (epoch + 1 - self.hparams.lr_stable_end_epochs) / (
                            self.hparams.max_epochs - self.hparams.lr_stable_end_epochs)

            lr_scheduler = {
                "scheduler": lrs.LambdaLR(optimizer, lr_lambda=wsd_lr_lambda),
            }
        else:
            raise ValueError('Invalid lr_scheduler type!')

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def load_model(self):
        model_name = self.hparams.model_name
        Model = getattr(importlib.import_module('.' + model_name, package='core.model'), model_name)
        self.model = self.instancialize(Model)

    def instancialize(self, Model):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        model_class_args = inspect.getfullargspec(Model.__init__).args[1:]  # 获取模型参数
        interface_args = self.hparams.keys()
        model_args_instance = {}
        for arg in model_class_args:
            if arg in interface_args:
                model_args_instance[arg] = getattr(self.hparams, arg)
        return Model(**model_args_instance)

    def inverse_transform_var(self, data):
        return (data * self.std) + self.mean

    def inverse_transform_time_marker(self, time_marker):
        time_marker[..., 0] = time_marker[..., 0] * (int((24 * 60) / self.hparams.freq - 1))
        time_marker[..., 1] = time_marker[..., 1] * 6
        time_marker[..., 2] = time_marker[..., 2] * 30
        time_marker[..., 3] = time_marker[..., 3] * 365

        if "max_event_per_day" in self.hparams:
            time_marker[..., -1] = time_marker[..., -1] * self.hparams.max_event_per_day

        return time_marker

    # def plot_losses(self):
    #     # Plot the loss values after training is complete
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, marker='o', label='Train Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.title('Training Loss vs Epoch')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()