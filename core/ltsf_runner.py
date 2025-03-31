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
from . import util
from . import ltsf_lossfunc
from functools import reduce


class LTSFRunner(L.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

        self.indicators_bool = kargs['indicators_list_01']
        self.dataset_name = kargs['dataset_name']
        self.optimal = kargs['optimal']

        self.daily_return_multiplication_train_list = []
        self.daily_return_multiplication_test_list = []

        self.train_losses = []
        self.test_losses = []

    def evaluate_trading_strategy(self, predictions_tomorrow, true_prices_tomorrow, true_prices_today, risk_free_rate=4.34):
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
        downside_returns = []

        # print(f"Number of testing trading days : {len(true_prices_today)}")
        
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

            # Calculate downside return and track downside returns
            downside_return = min(0, daily_return - risk_free_rate)
            downside_returns.append(downside_return)

        # Calculate downside deviation
        downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))

        # Calculate average daily return
        avg_daily_return = np.mean(daily_returns)
        
        # Compile the results into a dictionary
        evaluation_metrics = {
            'daily return': daily_returns,
            'length': len(daily_returns), 
            'average_daily_return': avg_daily_return,
            'cumulative_return': cumulative_return - 1,  # subtract 1 to get the net return
            'loss_days': loss_days,
            'total_profits': total_profits, 
            'downside_deviation': downside_deviation
        }
        
        return evaluation_metrics
    
    def on_test_epoch_end(self):
        if hasattr(self, 'predictions_tomorrow') and hasattr(self, 'true_prices_tomorrow') and hasattr(self, 'true_prices_today'):
            # Evaluate the trading strategy using the full predictions and actual prices
            evaluation_metrics = self.evaluate_trading_strategy(self.predictions_tomorrow, self.true_prices_tomorrow, self.true_prices_today)
            
            # Log the trading strategy evaluation metrics
            self.log('test/length', evaluation_metrics['length'], on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/average_daily_return', evaluation_metrics['average_daily_return'], on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/cumulative_return', evaluation_metrics['cumulative_return'], on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/loss_days', evaluation_metrics['loss_days'], on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/total_profits', evaluation_metrics['total_profits'], on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/downside_deviation', evaluation_metrics['downside_deviation'], on_step=False, on_epoch=True, sync_dist=True)

        if self.optimal:
            self.daily_return_multiplication_test_list = evaluation_metrics['daily return']
            

    def on_train_epoch_end(self):
        # Sort accumulated training predictions by their time index
        if hasattr(self, 'train_accum'):
            time_indices = np.array(self.train_accum['time_indices'])
            sort_idx = np.argsort(time_indices)
            predictions_tomorrow = np.array(self.train_accum['predictions_tomorrow'])[sort_idx]
            true_prices_tomorrow = np.array(self.train_accum['true_prices_tomorrow'])[sort_idx]
            true_prices_today = np.array(self.train_accum['true_prices_today'])[sort_idx]
            
            evaluation_metrics = self.evaluate_trading_strategy(
                predictions_tomorrow, true_prices_tomorrow, true_prices_today
            )
            self.log('train/length', evaluation_metrics['length'], on_epoch=True, sync_dist=True)
            self.log('train/average_daily_return', evaluation_metrics['average_daily_return'], on_epoch=True, sync_dist=True)
            self.log('train/cumulative_return', evaluation_metrics['cumulative_return'], on_epoch=True, sync_dist=True)
            self.log('train/loss_days', evaluation_metrics['loss_days'], on_epoch=True, sync_dist=True)
            self.log('train/total_profits', evaluation_metrics['total_profits'], on_epoch=True,sync_dist=True)
            self.log('train/downside_deviation', evaluation_metrics['downside_deviation'], on_epoch=True,sync_dist=True)

            # Save these metrics for later use by the FinalResultLoggerCallback.
            self.final_train_metrics = evaluation_metrics

        if self.optimal:
            self.daily_return_multiplication_train_list = evaluation_metrics['daily return']


        # Clear the training accumulator for the next epoch
        del self.train_accum

        # log the final loss   
        if self.train_losses:
            self.final_train_loss = self.train_losses[-1]
        else:
            self.final_train_loss = "NA"
        # Clear the training loss list for the next epoch
        self.train_losses = []

    def forward(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y, true, time_index = [_.float() for _ in batch]

        # Extract label from var_y.
        # (Note: var_y is already constructed to carry only the closing price information.)
        label = var_y[:, -self.hparams.pred_len:, 0]

        # Now, call the model and keep all output channels (which is only 1 channel now).
        prediction, confidence = self.model(var_x, marker_x)
        prediction = prediction[:, -self.hparams.pred_len:, :]
        
        true_price_today = var_x[:,-1,sum(self.indicators_bool[:-14])+4-1]
        
        return prediction, label, true_price_today, confidence, time_index

    # new training step for training profit calculation
    def training_step(self, batch, batch_idx):
        prediction, label, true_price_today, confidence, time_index = self.forward(batch, batch_idx)
        loss = self.loss_function(prediction, label, true_price_today, confidence)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_losses.append(loss.item())
        
        # Initialize the accumulator if it doesn't exist yet
        if not hasattr(self, 'train_accum'):
            self.train_accum = {
                'predictions_tomorrow': [],
                'true_prices_tomorrow': [],
                'true_prices_today': [],
                'time_indices': []
            }
        
        # For training, the batch size is >1; accumulate each sample individually
        batch_size = prediction.shape[0]
        for i in range(batch_size):
            self.train_accum['predictions_tomorrow'].append(prediction[i, 0, 0].item())
            self.train_accum['true_prices_tomorrow'].append(label[i, 0].item())
            self.train_accum['true_prices_today'].append(true_price_today[i].item())
            self.train_accum['time_indices'].append(time_index[i].item())
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack and ignore time_index in validation if not needed
        prediction, label, true_price_today, confidence, _ = self.forward(batch, batch_idx)
        loss = self.loss_function(prediction, label, true_price_today, confidence)
        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        prediction, label, true_price_today, confidence, time_index = self.forward(batch, batch_idx)
        mae = torch.nn.functional.l1_loss(prediction, label)
        mse = torch.nn.functional.mse_loss(prediction, label)
        custom_loss = self.loss_function(prediction, label, true_price_today, confidence)
        self.test_losses.append(custom_loss.item())
        mean_error_percentage = torch.mean(torch.abs((label - prediction) / label) * 100)
        self.log('test/mae', mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/mse', mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/custom_loss', custom_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test/error_percentage', mean_error_percentage, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        predicted_price_tomorrow = prediction.item()
        true_price_tomorrow = label.item()
        true_price_today = true_price_today.item()
        confidence_score = confidence.item()
        
        # Track predictions and actual prices for the entire testing period
        if not hasattr(self, 'predictions_tomorrow'):
            self.predictions_tomorrow = []
            self.true_prices_tomorrow = []
            self.true_prices_today = []
            self.confidences = []
            self.custom_losses = []
        
        # Accumulate predictions and actual prices
        self.predictions_tomorrow.append(predicted_price_tomorrow)
        self.true_prices_tomorrow.append(true_price_tomorrow)
        self.true_prices_today.append(true_price_today)
        self.confidences.append(confidence_score)
        self.custom_losses.append(custom_loss.item())

    def configure_loss(self):
        self.loss_function = ltsf_lossfunc.MSEPenaltyLoss(penalty_factor=5.0)
        
    def configure_optimizers(self):
        if self.hparams.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.optimizer_weight_decay)
        elif self.hparams.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.lr, betas=self.hparams.optimizer_betas, weight_decay=self.hparams.optimizer_weight_decay)
        elif self.hparams.optimizer == 'LBFGS':
            optimizer = torch.optim.LBFGS(self.parameters(), lr=self.hparams.lr, max_iter=self.hparams.lr_max_iter)
        else:
            raise ValueError('Invalid optimizer type!')

        if self.hparams.lr_scheduler == 'StepLR':
            lr_scheduler = {
                "scheduler": lrs.StepLR(
                    optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.lr_gamma, verbose=True)
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

    def train_plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, marker='o', label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss vs Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/train_loss_{self.dataset_name}.png")
        plt.close()

    def test_plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.test_losses) + 1), self.test_losses, marker='o', label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Testing Loss vs Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/test_loss_{self.dataset_name}.png")
        plt.close()