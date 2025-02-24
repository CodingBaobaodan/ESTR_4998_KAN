import torch
import torch.nn as nn

class SimpleProfitLoss(nn.Module):
    def __init__(self, k=1.0):
        super(SimpleProfitLoss, self).__init__()
        self.k = k  # scaling factor for tanh

    def forward(self, predictions, labels, actual_price_today):
        """
        Compute a trading-oriented loss for a single-step forecast.

        Args:
            predictions (torch.Tensor): shape [batch_size, 1, 1], predicted price for T+1
            labels (torch.Tensor):      shape [batch_size, 1, 1], actual price for T+1
            actual_price_today (torch.Tensor): shape [batch_size], the price at T

        Returns:
            torch.Tensor: scalar loss = -mean(profit).
        """
        # Flatten out the extra dimensions since pred_len = 1
        pred_tomorrow   = predictions[:, 0, 0]  # shape [batch_size]
        actual_tomorrow = labels[:, 0, 0]       # shape [batch_size]

        # print(f"pred_tomorrow: {pred_tomorrow}, actual_tomorrow: {actual_tomorrow}, actual_price_today: {actual_price_today}")

        # Compute the predicted difference and actual difference vs. today
        predicted_diff  = pred_tomorrow - actual_price_today      # shape [batch_size]
        actual_diff     = actual_tomorrow - actual_price_today    # shape [batch_size]

        # Smooth sign function => "actions" ∈ (-1, +1)
        actions = torch.tanh(self.k * predicted_diff)

        # Profit = actions * actual price change
        profit = actions * actual_diff  # shape [batch_size]

        # Loss = negative mean profit => maximizing profit
        loss = -torch.mean(profit)
        return loss

class MSEPenaltyLoss(nn.Module):
    """
    This loss function combines the traditional Mean Squared Error (MSE)
    with an additional penalty when the predicted price change direction is incorrect.

    Formally, let:
        predicted_diff = pred_tomorrow - actual_price_today
        actual_diff    = actual_tomorrow - actual_price_today

    The loss is defined as:
        Loss = MSE(pred_tomorrow, actual_tomorrow) +
               penalty_factor * (actual_diff^2)   if predicted_diff and actual_diff have opposite signs,
               otherwise just the MSE loss.

    Args:
        penalty_factor (float): A scaling factor for the penalty term.
                                Higher values make the loss more sensitive to wrong direction.
    """
    def __init__(self, penalty_factor=1.0):
        super(MSEPenaltyLoss, self).__init__()
        self.penalty_factor = penalty_factor

    def forward(self, predictions, labels, true_price_today, confidence):
        """
        Args:
            predictions (torch.Tensor): shape [batch_size, 1, 1],
                                        predicted price for time T+1.
            labels (torch.Tensor): shape [batch_size, 1, 1],
                                   actual price for time T+1.
            actual_price_today (torch.Tensor): shape [batch_size],
                                               the price at time T.
        Returns:
            torch.Tensor: A scalar loss computed as MSE + penalty (if wrong direction).
        """
        # Flatten extra dimensions (since pred_len = 1)
        pred_tomorrow = predictions[:, 0, 0]   # [batch_size]
        actual_tomorrow = labels[:, 0, 0]        # [batch_size]
        
        # Standard MSE loss for T+1 price prediction
        mse_loss = nn.functional.mse_loss(pred_tomorrow, actual_tomorrow)
        
        # Compute differences relative to today's price
        predicted_diff = pred_tomorrow - true_price_today  # predicted change from T to T+1
        actual_diff = actual_tomorrow - true_price_today     # actual change from T to T+1

        # Create a mask where the predicted and actual differences have opposite signs.
        # When multiplied together, a negative product means they are of opposite sign.
        penalty_mask = (predicted_diff * actual_diff < 0).float()  # 1 if wrong direction, 0 otherwise
        
        # Compute the extra penalty (scaled by the square of actual_diff)


        penalty_loss = (penalty_mask * self.penalty_factor * (actual_diff ** 2)).mean()
        
        # Total loss: Use MSE if direction is correct; otherwise, add the penalty.
        return mse_loss + penalty_loss

class MSELossWrapper(nn.Module):
    """
    A simple wrapper around nn.MSELoss that forces the interface to:
    loss(predictions, labels, actual_price_today)
    and simply ignores the third parameter.
    """
    def __init__(self, reduction='mean'):
        super(MSELossWrapper, self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, predictions, labels, actual_price_today, confidence):
        return self.mse(predictions, labels)