import random
from typing import Tuple 
import numpy as np
from arch import arch_model
import pandas as pd
from model_fitting import ModelFitting

# NOTE: This code is translated from Gibbes & Candes paper with some help from a GPT model, which was manually
# reviewed for correctness. The original paper is available at https://arxiv.org/abs/2306.05524.
class DtACI(ModelFitting):
    def __init__(self, data):
        super().__init__(data)

    def pinball(self, u, alpha):
        u = np.array(u)  # ensures u is a NumPy array
        return alpha * u - np.minimum(u, 0)


    ### Input values are the sequence beta_t, the target level, and the sequence of candidate gammas.
    ### Return value is a list containing the vectors alpha_t, err_t(alpha_t), err_t(alpha), 
    ### gamma_t, alphabar_t, err_t(alphabar_t), gammabar_t. Here we use the notation err_t(x)
    ### to refer to the errors computed using input x.
    def argarch_conformal_adapt_stable(self, betas, alpha, gammas, *, eta_adapt = False, sigma = None, eta = None, I=100):
        """
        Adaptive conformal inference using expert weighting and online updates.
        
        Parameters:
        - betas: np.array of length T (observations or thresholds)
        - gammas: np.array of length k (step sizes for each expert)
        - alpha_init: initial alpha value for all experts
        - alpha: fixed alpha value for fixed error sequence calculation
        - eta_lookback: lookback window size for eta adaptation
        - eta_adapt: boolean, whether to adapt eta or not
        - sigma: mixing parameter for expert weights update
        
        Returns:
        A tuple of:
        - alpha_seq: sequence of adaptive alpha values over time
        - err_seq_adapt: adaptive error sequence (0/1) over time
        - err_seq_fixed: fixed error sequence (0/1) over time
        - gamma_seq: gamma values used by the selected expert over time
        - mean_alpha_seq: weighted mean alpha over experts at each time
        - mean_err_seq: error sequence based on mean alpha
        - mean_gammas: weighted mean gammas over experts at each time
        """
        T = len(betas)           # Number of time points
        k = len(gammas)          # Number of experts
        alpha_init = alpha      # Initial alpha value for experts

        # Initialize sequences for storing results
        alpha_seq = np.full(T, alpha_init, dtype=float)  # Adaptive alpha sequence
        err_seq_adapt = np.zeros(T, dtype=int)           # Adaptive error (0/1)
        err_seq_fixed = np.zeros(T, dtype=int)           # Fixed error (0/1)
        gamma_seq = np.zeros(T, dtype=float)              # Gamma selected at each time
        mean_alpha_seq = np.zeros(T, dtype=float)         # Weighted mean alpha
        mean_err_seq = np.zeros(T, dtype=int)             # Error based on mean alpha
        mean_gammas = np.zeros(T, dtype=float)            # Weighted mean gamma
        loss_seq = np.zeros(T, dtype=float)               # Total loss at each time

        # Initialize expert parameters
        expert_alphas = np.full(k, alpha_init, dtype=float)  # Alphas for each expert
        expert_ws = np.ones(k, dtype=float)                   # Weights for each expert
        expert_cumulative_losses = np.zeros(k, dtype=float)  # Cumulative losses for each expert
        expert_probs = np.full(k, 1.0/k, dtype=float)           # Probabilities for selecting experts

        # Randomly select current expert initially
        cur_expert = np.random.choice(k)

        # initialize eta if adapting
        if sigma is None:
            sigma = 1/(2.0 * I)
        
        if eta is None:
            eta = np.sqrt(3.0/I) * np.sqrt( (np.log(2*k*I) + 1) / ((1-alpha)**2 * alpha**2) )  # set an initial eta value

        for t in range(T):
            # adapt eta if required and enough history available
            if t >= I and eta_adapt:
                loss_window = loss_seq[(t - I):t]
                denom = np.sum(loss_window**2)
                eta = np.sqrt((np.log(2*k*I) + 1) / denom) if denom > 0 else np.inf

            alphat = expert_alphas[cur_expert]       # alpha from current expert
            alpha_seq[t] = alphat                    # store adaptive alpha

            # Calculate adaptive error indicator: 1 if alphat > betas[t], else 0
            err_seq_adapt[t] = int(alphat > betas[t])
            # Calculate fixed error indicator using fixed alpha
            err_seq_fixed[t] = int(alpha > betas[t])
            # Store gamma value used by current expert
            gamma_seq[t] = gammas[cur_expert]

            # Compute weighted means over experts
            mean_alpha = np.sum(expert_probs * expert_alphas)
            mean_alpha_seq[t] = mean_alpha
            mean_err_seq[t] = int(mean_alpha > betas[t])
            mean_gammas[t] = np.sum(expert_probs * gammas)

            # Compute losses for each expert using pinball loss
            expert_losses = self.pinball(betas[t] - expert_alphas, alpha)
            loss_seq[t] = np.sum(expert_losses * expert_probs)

            # Update expert alphas by gradient-like step
            expert_alphas = expert_alphas + gammas * (alpha - (expert_alphas > betas[t]).astype(float))

            if eta < np.inf:
                # Weighted expert update with exponential weighting
                expert_bar_ws = expert_ws * np.exp(-eta * expert_losses)
                # Add mixing to avoid zero weights (sigma)
                expert_next_ws = (1 - sigma) * expert_bar_ws / np.sum(expert_bar_ws) + sigma / k

                # Normalize to get new expert probabilities
                expert_probs = expert_next_ws / np.sum(expert_next_ws)

                # Sample new current expert based on updated probabilities
                cur_expert = np.random.choice(k, p=expert_probs)
                expert_ws = expert_next_ws
            else:
                # If eta not adapted (infinite), choose expert with minimum cumulative loss
                expert_cumulative_losses += expert_losses
                cur_expert = np.argmin(expert_cumulative_losses)

        return alpha_seq, err_seq_adapt, err_seq_fixed, gamma_seq, mean_alpha_seq, mean_err_seq, mean_gammas
    

    ### Compute sequence of conformity scores by fitting an AR-GARCH model. 
    ### Returns is a sequence of stock market returns, lookback specifies the dataset \{R_s\}_{t-lookback+1 \leq s \leq t-1}
    ### used to fit the model and make predictions at time t, startup specifies the first time we make a prediction set,
    ### badscores = FALSE means we use the conformity scores S_t := |V_t- \hat{\sigma}^2_t|/\hat{\sigma}^2_t else we use
    ### S_t := |V_t- \hat{\sigma}^2_t|
    def argarch_conformal_forecasting_compute_scores(self, score_type, *, return_col = 'log_highlow_diff', lookback=300, ar=1, garch_p=1, garch_q=1, start_up = 100):
        # check if score_type is valid
        if score_type not in ["gamma", "residual_normalized", "absolute"]:
            raise ValueError("score_type must be one of 'gamma', 'residual_normalized', or 'absolute'")
        
        returns = self.data[return_col].values
        T = len(returns)
        start_up = max(start_up, lookback) # where to start the predictions
        scores = []
        mean_seq, variance_seq = [], []
        
        for t in range(start_up, T):

            # fit the AR-GARCH model
            # argarch_model = arch_model(returns[(t - lookback):t], mean='ARX', lags=ar, vol='GARCH', p=garch_p, q=garch_q, dist='t')
            if garch_p == 0:
                argarch_model = arch_model(returns[(t - lookback):t], mean="ARX", lags=ar, vol="ARCH", p=garch_q, dist="t")
            else:
                argarch_model = arch_model(returns[(t - lookback):t], mean='ARX', lags=ar, vol='GARCH', p=garch_p, q=garch_q, dist='t')
            argarch_fit = argarch_model.fit(disp="off")

            # forecast the next variance
            forecast = argarch_fit.forecast(horizon=1)
            forecast_mean = forecast.mean.iloc[-1, 0]
            forecast_variance = forecast.variance.iloc[-1, 0]
            forecast_std = np.sqrt(forecast_variance)
            
            # compute conformity scores
            if score_type == "residual_normalized":
                scores.append(abs((returns[t] - forecast_mean)/forecast_std))
            elif score_type == "absolute":
                scores.append(abs(returns[t] - forecast_mean))
            elif score_type == "gamma":
                # compute the gamma score
                scores.append(abs((returns[t] - forecast_mean) / forecast_mean))
      
            # store mean
            mean_seq.append(forecast_mean)

            # store variance
            variance_seq.append(forecast_variance)
            
        
        return scores, mean_seq, variance_seq
    
    def build_conformal_intervals(self, scores, mean_seq, variance_seq, alpha_seq, score_type, *, return_col = 'log_highlow_diff', lookback=300, burn_ind=100):
        """
        Construct conformal prediction intervals using precomputed scores, means, variances, and adaptive alpha sequence.

        Parameters:
        - scores: list or np.array of conformity scores (length N)
        - mean_seq: list or np.array of forecast means (length N)
        - variance_seq: list or np.array of forecast variances (length N)
        - alpha_seq: list or np.array of adaptive alpha values (length N)
        - score_type: str, type of score used ('gamma', 'residual_normalized', or 'absolute')
        - return_col: str, column name in self.data containing actual values (default 'log_highlow_diff')
        - lookback: int, number of previous scores to consider for beta calculation (default 300)
        - burn_ind: int, index to start the intervals from (default 100)

        Returns:
        - intervals_df: pd.DataFrame with columns ['lower_bound', 'upper_bound', 'actual', 'forecast', 'forecast_std', 'alpha', 'within_CI']
        """
        N = len(scores)
        
        # actual data is available in self.data[return_col] if it exists
        actuals = self.data.iloc[burn_ind + len(self.data) - N:][return_col].values if hasattr(self, 'data') and return_col in self.data else [np.nan]*N

        # get the forecasted standard deviation
        std_seq = np.sqrt(variance_seq)
        lower_bounds, upper_bounds = [], []

        i = burn_ind
        while i < N: # iterate through the scores
            alpha_t = alpha_seq[i]
            subset_scores = scores[max(0, i - lookback):i]  # Use a lookback of 300 for beta calculation
            alpha_quantile = np.quantile(subset_scores, 1 - max(alpha_t, 0))
            forecast_mean = mean_seq[i]
            forecast_std = std_seq[i]

            if score_type == "residual_normalized":
                # Residual normalized intervals
                lower_bound = forecast_mean - forecast_std * alpha_quantile
                upper_bound = forecast_mean + forecast_std * alpha_quantile
                lower_bounds.append(lower_bound)
                upper_bounds.append(upper_bound)
            elif score_type == "absolute":
                # Absolute intervals
                lower_bound = forecast_mean - alpha_quantile
                upper_bound = forecast_mean + alpha_quantile
                lower_bounds.append(lower_bound)
                upper_bounds.append(upper_bound)
            else: # gamma score
                # Gamma score intervals
                lower_bound = forecast_mean -  alpha_quantile * np.abs(forecast_mean)
                upper_bound = forecast_mean +  alpha_quantile * np.abs(forecast_mean)
                lower_bounds.append(lower_bound)
                upper_bounds.append(upper_bound)

            i += 1

        # create a DataFrame to hold the intervals and other information
        intervals_df = pd.DataFrame({
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds,
            'actual': actuals,
            'forecast': mean_seq[burn_ind:],
            'forecast_std': std_seq[burn_ind:],
            'alpha': [max(0, x) for x in alpha_seq[burn_ind:]]
        })
        intervals_df['within_CI'] = (intervals_df['actual'] >= intervals_df['lower_bound']) & (intervals_df['actual'] <= intervals_df['upper_bound'])

        return intervals_df
    

    ### Use a binary search to find the lowest quantile of recentScores that is above curScore
    ### Epsilon gives numerical error tolerance in the binary search
    def find_beta(self, recent_scores: list, cur_score: float, epsilon=0.001):
        """
        Finds the smallest quantile (beta) such that the (1-beta) quantile of recent_scores
        is greater than cur_score, using binary search for efficiency.

        Parameters:
        - recent_scores: array-like, the sequence of recent conformity scores
        - cur_score: float, the current conformity score to compare against
        - epsilon: float, numerical tolerance for the binary search

        Returns:
        - mid: float, the estimated quantile (beta) value
        """
        top = 1      # upper bound for quantile search
        bot = 0      # lower bound for quantile search
        mid = (top + bot) / 2
        while top - bot > epsilon:
            # Compute the (1-mid) quantile of recent_scores
            if np.quantile(recent_scores, 1 - mid) > cur_score:
                # If quantile is above cur_score, search higher quantiles (increase beta)
                bot = mid
                mid = (top + bot) / 2
            else:
                # Otherwise, search lower quantiles (decrease beta)
                top = mid
                mid = (top + bot) / 2
        return mid

    ### Given a sequence of conformity scores compute the corresponding values for beta_t
    ### Epsilon gives numerical error tolerance in a binary search
    def garch_conformal_forecasting_compute_betas(self, scores, lookback=np.inf, epsilon=0.001):
        """
        Computes the sequence of beta_t values for a given sequence of conformity scores.
        For each time t, beta_t is the lowest quantile such that the (1-beta_t) quantile
        of the recent scores is above the current score.

        Parameters:
        - scores: array-like, sequence of conformity scores
        - lookback: int or np.Inf, number of previous scores to consider (window size)
        - epsilon: float, numerical tolerance for the binary search

        Returns:
        - beta_seq: np.array, sequence of beta_t values
        """
        T = len(scores)

        # Initialize array to store beta values
        beta_seq = np.zeros(T)
        
        for t in range(1, T):
            # Determine the window of recent scores to use for quantile calculation
            recent_scores = scores[max(t - lookback, 0):t]
            # Compute beta for current score using binary search
            beta_seq[t - 1] = self.find_beta(recent_scores, scores[t], epsilon)
        
        return beta_seq
    
    def get_dtaci_forecast_df(self, score_type: str, *, alpha = 0.05, lookback=300, start_up=100, burn_ind = 100, ar=1, garch_p=1, garch_q=1, sigma = None, eta = None, I=100, eta_adapt=False) -> pd.DataFrame:
        # keep these values consistent
        gamma_grid= [0.001,0.002,0.004,0.008,0.0160,0.032,0.064,0.128]
        Keps = 2.12
        
        # (1) Compute the scores, means, and variances
        scores, means, variances = self.argarch_conformal_forecasting_compute_scores(score_type=score_type, lookback=300, start_up=start_up, ar=ar, garch_p=garch_p, garch_q=garch_q)

        # (2) Compute the betas
        betas = self.garch_conformal_forecasting_compute_betas(scores,lookback=lookback)

        # (3) Run DtACI
        stable_grid_gammas = self.argarch_conformal_adapt_stable(betas, alpha, gamma_grid, sigma=sigma, eta=eta, eta_adapt=eta_adapt, I=I)
        alpha_seq, err_seq_adapt, err_seq_fixed, gamma_seq, mean_alpha_seq, mean_error_seq, mean_gammas = stable_grid_gammas

        # (4) Get the forecast dataframe
        forecast_df = self.build_conformal_intervals(scores, means, variances, mean_alpha_seq, score_type = score_type, burn_ind=burn_ind)

        # (5) Return summary dataframe
        return forecast_df