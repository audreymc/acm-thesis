import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
from arch import arch_model
import pandas as pd
from arch.__future__ import reindexing
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import logging
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)

class ModelFitting:
    def __init__(self, data):
        self.data = data
    
    # fit and forecast AR(1)-GARCH(1,1) model using rolling window approach
    def fit_forecast_ar_garch(self, *, window_size=300, alpha = 0.05, pred_col = "log_highlow_diff", ar_val = 1, garch_p = 1, garch_q = 1):
        # prepare to store results
        forecast_results = []
        actuals = []

        # rolling window loop
        for start in range(len(self.data[pred_col]) - window_size):
            end = start + window_size
            train_data = self.data.iloc[start:end][pred_col]
            
            # fit AR(1)-GARCH(1,1) model
            if garch_p == 0:
                am = arch_model(train_data, mean="ARX", lags=ar_val, vol="ARCH", p=garch_q, dist="t")
            else:
                am = arch_model(train_data, mean='ARX', lags=ar_val, vol='GARCH', p=garch_p, q=garch_q, dist='t')
            res = am.fit(disp='off')

            # forecast 1 step ahead
            forecast = res.forecast(horizon=1)
            forecast_mean = forecast.mean.iloc[-1, 0]
            forecast_variance = forecast.variance.iloc[-1, 0]
            forecast_std = np.sqrt(forecast_variance)
            
            # get degrees of freedom
            df = res.params['nu']

            # calculate 95% confidence interval for the forecast
            t_critical = t.ppf(1 - alpha/2, df)
            lower_bound = forecast_mean - t_critical * forecast_std
            upper_bound = forecast_mean + t_critical * forecast_std
            
            # get the actual return (next value)
            actual = self.data.iloc[end][pred_col]
            
            # store results
            forecast_results.append((forecast_mean, lower_bound, upper_bound))
            actuals.append(actual)
        
        # convert results to DataFrame
        forecast_df = pd.DataFrame(forecast_results, columns=['forecast', 'lower_bound', 'upper_bound'])
        forecast_df['actual'] = actuals

        # check if actuals lie within the confidence intervals
        forecast_df['within_CI'] = (forecast_df['actual'] >= forecast_df['lower_bound']) & (forecast_df['actual'] <= forecast_df['upper_bound'])

        # return the forecast DataFrame
        return forecast_df
    
    # fit and forecast using Prophet model with rolling window approach
    def fit_forecast_prophet(self, window_size=300, alpha=0.05, date_col = "dlycaldt", pred_col = "log_highlow_diff", horizon=1):
        # prepare the data for Prophet
        df = self.data[[date_col, pred_col]].rename(columns={date_col:"ds", pred_col:"y"})

        # prepare to store results
        forecast_results = []
        actuals = []

        for start in range(len(df) - window_size):
            end = start + window_size
            train_data = df.iloc[start:end]
            
            # fit the Prophet model
            model = Prophet(interval_width=1-alpha)
            model.fit(train_data)
            
            # make future dataframe
            future = model.make_future_dataframe(periods=horizon)
            
            # forecast
            forecast = model.predict(future)

            # get one step ahead prediction
            one_step_ahead = forecast.iloc[-1].to_dict()  # last row

            forecast_mean = one_step_ahead['yhat']
            lower_bound = one_step_ahead['yhat_lower']
            upper_bound = one_step_ahead['yhat_upper']
            actual = df.iloc[end]["y"]  # actual value

            # store results
            forecast_results.append((forecast_mean, lower_bound, upper_bound))
            actuals.append(actual)
            
        # convert results to DataFrame
        forecast_df = pd.DataFrame(forecast_results, columns=['forecast', 'lower_bound', 'upper_bound'])
        forecast_df['actual'] = actuals

        # check if actuals lie within the confidence intervals
        forecast_df['within_CI'] = (forecast_df['actual'] >= forecast_df['lower_bound']) & (forecast_df['actual'] <= forecast_df['upper_bound'])

        # return the forecast DataFrame
        return forecast_df
    
    ###########################################################################
    # ARMA Conformal Forecasting
    # This method is using a simple ARMA base model for forecasting the mean return but does not model the volatility.
    # It produces online conformal prediction intervals based on the ARMA model's predictions and a gamma score.
    # The method allows for dynamic adjustment of the alpha level based on past errors.
    # Parameters:
    # - alpha: initial significance level for the prediction intervals
    # - gamma: learning rate for updating the alpha level
    # - pred_col: column name in the data containing the returns to be predicted
    # - lookback: number of past observations to consider for the ARMA model
    # - ar: order of the autoregressive part of the ARMA model
    # - ma: order of the moving average part of the ARMA model
    # - startUp: number of initial observations to "burn in" the model before starting predictions
    # - verbose: whether to print progress information
    # - updateMethod: method for updating the alpha level ("Simple" or "Momentum")
    # - momentumBW: bandwidth for the momentum-based update method
    # Returns:
    # - online_df: DataFrame containing the online conformal prediction intervals and results
    # - naive_df: DataFrame containing the naive conformal prediction intervals and results
    # NOTE: Need extra observations to "burn in" the model
    ###########################################################################
    def arma_conformal_forecasting(self, alpha, gamma, pred_col = "log_highlow_diff", lookback=300, ar=1, ma=0, startUp=100, verbose=False, updateMethod="Simple", momentumBW=0.95):
        returns = self.data[pred_col].values
        T = len(returns) # total number of returns
        startUp = max(startUp, lookback) # ensure startUp is at least as large as lookback
    
        alphat = alpha  # initialize dynamic alpha
        
        ### initialize data storage lists
        errSeqOC = [0] * (T - startUp)  # online conformal error sequence
        errSeqNC = [0] * (T - startUp)  # naive conformal error sequence
        alphaSequence = [alpha] * (T - startUp)  # track evolving alpha values
        scores = [0] * (T - startUp)  # track conformity scores
        pred_returns = []
        online_intervals = []
        naive_intervals = []

        for t in range(startUp, T):
            if (verbose):
                print(t)
            
            ### 1. Fit basic ARMA model
            arma_model = ARIMA(returns[(t - lookback):t], order=(ar, 0, ma)).fit()
            
            ### 2. Forecast next-period high-low return
            retNext = arma_model.forecast(steps=1)[0]
            pred_returns.append(retNext)
            
            ### 3. Compute conformity score: Adapted Gamma
            scores[t - startUp] = abs((returns[t] - retNext)/retNext) # take the absolute value of the normalized residual
            
            ### 4. Collect recent past scores for quantile estimation
            recentScores = scores[max(t - startUp - lookback, 0):(t - startUp + 1)]
            
            ### 5. Update error sequences
            index = t - startUp
            
            # Online Conformal: using current dynamic alpha
            alphat_quantile = np.quantile(recentScores, 1 - max(alphat, 0))
            errSeqOC[index] = int(scores[index] > alphat_quantile) # return the alphat quantile
            online_intervals.append([retNext - np.abs(retNext) * alphat_quantile, retNext + np.abs(retNext) * alphat_quantile])
            
            # Naive Conformal: using fixed alpha
            alpha_quantile = np.quantile(recentScores, 1 - alpha)
            errSeqNC[index] = int(scores[index] > alpha_quantile)
            naive_intervals.append([retNext - np.abs(retNext) * alpha_quantile, retNext + np.abs(retNext) * alpha_quantile])
                    
            ### 6. Update alphat based on past errors
            alphaSequence[t - startUp] = alphat
            
            if updateMethod == "Simple":
                # Simple online update rule
                alphat = alphat + gamma * (alpha - errSeqOC[t - startUp])          
            elif updateMethod == "Momentum":
                # Momentum-based update rule
                w = np.power(momentumBW, np.arange(0, t - startUp + 1))[::-1]
                w = w / sum(w)
                alphat = alphat + gamma * (alpha - sum(errSeqOC[0:(t - startUp)] * w))
            
            ### Optional: Print progress every 100 steps
            if t % 100 == 0:
                print("Done", t, "steps")

        # create result dataframes
        # online intervals
        online_df = pd.DataFrame(online_intervals, columns=['lower_bound', 'upper_bound'])
        online_df['actual'] = returns[startUp:]
        online_df['forecast'] = pred_returns
        online_df['within_CI'] = (online_df['actual'] >= online_df['lower_bound']) & (online_df['actual'] <= online_df['upper_bound'])
        online_df['alpha'] = alphaSequence

        # naive intervals
        naive_df = pd.DataFrame(naive_intervals, columns=['lower_bound', 'upper_bound'])
        naive_df['actual'] = returns[startUp:]
        naive_df['forecast'] = pred_returns
        naive_df['within_CI'] = (naive_df['actual'] >= naive_df['lower_bound']) & (naive_df['actual'] <= naive_df['upper_bound'])
        naive_df['alpha'] = alpha
    
        # Return the results
        return (online_df, naive_df)

    # NOTE: Need extra observations to "burn in" the model
    def argarch_conformal_forecasting(self, alpha, gamma, 
                                         pred_col = "log_highlow_diff", lookback=300, 
                                         ar=1,
                                         garch_p = 1, garch_q = 1, 
                                         startUp=100, verbose=False, updateMethod="Simple", momentumBW=0.95):
        returns = self.data[pred_col].values
        T = len(returns) # total number of returns
        startUp = max(startUp, lookback) # ensure startUp is at least as large as lookback
    
        alphat = alpha  # initialize dynamic alpha
        
        ### initialize data storage lists
        errSeqOC = [0] * (T - startUp)  # online conformal error sequence
        errSeqNC = [0] * (T - startUp)  # naive conformal error sequence
        alphaSequence = [alpha] * (T - startUp)  # track evolving alpha values
        scores = [0] * (T - startUp)  # track conformity scores
        pred_returns = []
        pred_std = []
        online_intervals = []
        naive_intervals = []

        for t in range(startUp, T):
            if (verbose):
                print(t)

            ### 1. Fit AR-GARCH model
            am = arch_model(returns[(t - lookback):t], mean='ARX', lags=ar, vol='GARCH', p=garch_p, q=garch_q, dist='t')
            res = am.fit(disp='off')

            ### 2. Forecast next-pe7riod high-low return
            forecast = res.forecast(horizon=1)
            forecast_mean = forecast.mean.iloc[-1, 0]
            forecast_variance = forecast.variance.iloc[-1, 0]
            forecast_std = np.sqrt(forecast_variance)
            pred_returns.append(forecast_mean)
            pred_std.append(forecast_std)
            
            ### 3. Compute conformity score: Residual Normalized Score
            scores[t - startUp] = abs((returns[t] - forecast_mean)/forecast_std)
            
            ### 4. Collect recent past scores for quantile estimation
            recentScores = scores[max(t - startUp - lookback, 0):(t - startUp + 1)]
            
            ### 5. Update error sequences
            index = t - startUp
            
            # Online Conformal: using current dynamic alpha
            alphat_quantile = np.quantile(recentScores, 1 - max(alphat, 0))
            errSeqOC[index] = int(scores[index] > alphat_quantile) # return the alphat quantile
            online_intervals.append([forecast_mean - forecast_std * alphat_quantile, forecast_mean + forecast_std * alphat_quantile])
            
            # Naive Conformal: using fixed alpha
            alpha_quantile = np.quantile(recentScores, 1 - alpha)
            errSeqNC[index] = int(scores[index] > alpha_quantile)

            # NOTE: This is a residual normalized interval
            naive_intervals.append([forecast_mean - forecast_std * alpha_quantile, forecast_mean + forecast_std * alpha_quantile])
                    
            ### 6. Update alphat based on past errors
            alphaSequence[t - startUp] = alphat
            
            if updateMethod == "Simple":
                # Simple online update rule
                alphat = alphat + gamma * (alpha - errSeqOC[t - startUp])          
            elif updateMethod == "Momentum":
                # Momentum-based update rule
                w = np.power(momentumBW, np.arange(0, t - startUp + 1))[::-1]
                w = w / sum(w)
                alphat = alphat + gamma * (alpha - sum(errSeqOC[0:(t - startUp)] * w))
            
            ### Optional: Print progress every 100 steps
            if t % 100 == 0:
                print("Done", t, "steps")

        # create result dataframes
        # online intervals
        online_df = pd.DataFrame(online_intervals, columns=['lower_bound', 'upper_bound'])
        online_df['actual'] = returns[startUp:]
        online_df['forecast'] = pred_returns
        online_df['forecast_std'] = pred_std
        online_df['within_CI'] = (online_df['actual'] >= online_df['lower_bound']) & (online_df['actual'] <= online_df['upper_bound'])
        online_df['alpha'] = alphaSequence

        # naive intervals
        naive_df = pd.DataFrame(naive_intervals, columns=['lower_bound', 'upper_bound'])
        naive_df['actual'] = returns[startUp:]
        naive_df['forecast'] = pred_returns
        naive_df['forecast_std'] = pred_std
        naive_df['within_CI'] = (naive_df['actual'] >= naive_df['lower_bound']) & (naive_df['actual'] <= naive_df['upper_bound'])
        naive_df['alpha'] = alpha
    
        # Return the results
        return (online_df, naive_df)      

    
    # function to get coverage statistics
    def get_coverage_stats(self, forecast_df, level = 0.95):
        # ensure we don't modify the original DataFrame
        forecast_df = forecast_df.copy()

        # calculate coverage
        coverage = forecast_df['within_CI'].mean() * 100

        # calculate average, median, max, and min width of the confidence interval
        forecast_df['width'] = forecast_df['upper_bound'] - forecast_df['lower_bound']
        avg_width = forecast_df['width'].mean()
        median_width = forecast_df['width'].median()
        max_width = forecast_df['width'].max()
        min_width = forecast_df['width'].min()

        # calculate the MWI score
        # calculate MWI score: sum of max(0, |actual - closest bound|)
        
        # (1) MWI Score
        out_of_bounds = ~forecast_df['within_CI']
        mwi_score = avg_width + (1/len(forecast_df)) * (2/(1-level)) * np.sum(
            np.maximum(
            0,
            np.abs(forecast_df.loc[out_of_bounds, 'actual'] -
                np.where(
                np.abs(forecast_df.loc[out_of_bounds, 'actual'] - forecast_df.loc[out_of_bounds, 'lower_bound']) 
                < np.abs(forecast_df.loc[out_of_bounds, 'actual'] - forecast_df.loc[out_of_bounds, 'upper_bound']),
                forecast_df.loc[out_of_bounds, 'lower_bound'],
                forecast_df.loc[out_of_bounds, 'upper_bound']
                )
            )
            )
        )

        # (2) CWC score
        nmpiw_score = avg_width / (forecast_df['actual'].max() - forecast_df['actual'].min())
        mu = level
        picp_score = (1/len(forecast_df)) * np.sum(forecast_df['within_CI'].astype(int))
        gamma = 0 if picp_score >= mu else 1
        eta = 1 # set eta to 1 for simplicity
        cwc_score = nmpiw_score * (1 + (gamma * picp_score * np.exp(-eta * (picp_score - mu))))

        # calculate mean absolute error
        mae = np.abs(forecast_df['actual'] - forecast_df['forecast']).mean()

        # calculate RMSE
        rmse = np.sqrt(np.mean((forecast_df['actual'] - forecast_df['forecast'])**2))

        return {'coverage': coverage, 
                'target_coverage': level * 100,
                'avg_width': avg_width,
                'median_width': median_width,
                'max_width': max_width,
                'mwi_score': mwi_score,
                'cwc_score': cwc_score,
                'min_width': min_width,
                'mae': mae,
                'rmse': rmse}
    
    # function to plot the forecast
    def plot_forecast(self, forecast_df, level = 0.95):
        # set the target coverage level
        target_coverage = level * 100

        # plot the actual returns and forecasted mean with confidence intervals
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df['actual'], label='Actual Return', color='black', alpha=0.7)
        plt.plot(forecast_df['forecast'], label='Forecasted Mean', color='orange', linestyle='--')
        plt.fill_between(forecast_df.index, forecast_df['lower_bound'], forecast_df['upper_bound'], color='orange', alpha=0.2, label=f'{target_coverage}% CI')
        plt.title(f'One-Step-Ahead Forecast with {target_coverage}% Confidence Interval')
        plt.legend()
        plt.show()

    
    # function to reset data
    def reset_data(self, new_data):
        self.data = new_data
