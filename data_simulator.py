import pandas as pd
import numpy as np
import wrds
import sqlite3
from arch import arch_model
import numpy as np
from typing import Tuple
from scipy.stats import genextreme
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import genextreme, anderson_ksamp
from scipy.stats import entropy
from scipy.special import kl_div
import matplotlib.pyplot as plt
import concurrent.futures
import os

## PLOTTING FUNCTIONS
def plot_extremes_compare(sim_data, x, y, y2, color_col, differenced = True):
    fig = go.Figure()

    if "_str" not in color_col:
        color_col_str = color_col + "_str"
        sim_data[color_col_str] = sim_data[color_col].astype(str)
    else:
        color_col_str = color_col
    

    
    # first: scatter plot with color
    fig.add_trace(
        go.Scatter(
            x=sim_data.loc[sim_data[color_col_str] == '0', x],
            y=sim_data.loc[sim_data[color_col_str] == '0', y],
            mode='markers',
            marker=dict(color='#3B3B3B'),
            showlegend=False
        )
    )
    
    # green points
    fig.add_trace(
        go.Scatter(
            x=sim_data.loc[sim_data[color_col_str] == '1', x],
            y=sim_data.loc[sim_data[color_col_str] == '1', y],
            mode='markers',
            marker=dict(color='red'),
            name='original extreme values',   # legend label
            showlegend=True,
        )
    )

    # Add line connecting points for `y`
    fig.add_trace(
        go.Scatter(
            x=sim_data[x],
            y=sim_data[y],
            mode='lines',
            line=dict(color='#3B3B3B'),
            name='original line',
            showlegend=True,
        )
    )

    # second scatter plot for `y2` with color
    fig.add_trace(
        go.Scatter(
            x=sim_data.loc[sim_data[color_col_str] == '0', x],
            y=sim_data.loc[sim_data[color_col_str] == '0', y2],
            mode='markers',
            marker=dict(color='#3B3B3B'),
            name='original values',   # legend label
            showlegend=True,
        )
    )
    
    # green points
    fig.add_trace(
        go.Scatter(
            x=sim_data.loc[sim_data[color_col_str] == '1', x],
            y=sim_data.loc[sim_data[color_col_str] == '1', y2],
            mode='markers',
            marker=dict(color='green'),
            name='injected extreme values',   # legend label
            showlegend=True,
        )
    )

    # Add line for `y2`
    fig.add_trace(
        go.Scatter(
            x=sim_data[x],
            y=sim_data[y2],
            mode='lines',
            line=dict(color='#3B3B3B', dash='dot'),
            name='simulated extreme line',
            showlegend=True,
        )
    )

    fig.update_layout(
        title=dict(
            text="Simulated Series with Injected Extreme Values",
            x=0.5,  # center
            xanchor='center',
            font=dict(size=20, family='Arial', color='black')
        )
    )

    if differenced:
        y_title = "diff[log(H/L)]"
    else:
        y_title = "log(H/L)"
        
    fig.update_layout(
        xaxis_title="t",
        yaxis_title=y_title,
        legend_title="Key",
        template="ggplot2"
    )

    fig.show()

# function to plot multiple GEV distributions
def plot_genextreme_distributions(results: list, colors=None, labels=None):
    plt.figure(figsize=(10, 6))

    i = 0
    for res in results:
        c = res['parameters']['c']
        loc = res['parameters'].get('loc', 0)
        scale = res['parameters'].get('scale', 1)

        # Generate x-values safely within distribution support
        x = np.linspace(
            genextreme.ppf(0.01, c, loc=loc, scale=scale),
            genextreme.ppf(0.9, c, loc=loc, scale=scale),
            300
        )
        y = genextreme.pdf(x, c, loc=loc, scale=scale)

        color = colors[i] if colors and i < len(colors) else None
        label = labels[i] if labels and i < len(labels) else f'{i}: c={c:.2f}'

        plt.plot(x, y, lw=2, color=color, label=label)
        i += 1

    plt.title('Generalized Extreme Value (GEV) Distributions')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_genextreme_relative_entropy(params1, params2):
    # get parameters
    c1, loc1, scale1 = params1[0], params1[1], params1[2]
    c2, loc2, scale2 = params2[0], params2[1], params2[2]

    # calculate the relative entropy between the two distributions
    x = np.linspace(
        min(genextreme.ppf(0.01, c1, loc=loc1, scale=scale1), genextreme.ppf(0.01, c2, loc=loc2, scale=scale2)),
        max(genextreme.ppf(0.99, c1, loc=loc1, scale=scale1), genextreme.ppf(0.99, c2, loc=loc2, scale=scale2)),
        1000
    )
    pdf1 = genextreme.pdf(x, c1, loc=loc1, scale=scale1)
    pdf2 = genextreme.pdf(x, c2, loc=loc2, scale=scale2)
    rel_entropy = entropy(pdf1, pdf2)

    return rel_entropy

# function to generate two distinct genextreme distributions
def generate_genextreme_distributions(c_min: float, c_max: float, loc_min: float, loc_max: float, scale_min: float, scale_max: float, kl_cutoff: float = 1):
    while True:
        # randomly generate parameters
        c1, loc1, scale1 = np.random.uniform(c_min, c_max), np.random.uniform(loc_min, loc_max), np.random.uniform(scale_min, scale_max)
        c2, loc2, scale2 = np.random.uniform(c_min, c_max), np.random.uniform(loc_min, loc_max), np.random.uniform(scale_min, scale_max)

        # compute the relative entropy
        rel_entropy = compute_genextreme_relative_entropy(params1=(c1, loc1, scale1), params2=(c2, loc2, scale2))
        
        # Check they are distinct
        if ((c1 != c2) or (loc1 != loc2) or (scale1 != scale2)) and rel_entropy > kl_cutoff:
            break
    
    dist1 = {'model': 'genextreme', 'parameters': {'c': c1, 'loc': loc1, 'scale': scale1}}
    dist2 = {'model': 'genextreme', 'parameters': {'c': c2, 'loc': loc2, 'scale': scale2}}

    # return the two distributions as a list
    return [dist1, dist2], rel_entropy

class DataSimulator:
    def __init__(self, reference_data: pd.DataFrame, init_value: float, fit_col='log_highlow_diff', untransformed_col='hl_ratio'):
        self.reference_data = reference_data
        self.fit_col = fit_col
        self.untransformed_col = untransformed_col
        self.init_value = init_value
        self.am = (reference_data, fit_col) # sets self._am via the setter method
        self.res = self.am                  # sets self._res via the setter method

    @property
    def am(self):
        return self._am
    
    @am.setter
    def am(self, value: Tuple[pd.DataFrame, str]):
        reference_data, fit_col = value
        self._am = arch_model(reference_data[fit_col].dropna(), vol='Garch', p=1, q=1, dist='normal', mean='AR', lags=1)

    @property
    def res(self):
        return self._res
    
    @res.setter
    def res(self, value: arch_model):
        am = value
        self._res = am.fit(disp="off") # results of the fitted model
    
    # function to transform the simulated log high-low difference series back to the original scale
    def transform_hl_diff_series(self, hl_diff_srs) -> pd.Series:
        # get the simulated series
        sim_srs = np.exp(hl_diff_srs/100.00)
        # this gives a series of observations like: (H_t / L_t)/(H_{t-1} / L_{t-1})
        # transformed from [log(H_t / L_t) - log(H_{t-1} / L_{t-1})] * 100

        # get the result series
        result_series = [self.init_value] # start with this as initial value
        prev_val = self.init_value       # keep track of previous value

        # loop through each value in the simulated series
        for val in sim_srs: 
            # recall that... val = (H_t / L_t)/(H_{t-1} / L_{t-1})
            transformed_val = max(1, float(val) * prev_val) # multiply by the previous value
            result_series.append(transformed_val)
        
            # set prev_val to transformed_val
            prev_val = transformed_val 

        return pd.Series(result_series)
    
    # block-maximum approach to flag the maximum value in each block of n values    
    def flag_max_n(self, series: pd.Series, n: int) -> pd.Series:
        flags = pd.Series(0, index=series.index)
        
        # Iterate over the series in chunks
        for start in range(0, len(series), n):
            end = min(start + n, len(series))
            chunk = series.iloc[start:end]
            
            if not chunk.empty:
                max_idx = chunk.idxmax()
                flags.loc[max_idx] = 1

        return flags

    # function to get a valid simulated dataframe
    def get_valid_simulated_dataframe(self, cutoff_val, *, nob_num = 200, burn_num = 100, x_cutoff = 3, extreme_n = 5, attempt_cutoff = 100, verbose=False) -> pd.DataFrame:
        i = 0 # attempt counter
        while True: # run until we get a valid simulated dataframe
            if verbose:
                print("Attempt", i)

            if i > attempt_cutoff:
                raise ValueError("Max attempts exceeded in get_valid_simulated_dataframe without finding a valid simulation.")
        
            # simulate data using the parameters from the AR-GARCH model
            sim_data = self.am.simulate(
                params=self.res.params, 
                nobs=nob_num, # generate nob_num additional sample points
                initial_value=self.init_value, 
                x=None,                       
                burn=burn_num # burn time of 100                    
            )

            # get the transformed series
            hl_diff_srs = sim_data["data"]
            transf_srs = self.transform_hl_diff_series(hl_diff_srs)

            i += 1

            # additionally, check that none of the generated values are LESS than 1, which is impossible given the nature of the series
            # check that only x_cutoff number of samples exceeds our maximum value
            if len(transf_srs[transf_srs <= 1]) == 0 and len(transf_srs[transf_srs > cutoff_val]) <= x_cutoff:
                # add additional columns: max flag, t, and block label
                sim_data['max_flag'] = self.flag_max_n(sim_data['data'], n=extreme_n)
                sim_data['t'] = sim_data.index
                sim_data['block'] = (np.array(range(len(sim_data))) // extreme_n)

                # return the valid simulated dataframe
                return sim_data
    
    # simple accept-reject sampling method to get a valid extreme value
    def simple_accept_reject_sample(self, c, loc, scale, conditional_std, block_max, current_val, next_val, block_match, max_deviation = 3, *, max_tries=300):
        valid_samples = np.array([])
        
        # threshold is determined by the conditional standard deviation
        threshold = max_deviation * conditional_std

        # if, for some reason, the block_max is greater than our threshold, adjust the threshold
        if block_max > threshold:
            threshold = block_max + conditional_std
        
        # to prevent infinite loops, set a maximum number of tries
        count_i, max_i = 0, 100
        
        while len(valid_samples) == 0 and count_i < max_i:
            samples = genextreme.rvs(c, loc=loc, scale=scale, size=max_tries)
            valid_samples = samples[samples <= threshold] # samples need to be less than threshold
            valid_samples = valid_samples[valid_samples >= block_max] # samples need to be greater than the block maximum
            
            if block_match: # if next_val is not nan, perform an additional comparison
                # make sure it's bigger than the next (adjusted) jump
                shifted_values = next_val - (valid_samples - current_val)
                valid_samples = valid_samples[valid_samples > shifted_values]

            count_i += 1

        # if we exceed max tries, raise an error
        if count_i == max_i:
            raise ValueError("Max tries exceeded in simple_accept_reject_sample without finding a valid sample.")
        
        return valid_samples[0]
    
    # function to generate a new series with distributional shift in the extreme values
    def generate_distribution_shift_series(self, reference_ev_orig: dict | None= None, reference_ev_new: dict = None, *, sim_data: pd.DataFrame | None = None, as_dataframe: bool = False, verbose=False) -> pd.Series | pd.DataFrame:
        if sim_data is None:
            sim_data = self.get_valid_simulated_dataframe()

        # convert to numpy array for in-place updates
        sim_np = sim_data[["t", "data", "volatility", "max_flag", "block"]].to_numpy()
    
        # IN-PLACE VERSION
        row_i = 0
        max_orig = len(sim_data) / 2
        for row in sim_np:
            if verbose:
                print("Iteration:", row_i, "/", sim_np.shape[0])

            # get row's values
            time            = row[0]
            current_val     = row[1]
            conditional_std = row[2]
            max_flag        = row[3]
            block           = row[4]

            # LOGIC TO DETERMINE NEXT VALUE AND NEXT BLOCK
            if row_i + 1 < sim_np.shape[0]:
                next_val   = sim_np[row_i + 1, 1]
                next_block = sim_np[row_i + 1, 4]
            else:
                next_val   = np.nan
                next_block = None

            # check whether this block matches the next val's block, need to adjust in that case
            block_match: bool = block == next_block 

            # get the block maximum EXCLUDING the currently-labeled maximum
            block_max = sim_np[(sim_np[:, 4] == block) & (sim_np[:, 1] != current_val), 1].max()

            if max_flag == 1:
                if row_i <= max_orig:
                    gen_sample = self.simple_accept_reject_sample(reference_ev_orig['parameters']['c'], 
                                                            reference_ev_orig['parameters']['loc'], 
                                                            reference_ev_orig['parameters']['scale'], 
                                                            conditional_std, block_max, current_val, next_val, block_match, max_deviation = 4) # UPDATE
                    
                else:
                    gen_sample = self.simple_accept_reject_sample(reference_ev_new['parameters']['c'], 
                                                                reference_ev_new['parameters']['loc'], 
                                                                reference_ev_new['parameters']['scale'], 
                                                                conditional_std, block_max, current_val, next_val, block_match, max_deviation = 4)
                    
                # update the current value in-place
                sim_np[row_i, 1] = gen_sample # in-place update
                
                if row_i + 1 < sim_np.shape[0]: # update the next value to adjust
                    sim_np[row_i + 1, 1] = next_val - (gen_sample - current_val)

            # increase row
            row_i += 1
        if as_dataframe:
            sim_data['data_ext'] = sim_np[:,1]
            return sim_data
        else:
            return pd.Series(sim_np[:,1])
    
    # function to test whether the injected values are follow the desired distributions and are different from each other
    def test_injected_values(self, sim_data: pd.DataFrame, reference_ev_orig, reference_ev_new, *, split_val = 100, dist_samp_size = 1000, alpha = 0.025):
        # get first and second half series
        first_half  = sim_data[:split_val][sim_data["max_flag"] == 1]['data_ext']
        second_half = sim_data[split_val:][sim_data["max_flag"] == 1]['data_ext']

        # get shapes, locs, and scales
        shape_1, loc_1, scale_1 = reference_ev_orig['parameters']['c'], reference_ev_orig['parameters']['loc'], reference_ev_orig['parameters']['scale']
        shape_2, loc_2, scale_2 = reference_ev_new['parameters']['c'], reference_ev_new['parameters']['loc'], reference_ev_new['parameters']['scale']

        # get generated data
        data_1 = genextreme.rvs(shape_1, loc_1, scale_1, size=dist_samp_size)
        data_2 = genextreme.rvs(shape_2, loc_2, scale_2, size=dist_samp_size)

        # get p-values
        pvals = np.array([anderson_ksamp([data_1, first_half]).pvalue, 
                        anderson_ksamp([data_2, first_half]).pvalue,
                        anderson_ksamp([data_1, second_half]).pvalue,
                        anderson_ksamp([data_2, second_half]).pvalue]).reshape(2,2)

        # condition checks
        cond = (
            (pvals[0, 0] > alpha) and
            (pvals[1, 1] > alpha) and
            (pvals[0, 1] <= alpha) and
            (pvals[1, 0] <= alpha)
        )

        # check that the two are different
        two_pval = anderson_ksamp([first_half, second_half]).pvalue
        two_cond = two_pval > alpha

        # return results
        return bool(cond), bool(two_cond), pvals, two_pval
    
    # function to attempt multiple simulations in parallel until one is successful
    def attempt_simulation(self, ev_parameters, alpha, sim_data = None):
        reference_evs, _ = generate_genextreme_distributions(ev_parameters['c_min'], ev_parameters['c_max'], 
                                                             ev_parameters['loc_min'], ev_parameters['loc_max'], 
                                                             ev_parameters['scale_min'], ev_parameters['scale_max'])
        reference_ev_orig, reference_ev_new = reference_evs[0], reference_evs[1]

        # try and attempt the simulation
        try:
            sim_data_local = self.generate_distribution_shift_series(reference_ev_orig, reference_ev_new, sim_data = sim_data, as_dataframe=True, verbose=False)
        except ValueError as e:
            return None, reference_evs
        
        # test that the injected values follow the desired distributions
        cond, two_cond, _, _ = self.test_injected_values(sim_data_local, reference_ev_orig, reference_ev_new, alpha=alpha)

        # if both conditions are met, return the simulated data and the reference extreme value distributions
        if cond and two_cond:
            return sim_data_local, reference_evs
        
        # otherwise, return no simulated data
        return None, reference_evs

    def get_final_distribution_shifted_dataframe(self, ev_parameters: dict, *, sim_data: pd.DataFrame | None = None, alpha: float = 0.025, verbose: bool = False, max_attempts=10000) -> pd.DataFrame:
        if sim_data is None:
            if verbose:
                print("Generating initial valid simulated dataframe...")

            sim_data = self.get_valid_simulated_dataframe(cutoff_val = self.reference_data[self.untransformed_col].max() + 0.1)

            if verbose:
                print("Initial valid simulated dataframe generated.")

        logical_cores = os.cpu_count()
        i = 0  # attempt counter
        while True:
            if verbose:
                print("Attempt", i)

            if i > max_attempts:
                raise ValueError("Max attempts exceeded in get_final_distribution_shifted_dataframe without finding a valid simulation.")

            found = None
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.attempt_simulation, ev_parameters, alpha, sim_data)
                    for _ in range(logical_cores)  # logical_cores parallel attempts
                ]

                for future in concurrent.futures.as_completed(futures):
                    result, reference_evs = future.result()
                    if result is not None:
                        found = result
                        break

            # Cancel any remaining futures if we already found a result
            if found is not None:
                for f in futures:
                    f.cancel()
                sim_data = found
                break

            i += logical_cores  # increment attempt counter

        if verbose:
            print("Successful after", i, "testing attempts!")

        return sim_data, reference_evs
