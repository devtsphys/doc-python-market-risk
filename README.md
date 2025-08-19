# Market Risk in Python - Complete Reference Card

## Table of Contents

1. [Essential Libraries](#essential-libraries)
1. [Basic Risk Metrics](#basic-risk-metrics)
1. [Value at Risk (VaR) Methods](#value-at-risk-var-methods)
1. [Expected Shortfall (ES/CVaR)](#expected-shortfall-escvar)
1. [Portfolio Risk Measures](#portfolio-risk-measures)
1. [Monte Carlo Simulation](#monte-carlo-simulation)
1. [GARCH Models](#garch-models)
1. [Backtesting Methods](#backtesting-methods)
1. [Stress Testing](#stress-testing)
1. [Risk Attribution](#risk-attribution)
1. [Complete Methods Reference Table](#complete-methods-reference-table)

## Essential Libraries

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import yfinance as yf
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
```

## Basic Risk Metrics

### Volatility Calculations

```python
# Historical Volatility (Annualized)
def historical_volatility(returns, periods=252):
    """Calculate annualized historical volatility"""
    return returns.std() * np.sqrt(periods)

# Exponentially Weighted Moving Average (EWMA) Volatility
def ewma_volatility(returns, lambda_factor=0.94):
    """Calculate EWMA volatility"""
    weights = np.array([(lambda_factor**i) for i in range(len(returns))])
    weights = weights / weights.sum()
    weighted_returns = returns * weights[::-1]
    return np.sqrt(np.sum(weighted_returns**2))

# Rolling Volatility
def rolling_volatility(returns, window=30, periods=252):
    """Calculate rolling volatility"""
    return returns.rolling(window).std() * np.sqrt(periods)
```

### Basic Risk Measures

```python
# Maximum Drawdown
def max_drawdown(returns):
    """Calculate maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

# Sharpe Ratio
def sharpe_ratio(returns, risk_free_rate=0.02, periods=252):
    """Calculate annualized Sharpe ratio"""
    excess_returns = returns.mean() - risk_free_rate/periods
    return (excess_returns * periods) / (returns.std() * np.sqrt(periods))

# Sortino Ratio
def sortino_ratio(returns, risk_free_rate=0.02, periods=252):
    """Calculate Sortino ratio"""
    excess_returns = returns.mean() - risk_free_rate/periods
    downside_std = returns[returns < 0].std()
    return (excess_returns * periods) / (downside_std * np.sqrt(periods))
```

## Value at Risk (VaR) Methods

### Historical VaR

```python
def historical_var(returns, confidence_level=0.95):
    """Calculate Historical VaR"""
    return np.percentile(returns, (1 - confidence_level) * 100)

def historical_var_dataframe(returns, confidence_levels=[0.95, 0.99]):
    """Calculate Historical VaR for multiple confidence levels"""
    var_results = {}
    for cl in confidence_levels:
        var_results[f'VaR_{int(cl*100)}%'] = np.percentile(returns, (1-cl)*100)
    return pd.Series(var_results)
```

### Parametric VaR (Normal Distribution)

```python
def parametric_var(returns, confidence_level=0.95):
    """Calculate Parametric VaR assuming normal distribution"""
    mean = returns.mean()
    std = returns.std()
    z_score = stats.norm.ppf(1 - confidence_level)
    return mean + z_score * std

# Student's t-distribution VaR
def t_var(returns, confidence_level=0.95):
    """Calculate VaR using Student's t-distribution"""
    params = stats.t.fit(returns)
    df, loc, scale = params
    t_score = stats.t.ppf(1 - confidence_level, df, loc, scale)
    return t_score
```

### Monte Carlo VaR

```python
def monte_carlo_var(returns, confidence_level=0.95, num_simulations=10000):
    """Calculate Monte Carlo VaR"""
    mean = returns.mean()
    std = returns.std()
    
    # Generate random scenarios
    simulated_returns = np.random.normal(mean, std, num_simulations)
    
    # Calculate VaR
    var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
    return var

# Geometric Brownian Motion VaR
def gbm_var(S0, mu, sigma, T, confidence_level=0.95, num_simulations=10000):
    """Monte Carlo VaR using Geometric Brownian Motion"""
    dt = T / 252  # Daily time step
    Z = np.random.standard_normal(num_simulations)
    
    ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    returns = (ST - S0) / S0
    
    return np.percentile(returns, (1 - confidence_level) * 100)
```

## Expected Shortfall (ES/CVaR)

```python
def expected_shortfall(returns, confidence_level=0.95):
    """Calculate Expected Shortfall (Conditional VaR)"""
    var = historical_var(returns, confidence_level)
    return returns[returns <= var].mean()

def parametric_es(returns, confidence_level=0.95):
    """Calculate Parametric Expected Shortfall"""
    mean = returns.mean()
    std = returns.std()
    z_score = stats.norm.ppf(1 - confidence_level)
    
    # Expected Shortfall formula for normal distribution
    es = mean - std * stats.norm.pdf(z_score) / (1 - confidence_level)
    return es

# Combined VaR and ES calculation
def var_es_summary(returns, confidence_levels=[0.95, 0.99]):
    """Calculate both VaR and ES for multiple confidence levels"""
    results = []
    for cl in confidence_levels:
        var_hist = historical_var(returns, cl)
        var_param = parametric_var(returns, cl)
        es_hist = expected_shortfall(returns, cl)
        es_param = parametric_es(returns, cl)
        
        results.append({
            'Confidence_Level': f'{int(cl*100)}%',
            'Historical_VaR': var_hist,
            'Parametric_VaR': var_param,
            'Historical_ES': es_hist,
            'Parametric_ES': es_param
        })
    
    return pd.DataFrame(results)
```

## Portfolio Risk Measures

### Portfolio VaR and Risk Decomposition

```python
def portfolio_var(returns_matrix, weights, confidence_level=0.95):
    """Calculate portfolio VaR"""
    portfolio_returns = (returns_matrix * weights).sum(axis=1)
    return historical_var(portfolio_returns, confidence_level)

def portfolio_risk_decomposition(returns_matrix, weights, confidence_level=0.95):
    """Decompose portfolio risk by component"""
    portfolio_returns = (returns_matrix * weights).sum(axis=1)
    portfolio_var = historical_var(portfolio_returns, confidence_level)
    
    # Component VaR
    component_vars = []
    for i in range(len(weights)):
        if weights[i] != 0:
            # Calculate marginal VaR
            temp_weights = weights.copy()
            temp_weights[i] = 0
            reduced_portfolio = (returns_matrix * temp_weights).sum(axis=1)
            reduced_var = historical_var(reduced_portfolio, confidence_level)
            marginal_var = portfolio_var - reduced_var
            component_vars.append(marginal_var)
        else:
            component_vars.append(0)
    
    return pd.Series(component_vars, index=returns_matrix.columns)

# Covariance Matrix Approach
def portfolio_parametric_var(returns_matrix, weights, confidence_level=0.95):
    """Calculate portfolio VaR using covariance matrix"""
    cov_matrix = returns_matrix.cov()
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_std = np.sqrt(portfolio_variance)
    portfolio_mean = (returns_matrix.mean() * weights).sum()
    
    z_score = stats.norm.ppf(1 - confidence_level)
    return portfolio_mean + z_score * portfolio_std
```

## Monte Carlo Simulation

### Advanced Monte Carlo Methods

```python
class MonteCarloRisk:
    def __init__(self, returns_matrix, num_simulations=10000):
        self.returns = returns_matrix
        self.num_sims = num_simulations
        self.mean_returns = returns_matrix.mean()
        self.cov_matrix = returns_matrix.cov()
        
    def simulate_portfolio_returns(self, weights, time_horizon=1):
        """Simulate portfolio returns using multivariate normal"""
        simulated_returns = np.random.multivariate_normal(
            self.mean_returns * time_horizon,
            self.cov_matrix * time_horizon,
            self.num_sims
        )
        
        portfolio_returns = np.dot(simulated_returns, weights)
        return portfolio_returns
    
    def portfolio_risk_metrics(self, weights, confidence_levels=[0.95, 0.99]):
        """Calculate comprehensive risk metrics"""
        portfolio_sims = self.simulate_portfolio_returns(weights)
        
        results = {}
        for cl in confidence_levels:
            results[f'VaR_{int(cl*100)}%'] = np.percentile(portfolio_sims, (1-cl)*100)
            results[f'ES_{int(cl*100)}%'] = portfolio_sims[
                portfolio_sims <= results[f'VaR_{int(cl*100)}%']
            ].mean()
        
        results['Expected_Return'] = portfolio_sims.mean()
        results['Volatility'] = portfolio_sims.std()
        results['Skewness'] = stats.skew(portfolio_sims)
        results['Kurtosis'] = stats.kurtosis(portfolio_sims)
        
        return results

# Scenario Analysis
def scenario_analysis(returns_matrix, weights, scenarios):
    """Analyze portfolio performance under specific scenarios"""
    results = []
    for scenario_name, scenario_returns in scenarios.items():
        portfolio_return = np.dot(scenario_returns, weights)
        results.append({
            'Scenario': scenario_name,
            'Portfolio_Return': portfolio_return,
            'Individual_Returns': dict(zip(returns_matrix.columns, scenario_returns))
        })
    return pd.DataFrame(results)
```

## GARCH Models

```python
# GARCH(1,1) Volatility Forecasting
def fit_garch_model(returns, model_type='GARCH', p=1, q=1):
    """Fit GARCH model to returns"""
    returns_pct = returns * 100  # Convert to percentage
    
    model = arch_model(returns_pct, vol=model_type, p=p, q=q)
    fitted_model = model.fit(disp='off')
    
    return fitted_model

def garch_var_forecast(returns, confidence_level=0.95, horizon=1):
    """Forecast VaR using GARCH model"""
    garch_model = fit_garch_model(returns)
    
    # Forecast volatility
    forecast = garch_model.forecast(horizon=horizon)
    forecasted_vol = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
    
    # Calculate VaR
    z_score = stats.norm.ppf(1 - confidence_level)
    var_forecast = z_score * forecasted_vol
    
    return var_forecast, forecasted_vol

# EGARCH Model for asymmetric volatility
def egarch_model(returns):
    """Fit EGARCH model"""
    returns_pct = returns * 100
    model = arch_model(returns_pct, vol='EGARCH', p=1, o=1, q=1)
    return model.fit(disp='off')
```

## Backtesting Methods

```python
class VaRBacktesting:
    def __init__(self, returns, var_estimates, confidence_level=0.95):
        self.returns = returns
        self.var_estimates = var_estimates
        self.confidence_level = confidence_level
        self.expected_violations = len(returns) * (1 - confidence_level)
        
    def violation_ratio(self):
        """Calculate violation ratio"""
        violations = (self.returns < self.var_estimates).sum()
        return violations / len(self.returns)
    
    def kupiec_test(self):
        """Kupiec unconditional coverage test"""
        violations = (self.returns < self.var_estimates).sum()
        n = len(self.returns)
        p = 1 - self.confidence_level
        
        if violations == 0:
            lr_stat = 2 * n * np.log(1 - p)
        elif violations == n:
            lr_stat = 2 * n * np.log(p)
        else:
            lr_stat = 2 * (violations * np.log(violations / (n * p)) + 
                          (n - violations) * np.log((n - violations) / (n * (1 - p))))
        
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        return lr_stat, p_value
    
    def christoffersen_test(self):
        """Christoffersen conditional coverage test"""
        violations = (self.returns < self.var_estimates).astype(int)
        
        # Calculate transition probabilities
        n00 = n01 = n10 = n11 = 0
        
        for i in range(1, len(violations)):
            if violations[i-1] == 0 and violations[i] == 0:
                n00 += 1
            elif violations[i-1] == 0 and violations[i] == 1:
                n01 += 1
            elif violations[i-1] == 1 and violations[i] == 0:
                n10 += 1
            elif violations[i-1] == 1 and violations[i] == 1:
                n11 += 1
        
        # Independence test
        if (n00 + n01) > 0 and (n10 + n11) > 0 and n01 > 0 and n11 > 0:
            pi01 = n01 / (n00 + n01)
            pi11 = n11 / (n10 + n11)
            pi = (n01 + n11) / (n00 + n01 + n10 + n11)
            
            lr_ind = 2 * ((n00 + n01) * np.log(1 - pi01) + n01 * np.log(pi01) +
                         (n10 + n11) * np.log(1 - pi11) + n11 * np.log(pi11) -
                         (n00 + n10) * np.log(1 - pi) - (n01 + n11) * np.log(pi))
            
            p_value = 1 - stats.chi2.cdf(lr_ind, df=1)
            return lr_ind, p_value
        else:
            return np.nan, np.nan
    
    def backtesting_summary(self):
        """Generate comprehensive backtesting summary"""
        violation_rate = self.violation_ratio()
        expected_rate = 1 - self.confidence_level
        
        kupiec_stat, kupiec_p = self.kupiec_test()
        christoffersen_stat, christoffersen_p = self.christoffersen_test()
        
        return {
            'Actual_Violation_Rate': violation_rate,
            'Expected_Violation_Rate': expected_rate,
            'Number_of_Violations': (self.returns < self.var_estimates).sum(),
            'Total_Observations': len(self.returns),
            'Kupiec_LR_Statistic': kupiec_stat,
            'Kupiec_p_value': kupiec_p,
            'Christoffersen_LR_Statistic': christoffersen_stat,
            'Christoffersen_p_value': christoffersen_p
        }
```

## Stress Testing

```python
def historical_stress_testing(returns_matrix, weights, stress_periods):
    """Historical stress testing using specific periods"""
    results = []
    
    for period_name, period_data in stress_periods.items():
        start_date, end_date = period_data
        stress_returns = returns_matrix.loc[start_date:end_date]
        
        if len(stress_returns) > 0:
            portfolio_stress_returns = (stress_returns * weights).sum(axis=1)
            
            results.append({
                'Stress_Period': period_name,
                'Start_Date': start_date,
                'End_Date': end_date,
                'Portfolio_Return': portfolio_stress_returns.sum(),
                'Worst_Day': portfolio_stress_returns.min(),
                'Volatility': portfolio_stress_returns.std() * np.sqrt(252),
                'Max_Drawdown': max_drawdown(portfolio_stress_returns)
            })
    
    return pd.DataFrame(results)

def monte_carlo_stress_testing(returns_matrix, weights, stress_scenarios):
    """Monte Carlo stress testing with correlation breakdown"""
    base_mean = returns_matrix.mean()
    base_cov = returns_matrix.cov()
    
    results = []
    
    for scenario_name, scenario_params in stress_scenarios.items():
        # Modify parameters based on scenario
        stressed_mean = base_mean * scenario_params.get('mean_multiplier', 1)
        stressed_cov = base_cov * scenario_params.get('vol_multiplier', 1)
        
        # Generate stressed scenarios
        stressed_returns = np.random.multivariate_normal(
            stressed_mean, stressed_cov, 1000
        )
        
        portfolio_returns = np.dot(stressed_returns, weights)
        
        results.append({
            'Scenario': scenario_name,
            'Mean_Return': portfolio_returns.mean(),
            'VaR_95%': np.percentile(portfolio_returns, 5),
            'VaR_99%': np.percentile(portfolio_returns, 1),
            'ES_95%': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean(),
            'Worst_Case': portfolio_returns.min()
        })
    
    return pd.DataFrame(results)
```

## Risk Attribution

```python
def risk_attribution(returns_matrix, weights, confidence_level=0.95):
    """Comprehensive risk attribution analysis"""
    portfolio_returns = (returns_matrix * weights).sum(axis=1)
    portfolio_var = historical_var(portfolio_returns, confidence_level)
    
    # Component VaR
    component_var = []
    marginal_var = []
    
    for i, asset in enumerate(returns_matrix.columns):
        # Marginal VaR
        epsilon = 0.001
        new_weights = weights.copy()
        new_weights[i] += epsilon
        
        new_portfolio_returns = (returns_matrix * new_weights).sum(axis=1)
        new_var = historical_var(new_portfolio_returns, confidence_level)
        
        mvar = (new_var - portfolio_var) / epsilon
        marginal_var.append(mvar)
        
        # Component VaR
        cvar = weights[i] * mvar
        component_var.append(cvar)
    
    # Risk contribution percentage
    total_component_var = sum(component_var)
    risk_contribution_pct = [cv / total_component_var * 100 for cv in component_var]
    
    attribution_df = pd.DataFrame({
        'Asset': returns_matrix.columns,
        'Weight': weights,
        'Marginal_VaR': marginal_var,
        'Component_VaR': component_var,
        'Risk_Contribution_%': risk_contribution_pct
    })
    
    return attribution_df

# Factor-based risk attribution
def factor_risk_attribution(returns_matrix, factor_loadings, factor_returns, weights):
    """Factor-based risk attribution"""
    # Calculate factor exposures
    portfolio_exposures = np.dot(weights, factor_loadings)
    
    # Factor risk contributions
    factor_cov = factor_returns.cov()
    factor_risk_contributions = []
    
    for i, factor in enumerate(factor_returns.columns):
        factor_var = factor_cov.iloc[i, i]
        factor_contribution = (portfolio_exposures[i]**2) * factor_var
        factor_risk_contributions.append(factor_contribution)
    
    return pd.DataFrame({
        'Factor': factor_returns.columns,
        'Exposure': portfolio_exposures,
        'Risk_Contribution': factor_risk_contributions,
        'Risk_Contribution_%': np.array(factor_risk_contributions) / sum(factor_risk_contributions) * 100
    })
```

## Complete Methods Reference Table

|Category              |Method                  |Function                                     |Description                      |Key Parameters                     |
|----------------------|------------------------|---------------------------------------------|---------------------------------|-----------------------------------|
|**Basic Risk**        |Historical Volatility   |`historical_volatility()`                    |Annualized standard deviation    |`periods=252`                      |
|                      |EWMA Volatility         |`ewma_volatility()`                          |Exponentially weighted volatility|`lambda_factor=0.94`               |
|                      |Maximum Drawdown        |`max_drawdown()`                             |Largest peak-to-trough decline   |None                               |
|                      |Sharpe Ratio            |`sharpe_ratio()`                             |Risk-adjusted return metric      |`risk_free_rate=0.02`              |
|                      |Sortino Ratio           |`sortino_ratio()`                            |Downside risk-adjusted return    |`risk_free_rate=0.02`              |
|**VaR Methods**       |Historical VaR          |`historical_var()`                           |Empirical quantile method        |`confidence_level=0.95`            |
|                      |Parametric VaR          |`parametric_var()`                           |Normal distribution assumption   |`confidence_level=0.95`            |
|                      |Monte Carlo VaR         |`monte_carlo_var()`                          |Simulation-based approach        |`num_simulations=10000`            |
|                      |GARCH VaR               |`garch_var_forecast()`                       |Time-varying volatility          |`horizon=1`                        |
|                      |Student’s t VaR         |`t_var()`                                    |Heavy-tailed distribution        |`confidence_level=0.95`            |
|**Expected Shortfall**|Historical ES           |`expected_shortfall()`                       |Tail expectation                 |`confidence_level=0.95`            |
|                      |Parametric ES           |`parametric_es()`                            |Normal distribution ES           |`confidence_level=0.95`            |
|**Portfolio Risk**    |Portfolio VaR           |`portfolio_var()`                            |Multi-asset risk measure         |`weights`, `confidence_level=0.95` |
|                      |Risk Decomposition      |`portfolio_risk_decomposition()`             |Component risk analysis          |`weights`, `confidence_level=0.95` |
|                      |Parametric Portfolio VaR|`portfolio_parametric_var()`                 |Covariance matrix approach       |`weights`, `confidence_level=0.95` |
|**Monte Carlo**       |Portfolio Simulation    |`MonteCarloRisk.simulate_portfolio_returns()`|Multivariate simulation          |`weights`, `time_horizon=1`        |
|                      |Risk Metrics            |`MonteCarloRisk.portfolio_risk_metrics()`    |Comprehensive risk analysis      |`weights`, `confidence_levels`     |
|                      |Scenario Analysis       |`scenario_analysis()`                        |Specific scenario testing        |`scenarios`                        |
|**GARCH Models**      |GARCH Fitting           |`fit_garch_model()`                          |Volatility clustering model      |`model_type='GARCH'`, `p=1`, `q=1` |
|                      |EGARCH Model            |`egarch_model()`                             |Asymmetric volatility model      |None                               |
|**Backtesting**       |Violation Ratio         |`VaRBacktesting.violation_ratio()`           |Actual vs expected violations    |None                               |
|                      |Kupiec Test             |`VaRBacktesting.kupiec_test()`               |Unconditional coverage test      |None                               |
|                      |Christoffersen Test     |`VaRBacktesting.christoffersen_test()`       |Conditional coverage test        |None                               |
|                      |Backtesting Summary     |`VaRBacktesting.backtesting_summary()`       |Comprehensive test results       |None                               |
|**Stress Testing**    |Historical Stress       |`historical_stress_testing()`                |Historical period analysis       |`stress_periods`                   |
|                      |Monte Carlo Stress      |`monte_carlo_stress_testing()`               |Simulated stress scenarios       |`stress_scenarios`                 |
|**Risk Attribution**  |Risk Attribution        |`risk_attribution()`                         |Component risk analysis          |`confidence_level=0.95`            |
|                      |Factor Attribution      |`factor_risk_attribution()`                  |Factor-based risk breakdown      |`factor_loadings`, `factor_returns`|

## Example Usage Workflow

```python
# Load sample data
import yfinance as yf

# Download stock data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
data = yf.download(tickers, start='2020-01-01', end='2023-12-31')['Adj Close']
returns = data.pct_change().dropna()

# Equal weights portfolio
weights = np.array([0.25, 0.25, 0.25, 0.25])

# Calculate basic risk metrics
print("=== Basic Risk Metrics ===")
portfolio_returns = (returns * weights).sum(axis=1)
print(f"Portfolio Volatility: {historical_volatility(portfolio_returns):.2%}")
print(f"Maximum Drawdown: {max_drawdown(portfolio_returns):.2%}")
print(f"Sharpe Ratio: {sharpe_ratio(portfolio_returns):.2f}")

# Calculate VaR and ES
print("\n=== VaR and Expected Shortfall ===")
var_summary = var_es_summary(portfolio_returns)
print(var_summary)

# Monte Carlo Analysis
print("\n=== Monte Carlo Risk Analysis ===")
mc_risk = MonteCarloRisk(returns)
mc_results = mc_risk.portfolio_risk_metrics(weights)
for metric, value in mc_results.items():
    print(f"{metric}: {value:.4f}")

# Risk Attribution
print("\n=== Risk Attribution ===")
attribution = risk_attribution(returns, weights)
print(attribution)

# Backtesting
print("\n=== VaR Backtesting ===")
var_estimates = [parametric_var(returns.iloc[:i+252], 0.95) 
                for i in range(len(returns)-252)]
backtest = VaRBacktesting(portfolio_returns.iloc[252:], 
                         var_estimates, 0.95)
backtest_results = backtest.backtesting_summary()
for metric, value in backtest_results.items():
    print(f"{metric}: {value}")
```

## Key Risk Management Principles

1. **Diversification**: Use correlation analysis and risk attribution to ensure proper diversification
1. **Multiple Methods**: Always use multiple VaR methodologies for robustness
1. **Regular Backtesting**: Continuously validate risk models with out-of-sample testing
1. **Stress Testing**: Regularly conduct scenario analysis and stress tests
1. **Model Risk**: Be aware of model limitations and regularly update methodologies
1. **Time-Varying Risk**: Consider GARCH models for time-varying volatility
1. **Fat Tails**: Use Student’s t-distribution or empirical methods for heavy-tailed distributions

This reference card provides a comprehensive toolkit for implementing market risk management in Python, covering everything from basic volatility calculations to advanced portfolio risk attribution.