import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# Charger les données
data = pd.read_csv(r'C:\Users\jerem\OneDrive\Documents\1_SABR project GPT\excels\Refinitiv_datas_with_maturity.csv')

# Fonction pour calculer les volatilités par delta
def calculate_volatilities(row):
    vol_25d_call = row['ATM volatility'] + row['25D RR'] / 2 + row['25D BF']
    vol_25d_put = row['ATM volatility'] - row['25D RR'] / 2 + row['25D BF']
    vol_10d_call = row['ATM volatility'] + row['10D RR'] / 2 + row['10D BF']
    vol_10d_put = row['ATM volatility'] - row['10D RR'] / 2 + row['10D BF']
    return pd.Series([vol_25d_call, vol_25d_put, vol_10d_call, vol_10d_put], 
                     index=['25D Call', '25D Put', '10D Call', '10D Put'])

def calculate_strike(F, sigma, T, delta, phi):
    return F * np.exp((sigma**2/2)*T - phi*sigma*np.sqrt(T)*norm.ppf(phi*np.exp(-sigma**2*T/2)*delta))

def sabr_vol(alpha, beta, rho, nu, F, K, T):
    if F == K:
        return alpha / (F**(1 - beta)) * (1 + (((2 - 3 * rho**2) * nu**2 * T) / 24))
    FK_beta = F**(1 - beta) - K**(1 - beta)
    z = nu / alpha * FK_beta / (1 - beta)
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    factor = alpha * z / x_z if x_z != 0 else alpha
    corr = 1 + (
        (((1 - beta)**2 * alpha**2) / (24 * (F * K)**(1 - beta))) +
        (rho * beta * nu * alpha / (4 * (F * K)**((1 - beta) / 2))) +
        (((2 - 3 * rho**2) * nu**2) / 24)
    ) * T
    return factor * corr / ((F * K)**((1 - beta) / 2))

def objective(params, strikes, vols, F, T, fixed_alpha):
    beta, rho, nu = params
    alpha = fixed_alpha
    model_vols = [sabr_vol(alpha, beta, rho, nu, F, K, T) for K in strikes]
    return np.sum((np.array(model_vols) - np.array(vols))**2)

def get_bounds(maturity):
    if maturity <= 0.08:  # 1M
        return [(0.64, 0.65), (-0.5, 0.5), (1e-4, 2)]
    elif maturity <= 0.16:  # 2M
        return [(0.64, 0.65), (-0.6, 0.6), (1e-4, 3)]
    elif maturity <= 0.25:  # 3M
        return [(0.64, 0.65), (-0.7, 0.7), (1e-4, 4)]
    elif maturity <= 0.5:  # 6M
        return [(0.64, 0.65), (-0.8, 0.8), (1e-4, 5)]
    else:  # 9M
        return [(0.64, 0.65), (-0.9, 0.9), (1e-4, 6)]

# Calculer les volatilités
data[['25D Call', '25D Put', '10D Call', '10D Put']] = data.apply(calculate_volatilities, axis=1)

# Calculer les strikes
for index, row in data.iterrows():
    F = row['Forward']
    T = row[' Maturity ']
    
    data.loc[index, 'Strike_ATM'] = F
    data.loc[index, 'Strike_25D_Put'] = calculate_strike(F, row['25D Put'], T, -0.25, -1)
    data.loc[index, 'Strike_10D_Put'] = calculate_strike(F, row['10D Put'], T, -0.1, -1)
    data.loc[index, 'Strike_25D_Call'] = calculate_strike(F, row['25D Call'], T, 0.25, 1)
    data.loc[index, 'Strike_10D_Call'] = calculate_strike(F, row['10D Call'], T, 0.1, 1)

# Calculer les rendements et statistiques par tenor
data['Spot_Returns'] = np.log(data['Spot'] / data['Spot'].shift(1))
data['ATM_Vol_Returns'] = np.log(data['ATM volatility'] / data['ATM volatility'].shift(1))

tenor_stats = data.groupby(' Tenor ').apply(
    lambda group: pd.Series({
        'Vol_of_Vol': group['ATM_Vol_Returns'].std() * np.sqrt(252),
        'Spot_Vol_Correlation': group['Spot_Returns'].corr(group['ATM_Vol_Returns'])
    })
).reset_index()

# Calibration SABR
calibration_results = pd.DataFrame()

for index, row in data.iterrows():
    strikes = [
        row['Strike_10D_Put'],
        row['Strike_25D_Put'],
        row['Strike_ATM'],
        row['Strike_25D_Call'],
        row['Strike_10D_Call']
    ]
    
    vols = [
        row['10D Put'],
        row['25D Put'],
        row['ATM volatility'],
        row['25D Call'],
        row['10D Call']
    ]
    
    F = row['Forward']
    T = row[' Maturity ']
    fixed_alpha = row['ATM volatility']
    
    bounds = get_bounds(T)
    initial_guess = [0.5, 0.0, 0.2]
    
    result = minimize(
        objective,
        initial_guess,
        args=(strikes, vols, F, T, fixed_alpha),
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    result_series = pd.Series({
        'Date': row['Date'],
        'Tenor': row[' Tenor '],
        'alpha': fixed_alpha,
        'beta': result.x[0],
        'rho': result.x[1],
        'nu': result.x[2],
        'objective_value': result.fun,
        'Forward': F,
        'Maturity': T
    })
    
    calibration_results = pd.concat([calibration_results, result_series.to_frame().T], ignore_index=True)

# Obtenir les paramètres les plus récents
latest_date = data['Date'].iloc[0]
latest_params = calibration_results[calibration_results['Date'] == latest_date].copy()
latest_params = latest_params.merge(tenor_stats, left_on='Tenor', right_on=' Tenor ', how='left')

# Sauvegarder les résultats
with pd.ExcelWriter('C:/Users/jerem/OneDrive/Documents/1_SABR project GPT/excels/sabr_calibration_with_realised.xlsx') as writer:
    calibration_results.to_excel(writer, sheet_name='Calibration_Parameters', index=False)
    data.to_excel(writer, sheet_name='Raw_Data', index=False)
    latest_params.to_excel(writer, sheet_name='Latest_Parameters', index=False)
