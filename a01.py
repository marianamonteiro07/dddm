import warnings
import pandas as pd
import numpy as np
from arch.univariate import arch_model
from statsmodels.tsa.arima_model import ARMA

def get_best_params(sales):
    # To be optimized if i have the time
    sales = pd.DataFrame({'sales': np.sqrt(past_sales)})

    best_aic = float('inf')
    best_params = []
    for p in range(6):
        for q in range(6):
            try:
                model = ARMA(sales.sales, order=[p, q])
                model_fit = model.fit(disp=0, iprint=0)
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_params = [p, q]
            except:
                pass

    arma_model = ARMA(sales.sales, order=best_params)
    arma_fit = arma_model.fit(disp=0)

    best_garch_aic = float('inf')
    best_garch_params = []
    for p in range(4):
        for q in range(4):
            try:
                am = arch_model(arma_fit.resid, vol='garch', p=p, q=q)
                arch_model_fitted = am.fit(disp='off')
                if arch_model_fitted.aic < best_garch_aic:
                    best_garch_aic = arch_model_fitted.aic
                    best_garch_params = [p, q]
            except:
                pass

    print('new parameters for ARMA ', best_params, ' and for GARCH ', best_garch_params)
    return best_params, best_aic, best_garch_params, best_garch_aic

def forecast_horizon_1(past_sales, arma_params, garch_params):
    sales = pd.DataFrame({'sales': np.sqrt(past_sales)})

    # Fit ARMA
    arma_model = ARMA(sales.sales, order=arma_params)
    arma_fit = arma_model.fit(disp=0)

    # Fit GARCH
    garch_model = arch_model(arma_fit.resid, vol='garch', p=garch_params[0], q=garch_params[1])
    garch_fit = garch_model.fit(disp='off')

    # Forecast    
    arma_forec = arma_fit.forecast()
    arma_forec_mean = arma_forec[0][0]
    arma_forec_ci = arma_forec[2][0]

    garch_forec = garch_fit.forecast(horizon=1, method='simulation')
    garch_forec_mean = garch_forec.mean['h.1'].iloc[-1]
    garch_forec_variance = garch_forec.variance['h.1'].iloc[-1]

    return int((arma_forec_mean + garch_forec_mean)**2), arma_fit.aic, garch_fit.aic

def next_move(sales, best):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        temp_move, arma_aic, garch_aic = forecast_horizon_1(
            sales, best['arma_params'], best['garch_params'])
        
        if arma_aic > 1.05 * best['arma_aic'] or garch_aic > 1.05 * best['garch_aic']:
            print('Updating ARMA-GARCH parameters...')
            best['arma_params'], best['arma_aic'], best['garch_params'], best['garch_aic'] = get_best_params(past_sales)
            print('...ARMA-GARCH parameters updated')
            temp_move, arma_aic, garch_aic = forecast_horizon_1(
                sales, best['arma_params'], best['garch_params'])
        
        return temp_move

def order(past_moves, past_sales, market):
    """ function implementing a simple strategy; parameters:
        * past_sales: list with historical data for the market trend
        * market: function to evaluate the actual sales; 'market(value)' returns:
                - inventory, if smaller than demand (everything was sold)
                - true demand otherwise (i.e., some inventory wasn't sold)
    """
    count = 0
    # sales estimate increase percentage
    sp = 0.025
    # guesstimate past sales
    for i, value in enumerate(past_sales):
        if value == past_moves[i]:
            past_sales[i] = int(past_sales[i]*(1+sp))
    # guess initial parameters
    print('Finding initial ARMA-GARCH parameters...')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        best_arma_params, best_arma_aic, best_garch_params, best_garch_aic = get_best_params(past_sales)
    best_params = {
        'arma_params': best_arma_params,
        'arma_aic': best_arma_aic,
        'garch_params': best_garch_params,
        'garch_aic': best_garch_aic,
    }
    print('...initial ARMA-GARCH parameters found')

    while True:
        move = next_move(past_sales[-731:], best_params)

        sold = market(move)
        if sold == None:   # game over
            return
        
        # In case all sold assume 5% overdemand
        if move == sold:
            past_sales.append(int(sold*(1+sp)))
        else:
            past_sales.append(sold)
        
        count += 1
        print('iter. ', count)

    # # MOST BASIC MODEL profit 1 002 572€
    # move = past_moves[-1]   # ignoring all data except last period
    # sold = past_sales[-1]
    # count = 0

    # while True:
    #     if move == sold:
    #         move *= (1.1 + .03 * count)
    #         count = count + 1 if count >= 0 else 0
    #     else:
    #         move *= (.9 + .03 * count)
    #         count = count - 1 if count <= 0 else 0
    #         if move < sold:
    #             move = sold
    #     move = int(round(move, 0))

    #     sold = market(move)
    #     if sold == None:   # game over
    #         return


def evaluate(past_moves, past_sales, demand, gain, loss, order_fn):
    """ evaluate student's moves; parameters:
        * past_moves: history of past inventories
        * past_sales: history of past sales
        * demand: true future demand (unknown to students)
        * gain: profit per sold unit
        * loss: deficit generated per unit unsold
        * order_fn: function implementing student's method
    """
    moves = []
    def market(move):
        """ demand function ("censored", as it is limited by 'move'); parameter:
            * move: quantity available for selling (i.e., inventory)
        """
        global nmoves
        if nmoves >= len(demand):
            return None
        moves.append(move)
        sales = min(move, demand[nmoves])
        nmoves += 1
        return sales
    
    profit = 0
    n = len(demand)
    orders = []
    sales = []
    order_fn(past_moves, past_sales, market)

    for i in range(n):
        if moves[i] > demand[i]:
            profit += demand[i]*gain - (moves[i]-demand[i])*loss
        else:
            profit += moves[i]*gain
        print(f"{i+1}\t{demand[i]}\t{moves[i]}\t{moves[i]-demand[i]}\t{profit}")
    return profit


if __name__ == "__main__":
    # past_moves = [100,97,91,90,87,96,86,95,87,82,81,89,83,91,82,78,86,81,74,81]   # decided inventory
    # past_sales = [97,91,90,87,87,85,86,87,82,81,81,83,83,79,78,78,81,74,74,81]    # sales observed
    # future_demand = [104,121,126,122,125,115,118,113,108,103,104,106,120,124,133,137,148,167,183,202]   # (unknown) future demand
    data = pd.read_csv('data-a01.csv', sep='\t')
    past_moves = list(data.Inventory[:1000,])
    past_sales = list(data.Sales[:1000,])
    future_demand = list(data.Sales[1000:1150,])
    gain = 1
    loss = 9
    nmoves = 0
    # from student1 import order as order_fn   # import student's order function
    order_fn = order
    profit = evaluate(past_moves, past_sales, future_demand, gain, loss, order_fn)
    print("profit", profit)
