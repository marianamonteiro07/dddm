import warnings
import pandas as pd
import numpy as np
from arch.univariate import arch_model
from statsmodels.tsa.statespace.sarimax import SARIMAX

class MovePredictor(object):
    def __init__(self, past_moves, past_sales, market, gain=1, loss=9):
        self.past_moves = past_moves
        self.past_sales = past_sales
        self.market = market
        
        self.gain = gain
        self.loss = loss
        self.sp = 0.025   # sales estimate increase percentage
        self.tf = 365*2   # time frame to consider when fitting the model
        self.guesstimate_past_sales()

        self.arma_params = None
        self.arma_trend = None
        self.arma_aic = None
        self.garch_params = None
        self.garch_aic = None

    def guesstimate_past_sales(self):
        if len(self.past_sales) != len(self.past_moves):
            raise '[!] past_sales and past_moves should have the same length!'
        if len(self.past_sales) > self.tf:
            tf_sales = self.past_sales[-self.tf:]
            tf_moves = self.past_moves[-self.tf:]

        # guesstimate past sales
        for i, value in enumerate(tf_sales):
            if value == tf_moves[i]:
                tf_sales[i] = int(tf_sales[i]*(1 + self.sp))
        
        self.past_sales = np.sqrt(tf_sales)

    def predict(self):
        # guess initial parameters
        self.find_best_model_params()

        count = 0
        while True:
            move, q75 = self.next_move()

            sold = self.market(move)
            if sold == None:   # game over
                return
            
            # In case all sold assume 5% overdemand
            if move == sold:
                self.past_sales = np.insert(
                    self.past_sales[1:], self.past_sales.size-1, np.sqrt(q75))
            else:
                self.past_sales = np.insert(
                    self.past_sales[1:], self.past_sales.size-1, np.sqrt(sold))
            
            count += 1
            print('iter. ', count)


    def find_best_model_params(self):
        """
        This method finds the best parameters for the ARMA-GARCH model for the
        current time frame.
        
        The number of combinations is small and the method will only be called
        on occasion, so a more complex method for choosing the right parameters
        should not be needed.
        """
        print('Finding ARMA-GARCH parameters...')

        best_arma_aic = float('inf')
        best_arma_params = []
        for p in range(6):
            for q in range(6):
                for t in ['c', 't']:
                    try:
                        model = SARIMAX(self.past_sales, order=[p, 0, q], trend=t)
                        model_fit = model.fit(disp=0, iprint=0)
                        if model_fit.aic < best_arma_aic:
                            best_arma_aic = model_fit.aic
                            best_arma_params = [p, 0, q, t]
                    except:
                        pass

        arma_model = SARIMAX(
            self.past_sales,
            order=best_arma_params[:3],
            trend=best_arma_params[3],
        )
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

        self.arma_params = best_arma_params[:3]
        self.arma_trend = best_arma_params[3]
        self.arma_aic = best_arma_aic
        self.garch_params = best_garch_params
        self.garch_aic = best_garch_aic
        print('...new model found: ARMA({},{})-GARCH({},{}), trend: {}'.format(
            self.arma_params[0], self.arma_params[2],
            self.garch_params[0], self.garch_params[1],
            self.arma_trend,
        ))

    def next_move(self):
        try:
            temp_intervals, arma_aic, garch_aic = self.forecast_horizon_1()
        except:
            # Sometimes SARMIX crashes after many iterations, forcing
            # recalibration should fix it
            arma_aic = float('inf')
        
        # If AIC for either ARMA or GARCH increases by more than 5%, parameters
        # should be recalibrated. Saves time from calibrating every iteration.
        if arma_aic > 1.05 * self.arma_aic or garch_aic > 1.05 * self.garch_aic:
            self.find_best_model_params()
            temp_intervals, arma_aic, garch_aic = self.forecast_horizon_1()
        
        temp_move, q75 = self.get_best_from_intervals(temp_intervals)
        return temp_move, q75

    def forecast_horizon_1(self):
        # Fit ARMA
        arma_model = SARIMAX(self.past_sales, order=self.arma_params,
                             trend=self.arma_trend)
        arma_fit = arma_model.fit(disp=0)

        # Fit GARCH
        garch_model = arch_model(arma_fit.resid, vol='garch',
                                 p=self.garch_params[0], q=self.garch_params[1])
        garch_fit = garch_model.fit(disp='off')

        # Forecast    
        arma_forec = arma_fit.get_forecast()
        arma_forec_mean = arma_forec.predicted_mean
        arma_forec_ci = [arma_forec.conf_int()[0][0], arma_forec.conf_int()[0][1]]

        garch_forec = garch_fit.forecast(horizon=1, method='simulation')
        garch_forec_mean = garch_forec.mean['h.1'].iloc[-1]
        garch_forec_variance = garch_forec.variance['h.1'].iloc[-1]

        interval = [
            (arma_forec_ci[0] + garch_forec_mean - garch_forec_variance)**2,
            (arma_forec_ci[1] + garch_forec_mean + garch_forec_variance)**2
        ]
        return interval, arma_fit.aic, garch_fit.aic

    def get_best_from_intervals(self, interval):
        def calc_profit(demand, move):
            if demand >= move:
                return move * self.gain
            return demand * self.gain - (move - demand) * self.loss

        width = int(interval[1]) - int(interval[0])
        inbetween = int(width / 6)
        values_to_consider = [
            int(interval[0]),
            int(interval[0] + inbetween),
            int(interval[0] + inbetween * 2),
            int(interval[0] + inbetween * 3),
            int(interval[0] + inbetween * 4),
            int(interval[0] + inbetween * 5),
            int(interval[1]),
        ]

        max_profit = -float('inf')
        best_guess = None
        for guess in values_to_consider:
            profit = (
                calc_profit(values_to_consider[0], guess) * 0.025 +
                calc_profit(values_to_consider[1], guess) * 0.075 +
                calc_profit(values_to_consider[2], guess) * 0.2 +
                calc_profit(values_to_consider[3], guess) * 0.4 +
                calc_profit(values_to_consider[4], guess) * 0.2 +
                calc_profit(values_to_consider[5], guess) * 0.075 +
                calc_profit(values_to_consider[6], guess) * 0.025
            )
            if int(profit) >= max_profit:
                max_profit = int(profit)
                best_guess = guess

        return best_guess, values_to_consider[4]


def order(past_moves, past_sales, market):
    """ function implementing a simple strategy; parameters:
        * past_sales: list with historical data for the market trend
        * market: function to evaluate the actual sales; 'market(value)' returns:
                - inventory, if smaller than demand (everything was sold)
                - true demand otherwise (i.e., some inventory wasn't sold)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        mp = MovePredictor(past_moves, past_sales, market)
        mp.predict()


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
    past_moves = list(data.Inventory[:10000,])
    past_sales = list(data.Sales[:10000,])
    future_demand = list(data.Sales[10000:,])
    gain = 1
    loss = 9
    nmoves = 0
    # from student1 import order as order_fn   # import student's order function
    order_fn = order
    # profit = evaluate(past_moves, past_sales, future_demand, gain, loss, order_fn)
    # print("profit", profit)



    from datetime import datetime
    startTime = datetime.now()
    profit = evaluate(past_moves, past_sales, future_demand, gain, loss, order_fn)
    print("profit", profit)
    print('time: ', datetime.now() - startTime)
