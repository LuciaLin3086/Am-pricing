import numpy as np
import pandas as pd
from math import exp, log, sqrt
import statsmodels.api as sm

# input variables
St = 50
Save_t = 50
K = 50
r = 0.1
q = 0.05
sigma = 0.8
t = 0.25
left_time = 0.25

# hyperparameter
n = 3
N = 10000
rep = 20


class PriceSimulator:
    def __init__(self, St, Save_t, r, q, sigma, t, left_time, n, N):
        self.St = St
        self.Save_t = Save_t
        self.r = r
        self.q = q
        self.sigma = sigma
        self.t = t
        self.left_time = left_time
        self.n = n
        self.N = N

        self.delta_T = self.left_time / self.n

    def price_simulate(self):
        std_normal_sample = pd.DataFrame(np.random.randn(int(self.n), int(self.N)))  # 一個column代表一個path
        normal_sample = std_normal_sample * self.sigma * sqrt(self.delta_T) + \
                        (self.r - self.q - self.sigma ** 2 / 2) * self.delta_T
        path_price = normal_sample.cumsum(axis=0)  # 沿著row走，代表分別對每個column運算
        lnSt = path_price + log(self.St)
        St_df = np.exp(lnSt)  # St為 n * N 的dataframe，每個column為第1期到第n期的模擬股價路徑

        return St_df ## 沒有包含St


values_ls = []
for i in range(rep):
    price_simulate = PriceSimulator(St, Save_t, r, q, sigma, t, left_time, n, N)
    St_df = price_simulate.price_simulate()

    option_val_df = pd.DataFrame(np.zeros((n, N)))

    # maturity payoff = HV
    # 記得要加入Save_t計算平均
    Save_T = (St_df.sum(axis = 0) + Save_t * (t / left_time * n + 1)) / (n + t / left_time * n + 1)
    option_val_df.iloc[n - 1, :] = np.where(Save_T - K > 0, Save_T - K, 0)

    # backward-induction
    for time in range(n - 2, -1, -1):  ### t = n-2, n-3,...,1,0 ###
        # exercise value list  # [0: time + 1, :] 代表row 0 ~ row time
        # 記得要加入Save_t計算平均
        Save = (St_df.iloc[0: time + 1, :].sum(axis = 0) + Save_t * (t / left_time * n + 1)) / (time + 1 + t / left_time * n + 1)
        EV_ls = np.where(Save - K > 0, Save - K, 0)  # 1 * N
        # in-the-money paths
        path_index = EV_ls > 0  # path_index = [True False True ... False False]  # 1 * N
        if True in path_index:
            # holding value list: present values of the option values at the next time point
            delta_T = price_simulate.delta_T
            HV_ls = option_val_df.iloc[time + 1, :].values * exp(- r * delta_T)  # 1 * N
            # for in-the-money paths
            x1_ls = St_df.iloc[time, path_index].values
            x2_ls = Save[path_index]

            S_variables = np.column_stack((np.array(list(map(lambda St: St, x1_ls))),
                                           np.array(list(map(lambda St: St ** 2, x1_ls))),
                                           np.array(list(map(lambda Save_t: Save_t, x2_ls))),
                                           np.array(list(map(lambda Save_t: Save_t ** 2, x2_ls))),
                                           np.array(list(map(lambda St, Save_t: St * Save_t, x1_ls, x2_ls)))))
            S_variables = sm.add_constant(S_variables)  # remember the constant term

            regression_eq = sm.OLS(HV_ls[path_index], S_variables)
            result = regression_eq.fit()  # fit 完才有回歸結果
            expected_HV = result.predict()  # predicted values of HV

            option_val_df.iloc[time, :] = HV_ls
            option_val_df.iloc[time, path_index] = np.where(EV_ls[path_index] > expected_HV, EV_ls[path_index],
                                                            HV_ls[path_index])
        else:  # There is no True in path_index.
            HV_ls = option_val_df.iloc[time + 1, :].values * exp(- r * delta_T)  # 1 * N
            option_val_df.iloc[time, :] = HV_ls

    HV = np.mean(option_val_df.iloc[0, :].values) * exp(- r * delta_T)
    EV = max(Save_t - K , 0)
    option_value = max(HV, EV)

    values_ls.append(option_value)

values_arr = np.array(values_ls)
CI1 = values_arr.mean() - 2 * values_arr.std()
CI2 = values_arr.mean() + 2 * values_arr.std()

print(f"American arithmetic average calls = {values_arr.mean():.6f}")
print(f"95% CI: [{CI1:.6f}, {CI2:.6f}]")














