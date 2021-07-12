from datetime import datetime, timedelta
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def import_world_data():
    # Import deaths, infected and recovered datasets

    url_i = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    df_i = pd.read_csv(url_i)

    url_d = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    df_d = pd.read_csv(url_d)

    url_r = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
    df_r = pd.read_csv(url_r)

    # Get total population by country
    df_p = pd.read_csv("../data/population_by_country_2020.csv")

    # Rename some big countries which do not match
    df_p.loc[df_p['Country (or dependency)'] ==
             'United States', 'Country (or dependency)'] = 'US'
    df_p.loc[df_p['Country (or dependency)'] == 'Taiwan',
             'Country (or dependency)'] = 'Taiwan*'
    df_p.loc[df_p['Country (or dependency)'] == 'Czech Republic (Czechia)',
             'Country (or dependency)'] = 'Czechia'
    df_p.loc[df_p['Country (or dependency)'] == 'South Korea',
             'Country (or dependency)'] = 'Korea, South'

    return df_i, df_d, df_r, df_p


def extract_single_country(country, df_i, df_r, df_d, df_p):
    """
    Function that extracts s, i, r, d in time for a given country,
    and yields them up to the specified time.
    Also yields the total population of that country, as per 2020.
    Args:
        country (str); name of the country
        df_i (DataFrame): contains the number of infected people
        df_r (DataFrame): contains the number of recovered people
        df_d (DataFrame): contains the number of deaths
        df_p (DataFrame): contains the population per country

    Returns:
        df_final (DataFrame): dataframe with the time series for the country
        pop (int): population of that country
    """
    names = ['infected', 'recovered', 'deaths']
    for i, df in enumerate([df_i, df_r, df_d]):
        df_ = df[df['Country/Region'] == country]
        df_ = df_.drop(columns=['Province/State',
                                'Country/Region', 'Lat', 'Long'])

        # If a there is a single entry for a country
        if len(df_) == 1:
            cname = df_.index[0]
            df_ = df_.T
            df_.index = pd.to_datetime(df_.index)
            df_ = df_.rename(columns={cname: names[i]})

        # If a country has multiple regions, collapse them
        else:
            df_ = pd.DataFrame(df_.sum()).rename(columns={0: names[i]})
            df_.index = pd.to_datetime(df_.index)

        # Create or append columns
        if i == 0:
            df_final = df_
        else:
            df_final[names[i]] = df_

    pop = int(df_p[df_p['Country (or dependency)'] == country]
              ['Population (2020)'].values[0])
    df_final = df_final.rename(columns={'infected': 'total_infected'})
    df_final['recovered'] = df_final['recovered'] + df_final['deaths']
    df_final['healed'] = df_final['recovered'] - df_final['deaths']
    df_final['infected'] = df_final['total_infected'] - df_final['recovered']
    df_final['susceptible'] = pop - \
        df_final['recovered'] - df_final['infected']
    df_final = df_final.rolling(7, win_type='gaussian').mean(std=3)

    return df_final, pop


class Country():
    def __init__(self, name, data, population, i0, t0):
        self.name = name
        self.data = data
        self.population = population
        self.i0 = i0  # Number of infected at infection step 0
        self.t0 = t0  # Date of infection step 0
        self.r0 = 0
        self.s0 = self.population - self.i0 - self.r0
        self.fit_gamma()
        self.find_transitions()

    @staticmethod
    def build(name, df_i, df_r, df_d, df_p):
        data, pop = extract_single_country(name, df_i, df_r, df_d, df_p)
        t0 = data[data['infected'] > 0].index.strftime('%Y-%m-%d')[0]
        i0 = data.loc[t0, 'infected']
        country = Country(name, data.loc[t0:], pop, i0, t0)
        return country

    def find_transitions(self):
        res_diff = self.data.infected.resample('1W').sum()
        mins = res_diff[(res_diff.shift(1) > res_diff) &
                        (res_diff.shift(-1) > res_diff)
                        & (res_diff.shift(2) > res_diff) &
                        (res_diff.shift(-2) > res_diff)
                        & (res_diff.shift(-3) > res_diff) &
                        (res_diff.shift(3) > res_diff)
                        & (res_diff.shift(-4) > res_diff) &
                        (res_diff.shift(4) > res_diff)
                        & (res_diff.shift(-5) > res_diff) &
                        (res_diff.shift(5) > res_diff)
                        & (res_diff.shift(-6) > res_diff) &
                        (res_diff.shift(6) > res_diff)]
        self.minima = mins.index - timedelta(days=7)
        self.block_starts = [self.data.index[0]] + list(self.minima)
        self.block_ends = list(self.minima -
                               timedelta(days=1)) + [self.data.index[-1]]

    def fit_gamma(self):
        minlag = 10
        maxlag = 30
        autocorrs = np.array([self.data['recovered'].corr(
            self.data['total_infected'].shift(lag))
            for lag in range(minlag, maxlag)])
        self.gamma = 1/np.arange(minlag, maxlag)[np.argmax(autocorrs)]

    def sir_simulate(self, times, beta, gamma):
        """
        Function that integrates a SIR model given the steps, beta, gamma,
        i0, and the total popolation n.
        Args:
            times (list); list of time steps
            beta (float): infection rate
            gamma (float): inverse recovery time
        Returns:
            s (np.array): susceptible
            i (np.array): infected
            r (np.array): recovered
        """
        steps = len(times)
        s, i, r = np.empty(steps), np.empty(steps), np.empty(steps)
        s[0], i[0], r[0] = self.s0, self.i0, self.r0
        for t in np.arange(steps-1):
            v1 = beta*i[t]*s[t]/self.population
            v2 = gamma*i[t]
            s[t+1] = s[t] - v1
            i[t+1] = i[t] + v1 - v2
            r[t+1] = r[t] + v2

        return np.ravel([s, i, r])

    def sivr_simulate(self, times, beta, vrate, vstart):
        """
        Function that integrates a SIR model given the steps, beta, gamma,
        i0, and the total popolation n.
        Args:
            times (list); list of time steps
            beta (float): infection rate
            gamma (float): inverse recovery time
        Returns:
            s (np.array): susceptible
            i (np.array): infected
            r (np.array): recovered
        """
        steps = len(times)
        s, i, r, v = (np.empty(steps), np.empty(steps),
                      np.empty(steps), np.empty(steps))
        s[0], i[0], r[0], v[0] = self.s0, self.i0, self.r0, 0
        for t in np.arange(steps-1):
            v1 = beta*i[t]*s[t]/self.population
            v2 = self.gamma*i[t]
            v3 = vrate*s[t]*1/(1+np.exp(t*0.15 - vstart*0.15))
            s[t+1] = s[t] - v1 - v3
            i[t+1] = i[t] + v1 - v2
            r[t+1] = r[t] + v2
            v[t+1] = v[t] + v3
        return np.ravel([s, i, r+v])

    def sir_simulate_lockdown(self, times, beta1, beta2, tl):
        """
        Function that integrates a SIR model with variable infection
        rate beta. The beta is varied between a beta1 and a beta2,
        using a sigmoid function as smoother. The switch time is tl.
        Args:
            times (list); list of time steps
            beta1 (float): infection rate
            beta2 (float): infection rate during lockdown
            tl (float): time of lockdown (switch between betas)

        Returns:
            s (np.array): susceptible
            i (np.array): infected
            r (np.array): recovered
        """

        steps = len(times)
        s, i, r = np.empty(steps), np.empty(steps), np.empty(steps)
        s[0], i[0], r[0] = self.s0_temp, self.i0_temp, self.r0_temp
        beta_sigm = beta1 + \
            (beta2 - beta1) / \
            (1+np.exp(-0.15*np.arange(steps-1) + 0.15*tl))
        for t in np.arange(steps-1):
            v1 = beta_sigm[t]*i[t]*s[t]/self.population
            v2 = self.gamma*i[t]
            s[t+1] = s[t] - v1
            i[t+1] = i[t] + v1 - v2
            r[t+1] = r[t] + v2
        return np.ravel([s, i, r])

    def fit_sir(self, start, end):
        self.s0_temp = self.data.loc[start, 'susceptible']
        self.i0_temp = self.data.loc[start, 'infected']
        self.r0_temp = self.data.loc[start, 'recovered']
        ydata = np.hstack([self.data.loc[start:end, 'susceptible'].values,
                           self.data.loc[start:end, 'infected'].values,
                           self.data.loc[start:end, 'recovered'].values])
        xdata = np.arange(len(ydata)//3)
        popt, pcov = curve_fit(self.sir_simulate,
                               xdata, ydata, bounds=(
                                   [1e-4, 1e-4], [0.5, 1]),
                               p0=[0.1, self.gamma])
        self.beta1 = popt[0]
        self.gamma = popt[1]

    def predict_sir(self, start, end):
        xdata = np.arange(len(self.data.loc[start:end, 'infected']))
        out = self.sir_simulate(
            xdata, self.beta1, self.gamma)
        shat, ihat, rhat = out[:len(xdata)], out[len(
            xdata):2*len(xdata)], out[2*len(xdata):]
        return shat, ihat, rhat

    def fit_sivr(self, start, end):
        self.s0_temp = self.data.loc[start, 'susceptible']
        self.i0_temp = self.data.loc[start, 'infected']
        self.r0_temp = self.data.loc[start, 'recovered']
        ydata = np.hstack([self.data.loc[start:end, 'susceptible'].values,
                           self.data.loc[start:end, 'infected'].values,
                           self.data.loc[start:end, 'recovered'].values])
        xdata = np.arange(len(ydata)//3)
        popt, pcov = curve_fit(self.sivr_simulate,
                               xdata, ydata, bounds=(
                                   [1e-4, 1e-6, 1], [0.5, 1e-3, 1000]),
                               p0=[0.1, 2e-4, 200])
        self.beta1 = popt[0]
        self.vrate = popt[1]
        self.vstart = datetime.strptime(
            start, '%Y-%m-%d') + timedelta(days=int(popt[2]))

    def predict_sivr(self, start, end):
        xdata = np.arange(len(self.data.loc[start:end, 'infected']))
        delta_vstart = (self.vstart -
                        datetime.strptime(start, '%Y-%m-%d')).days
        out = self.sivr_simulate(
            xdata, self.beta1, self.vrate, delta_vstart)
        shat, ihat, rhat = out[:len(xdata)], out[len(
            xdata):2*len(xdata)], out[2*len(xdata):]
        return shat, ihat, rhat

    def fit_single_lockdown(self, start, end):
        self.s0_temp = self.data.loc[start, 'susceptible']
        self.i0_temp = self.data.loc[start, 'infected']
        self.r0_temp = self.data.loc[start, 'recovered']
        ydata = np.hstack([self.data.loc[start:end, 'susceptible'].values,
                           self.data.loc[start:end, 'infected'].values,
                           self.data.loc[start:end, 'recovered'].values])
        xdata = np.arange(len(ydata)//3)
        popt, pcov = curve_fit(self.sir_simulate_lockdown,
                               xdata, ydata, bounds=(
                                   [1e-4, 1e-4, 1], [0.3, 0.3, 300]),
                               p0=[0.1, 0.01, 60])
        self.beta1 = popt[0]
        self.beta2 = popt[1]
        self.tl = datetime.strptime(
            start, '%Y-%m-%d') + timedelta(days=int(popt[2]))

    def predict_single_lockdown(self, start, end):
        xdata = np.arange(len(self.data.loc[start:end, 'infected']))
        delta_tl = (self.tl -
                    datetime.strptime(start, '%Y-%m-%d')).days
        out = self.sir_simulate_lockdown(
            xdata, self.beta1, self.beta2, delta_tl)
        shat, ihat, rhat = out[:len(xdata)], out[len(
            xdata):2*len(xdata)], out[2*len(xdata):]
        return shat, ihat, rhat

    def fit_multiple_lockdown(self, start, end):
        # TODO: implement start funcitonality.
        # Now works only if start is not specified.
        self.s0_temp = self.data.loc[start, 'susceptible']
        self.i0_temp = self.data.loc[start, 'infected']
        self.r0_temp = self.data.loc[start, 'recovered']
        self.betas = []
        self.tls = []

        for s, e in zip(self.block_starts,  self.block_ends):
            ydata = np.hstack([self.data.loc[s:e, 'susceptible'].values,
                               self.data.loc[s:e, 'infected'].values,
                               self.data.loc[s:e, 'recovered'].values])
            xdata = np.arange(len(ydata)//3)
            if len(self.betas) == 0:
                popt, pcov = curve_fit(self.sir_simulate_lockdown,
                                       xdata, ydata,
                                       bounds=([1e-2, 1e-2, 1],
                                               [0.3, 0.3, 300]),
                                       p0=[0.1, 1e-2, 60])
                self.betas = [popt[0],  popt[1]]
                self.tls = [s + timedelta(days=int(popt[2]))]
                out = self.sir_simulate_lockdown(xdata, *popt)
                shat, ihat, rhat = out[:len(xdata)], out[len(
                    xdata):2*len(xdata)], out[2*len(xdata):]
                self.s0_temp, self.i0_temp, self.r0_temp = shat[
                    -1], ihat[-1], rhat[-1]
            else:
                popt, pcov = curve_fit(self.sir_simulate_lockdown,
                                       xdata, ydata,
                                       bounds=([1e-2, 1e-2, 1],
                                               [0.3, 0.3, 300]),
                                       p0=[0.1, 1e-2, 60])
                self.betas.extend([popt[0], popt[1]])
                self.tls.append(s + timedelta(days=int(popt[2])))
                out = self.sir_simulate_lockdown(xdata, *popt)
                shat, ihat, rhat = out[:len(xdata)], out[len(
                    xdata):2*len(xdata)], out[2*len(xdata):]
                self.s0_temp, self.i0_temp, self.r0_temp = shat[
                    -1], ihat[-1], rhat[-1]

    def predict_multiple_lockdown(self, start, end):
        # TODO: implement start funcitonality.
        # Now works only if start is not specified.
        self.s0_temp = self.data.loc[start, 'susceptible']
        self.i0_temp = self.data.loc[start, 'infected']
        self.r0_temp = self.data.loc[start, 'recovered']
        shat, ihat, rhat = np.array([]), np.array([]), np.array([])

        ends = sum([i.date() < datetime.strptime(end, '%Y-%m-%d').date()
                    for i in self.block_ends]) + 1
        block_ends_ = deepcopy(self.block_ends)
        block_ends_[ends-1] = datetime.strptime(end, '%Y-%m-%d')
        for i in np.arange(ends):
            s, e = self.block_starts[i], block_ends_[i]
            xdata = np.arange(
                len(self.data.loc[s:e, 'susceptible'].values))
            tl_ = (self.tls[i] - s).days
            out = self.sir_simulate_lockdown(
                xdata, self.betas[2*i], self.betas[2*i+1],
                tl_)
            shat = np.append(shat, out[:len(xdata)])
            ihat = np.append(ihat, out[len(xdata):2*len(xdata)])
            rhat = np.append(rhat, out[2*len(xdata):])
            self.s0_temp, self.i0_temp, self.r0_temp = shat[
                -1], ihat[-1], rhat[-1]
        return shat, ihat, rhat

    def fit_lockdown(self, start, end):
        if len(self.minima) == 0:
            self.fit_single_lockdown(start, end)
        else:
            self.fit_multiple_lockdown(start, end)

    def predict_lockdown(self, start, end):
        if len(self.minima) == 0:
            return self.predict_single_lockdown(start, end)
        else:
            return self.predict_multiple_lockdown(start, end)

    def fit(self, method='sir', start=None, end=None):
        if start is None:
            start = self.t0
        if end is None:
            end = self.data.index.strftime('%Y-%m-%d')[-1]
        self.fit_gamma()
        if method == 'sir':
            self.fit_sir(start, end)
            self.method = 'sir'
        elif method == 'single_lockdown':
            self.fit_single_lockdown(start, end)
            self.method = 'single_lockdown'
        elif method == 'lockdown':
            self.fit_lockdown(start, end)
            self.method = 'lockdown'
        elif method == 'sivr':
            self.fit_sivr(start, end)
            self.method = 'sivr'
        else:
            print("ERROR: method not understood. \
Options are: 'sir', 'single_lockdown', 'lockdown', 'sivr'")

    def predict(self, start=None, end=None):
        if start is None:
            start = self.t0
        if end is None:
            end = self.data.index.strftime('%Y-%m-%d')[-1]
        if self.method == 'sir':
            return self.predict_sir(start, end)
        elif self.method == 'single_lockdown':
            return self.predict_single_lockdown(start, end)
        elif self.method == 'lockdown':
            return self.predict_lockdown(start, end)
        elif self.method == 'sivr':
            return self.predict_sivr(start, end)

    def plot_predictions(self, shat, ihat, rhat, start=None, end=None):
        if start is None:
            start = self.t0
        if end is None:
            end = self.data.index.strftime('%Y-%m-%d')[-1]
        s, i, r = (self.data.loc[start:end, 'susceptible'],
                   self.data.loc[start:end, 'infected'],
                   self.data.loc[start:end, 'recovered'])
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        axs[0].plot(self.data.loc[start:end].index, s, color='tab:blue')
        axs[0].plot(self.data.loc[start:end].index, shat,
                    color='tab:blue', linestyle='--')
        axs[0].set_ylabel("susceptible")

        axs[1].plot(self.data.loc[start:end].index, i, color='tab:orange')
        axs[1].plot(self.data.loc[start:end].index, ihat,
                    color='tab:orange', linestyle='--')
        axs[1].set_ylabel("Infected")

        axs[2].plot(self.data.loc[start:end].index, r, color='tab:green')
        axs[2].plot(self.data.loc[start:end].index, rhat,
                    color='tab:green', linestyle='--')
        axs[2].set_ylabel("Recovered")

        # axs[2].vlines(self.minima, 0, np.max(r), 'red', linestyle='-.')
        # axs[1].vlines(self.minima, 0, np.max(i), 'red', linestyle='-.')
        # axs[0].set_ylim(np.min(s)-1e5, np.max(s)+1e5)
        # axs[1].set_ylim(0, 1.2*np.max(i))
        # axs[2].set_ylim(0, 1.2*np.max(r))

        for ax in axs:
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
        axs[1].set_title(self.name)
        plt.tight_layout()

        s_error = np.mean(np.abs(s-shat)/self.population)
        i_error = np.mean(np.abs(i-shat)/self.population)
        r_error = np.mean(np.abs(r-shat)/self.population)

        return s_error, i_error, r_error
