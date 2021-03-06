import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

def smape(y, yhat):
  n = y.shape[0]
  return 100.0 / n * np.sum(np.abs(y - yhat) / (0.5 * (np.abs(y) + np.abs(yhat))))

def tsplot(y, lags=None, figsize=(10, 8)):
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(1.5) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax