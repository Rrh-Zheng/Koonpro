import numpy as np
import pandas as pd
from gluonts.evaluation import MultivariateEvaluator
from gluonts.model.forecast import SampleForecast
import matplotlib.pyplot as plt
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')
import os
from sklearn.preprocessing import StandardScaler
from scipy import stats
scaler = StandardScaler()
import seaborn as sns


def get_metric(truth, mean, var):
    actual_values = truth
    mean_values = mean
    std_dev = var

    num_samples = 100
    num_dimensions = mean_values.shape[1]
    num_steps = mean_values.shape[0]

    predicted_samples = np.random.normal(
        loc=mean_values[np.newaxis, :, :],
        scale=std_dev[np.newaxis, :, :],
        size=(num_samples, num_steps, num_dimensions)
    )

    time_index = pd.date_range(start='2024-01-01', periods=num_steps, freq='D')
    actual_df = pd.DataFrame(data=actual_values, index=time_index)

    actual_df.index = pd.to_datetime(actual_df.index)

    forecast = SampleForecast(
        samples=predicted_samples,
        start_date=time_index[0],
        item_id="multivariate_forecast",
        freq='D'
    )

    evaluator = MultivariateEvaluator(quantiles=(np.arange(20) / 20.0)[2:],
                                          target_agg_funcs={'sum': np.sum})

    agg_metric, _ = evaluator(
        [actual_df],
        [forecast],
    )

    return agg_metric["m_sum_mean_wQuantileLoss"], agg_metric["m_sum_NRMSE"]



def plot_interval(truth, mean, var):
    actual_values = truth
    mean_values = mean
    std_dev = var
    intervals = [0.5, 0.9]
    error = np.abs(truth - mean)
    std = var
    error = np.mean(error, axis=1)
    std = np.mean(std, axis=1)

    num_samples = 100
    num_dimensions = mean_values.shape[1]
    num_steps = mean_values.shape[0]

    predicted_samples = np.random.normal(
        loc=mean_values[np.newaxis, :, :],
        scale=std_dev[np.newaxis, :, :],
        size=(num_samples, num_steps, num_dimensions)
    )

    time_index = pd.date_range(start='2024-01-01', periods=num_steps, freq='H')
    actual_df = pd.DataFrame(data=actual_values, index=time_index)

    actual_df.index = pd.to_datetime(actual_df.index)

    forecast = SampleForecast(
        samples=predicted_samples,
        start_date=time_index[0],
        item_id="multivariate_forecast",
        freq='H'
    )

    pdf_filename = "multivariate_forecast.pdf"
    with PdfPages(pdf_filename) as pdf:
        fig = plt.figure(figsize=(35, 12))
        gs = gridspec.GridSpec(2, 5, figure=fig, width_ratios=[2, 1, 1, 1, 1])

        axes = []

        ax = fig.add_subplot(gs[:, 0])
        axes.append(ax)

        for dimension in range(4):
            ax = fig.add_subplot(gs[0, dimension + 1])
            axes.append(ax)

        for dimension in range(4):
            ax = fig.add_subplot(gs[1, dimension + 1])
            axes.append(ax)

        ax = axes[0]
        data = pd.DataFrame({
            'Mean Absolute Error': error,
            'Variance': (std-19.95)*1000
        })


        slope, intercept, r_value, p_value, std_err = stats.linregress(error, std)


        sns.regplot(x='Mean Absolute Error', y='Variance', data=data, ax=ax, scatter_kws={"s": 50},
                    line_kws={"color": "olive"})

        ax.set_xlabel("Mean Absolute Error", fontsize=25)
        ax.set_ylabel("Variance", fontsize=25)
        ax.set_title(f"Correlation: {r_value:.2f}, Significant level: {p_value:.2e}", fontsize=25)

        for label in ax.get_xticklabels():
            label.set_rotation(45)
        ax.tick_params(axis='both', which='major', labelsize=20)

        for dimension in range(8):
            ax = axes[dimension + 1]
            st = 0
            ax.plot(actual_df.index, actual_df.iloc[:, st + dimension], label="Observations", color='green')

            ax.plot(forecast.index, mean[:, st + dimension], label="Prediction mean", color='red')

            for interval in intervals:
                low = (1 - interval) / 2
                high = 1 - low
                quantile_low = forecast.quantile(low)[:, st + dimension]
                quantile_high = forecast.quantile(high)[:, st + dimension]
                ax.fill_between(
                    actual_df.index,
                    quantile_low,
                    quantile_high,
                    facecolor='orange',
                    alpha=1 - interval / 1.2,
                    label=f"{int(interval * 100)}% prediction interval"
                )

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            ax.set_title(f"Dimension {dimension + 1}", fontsize=30)
            ax.tick_params(axis='both', which='major', labelsize=20)
            for label in ax.get_xticklabels():
                label.set_rotation(45)

        handles, labels = ax.get_legend_handles_labels()

        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), fontsize=30, ncol=4)

        plt.tight_layout()

        pdf.savefig(fig, bbox_inches='tight')
        plt.show()
        plt.close()

    print(f"Multivariate forecast saved to {pdf_filename}")
