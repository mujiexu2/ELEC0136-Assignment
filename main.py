import os
import csv
import urllib
import json
import math

import pandas as pd
import numpy as np
import datetime
from alpha_vantage.timeseries import TimeSeries
from covid19dh import covid19
import requests
import matplotlib.pyplot as plt
import dataframe_image as dfi
import pickle
import seaborn as sns
from scipy import stats
from matplotlib.ticker import MultipleLocator
import pymongo

import time
from meteostat import Point
from meteostat import Daily

from numpy import sqrt, abs, round
from scipy.stats import norm

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

#-----time selection-------
start_date = datetime.datetime(2017, 4, 1)
end_date = datetime.datetime(2022, 5, 31)

#------api key-------
api_key = "B1Q551T88OEDZJ52"
datasets_dir = "./Datasets"

#----necessary functions------

def metrics(result):
    result = result.dropna()
    mse = mean_squared_error(
        result.loc["2022-05-01":"2022-05-30", "y"],
        result.loc["2022-05-01":"2022-05-30", "yhat"],
    )
    print("mean squared error is: ", mse)
    mae = mean_absolute_error(
        result.loc["2022-05-01":"2022-05-30", "y"],
        result.loc["2022-05-01":"2022-05-30", "yhat"],
    )
    print("mean absolute error is :", mae)
    rmse = np.sqrt(mse)  # mse**(0.5)
    print("root mean squared error is: ", rmse)
    r2 = r2_score(
        result.loc["2022-05-01":"2022-05-30", "y"],
        result.loc["2022-05-01":"2022-05-30", "yhat"],
    )
    print("R-squared is: ", r2)
    return mse, mae, rmse, r2

def save_plot(name, plot):
    fig_dir = "./Figures"
    plt.tight_layout()  # Display appropiately.
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    file = os.path.join(fig_dir, (str(name) + " " + str(plot) + ".jpg"))
    plt.savefig(file)
    plt.clf()  # Clear figure
    plt.cla()  # Clear axes
    plt.close()
    return

def save_file(dir, df, filename):
    if not os.path.exists(dir):
        os.makedirs(dir)
    df.to_pickle(os.path.join(dir, f"{filename}.pkl"))
    df.to_csv(os.path.join(dir, f"{filename}.csv"))
    return


#------Data Acquisition-------
def acquire_AAPL_data():
    ts = TimeSeries(key=api_key, output_format="pandas")
    data_daily, meta_data = ts.get_daily_adjusted(
        symbol="AAPL", outputsize="full")

    # start_date = datetime.datetime(2017, 4, 1)
    # end_date = datetime.datetime(2022, 5, 31)

    # Create a filtered dataframe, and change the order it is displayed.
    stock_data = data_daily[
        (data_daily.index > start_date) & (data_daily.index <= end_date)
    ]
    stock_data = stock_data.sort_index(ascending=True)
    # date_filter
    stock_data = stock_data.rename(
        columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Original Close",
            "5. adjusted close": "Close",
            "6. volume": "Volume",
            "7. dividend amount": "Dividend",
            "8. split coefficient": "Split",
        }
    )
    return stock_data
    
def acquire_GOOG_data():
    ts = TimeSeries(key=api_key, output_format="pandas")
    data_daily, meta_data = ts.get_daily_adjusted(
        symbol="GOOG", outputsize="full")

    # start_date = datetime.datetime(2017, 4, 1)
    # end_date = datetime.datetime(2022, 5, 31)

    # Create a filtered dataframe, and change the order it is displayed.
    stock_data_GOOG = data_daily[
        (data_daily.index > start_date) & (data_daily.index <= end_date)
    ]
    stock_data_GOOG = stock_data_GOOG.sort_index(ascending=True)
    # date_filter
    stock_data_GOOG = stock_data_GOOG.rename(
        columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Original Close",
            "5. adjusted close": "Close",
            "6. volume": "Volume",
            "7. dividend amount": "Dividend",
            "8. split coefficient": "Split",
        }
    )
    return stock_data_GOOG

def acquire_covid_data():
    df_covid, src = covid19("USA", start=start_date,
                            end=end_date, verbose=False)
    return df_covid
    

def acquire_Cupertino_weather():
    location = Point(37.323, -122.032, 70)
    data = Daily(location, start_date, end_date)
    cupertino_weather = data.fetch()
    return cupertino_weather


#--------Data Preprocessing-------
def preprocessing(stock_data_AAPL, stock_data_GOOG, covid_data, weather_data):
    pdates = pd.date_range(start=start_date, end=end_date)
    preprocessed_path = "./preprocessed"

    if not os.path.exists(preprocessed_path):
        os.makedirs(preprocessed_path)

    """stock data APPL preprocessing"""
    # stock_data_AAPL=stock_data_AAPL.drop(['Open','High','Low','Original Close','Volume','Dividend','Split'],axis=1)
    stock_data_AAPL = stock_data_AAPL[
        ["Close"]
    ]  # only Close values of this stock is extracted
    stock_data_AAPL = stock_data_AAPL.reindex(
        pdates, fill_value=np.nan
    )  # interpolate and fill with nan values
    print(stock_data_AAPL)  # dataframe
    save_file(preprocessed_path, stock_data_AAPL, "stock_pre_nan_AAPL")

    """stock data GOOG preprocessing"""
    # stock_data_GOOG=stock_data_GOOG.drop(['Open','High','Low','Original Close','Volume','Dividend','Split'],axis=1)
    stock_data_GOOG = stock_data_GOOG[
        ["Close"]
    ]  # only Close values of this stock is extracted
    stock_data_GOOG = stock_data_GOOG.reindex(pdates, fill_value=np.nan)
    print(stock_data_GOOG)
    save_file(preprocessed_path, stock_data_GOOG, "stock_pre_nan_GOOG")

    """covid data preprocessing"""
    covid_data = covid_data[
        ["date", "confirmed"]
    ]  # only confirmed and date values of this stock are extracted
    covid_pre_filename = "covid_pre"
    covid_data.set_index("date", inplace=True)  # set datetime index

    covid_data["daily_confirmed"] = np.insert(
        np.diff(covid_data.values.flatten()), 0, 0
    )  # change to daily confirmed
    covid_confirmed = covid_data[["confirmed"]]
    covid_confirmed_pre = covid_confirmed.reindex(pdates, fill_value=np.nan)
    covid_daily = covid_data[["daily_confirmed"]]
    covid_daily_pre = covid_daily.reindex(pdates, fill_value=np.nan)
    print(covid_daily)
    print(covid_daily_pre)
    print(covid_confirmed_pre)
    save_file(preprocessed_path, covid_daily_pre, "covid_daily_pre")

    """weather data preprocessing"""
    weather_pre_filename = "weather_pre"
    # weather_data=weather_data.drop(['wpgt','snow','tsun','pres','wdir'],axis=1)
    weather_data = weather_data[
        ["tavg", "tmin", "tmax", "prcp", "wspd"]
    ]  # Only these 5 variables are kept
    weather_data = weather_data.fillna(0)
    print(weather_data)
    save_file(preprocessed_path, weather_data, "weather_data_pre")

    return (
        stock_data_AAPL,
        stock_data_GOOG,
        covid_daily,
        covid_daily_pre,
        covid_confirmed_pre,
        weather_data,
    )

#-------acquire stock data --------------
stock_AAPL = acquire_AAPL_data()
save_file(datasets_dir, stock_AAPL, "stock_AAPL")
stock_AAPL["Close"]

stock_GOOG = acquire_GOOG_data()
save_file(datasets_dir, stock_GOOG, "stock_GOOG")
stock_GOOG

covid_data = acquire_covid_data()
save_file(datasets_dir, covid_data, "Covid")
covid_data

weather_data = acquire_Cupertino_weather()
save_file(datasets_dir, weather_data, "Weather")
#------saving raw data files------------------

#------preprocessing raw data-------
(
    stock_AAPL_pre,
    stock_GOOG_pre,
    covid_daily,
    covid_daily_pre,
    covid_confirmed_pre,
    weather_data_pre,
) = preprocessing(stock_AAPL, stock_GOOG, covid_data, weather_data)
print(covid_daily_pre)
print(covid_daily)
print(covid_confirmed_pre)
print(stock_AAPL_pre)
print(stock_GOOG_pre)
#------preprocessing done--------

#-------data visualization--------------
def line_chart_plot(stock_AAPL, stock_GOOG, covid_daily, weather_data_pre):
    """plot AAPL stock prices"""
    stock_AAPL_pre = stock_AAPL[["Close"]]
    plt.figure(figsize=(16, 8))
    plt.title("APPLE stock prices")
    plt.ylabel("AAPL stock prices")
    plt.xlabel("Datetime")
    plt.plot(stock_AAPL_pre, label="AAPL: Close Price history")
    plt.legend()
    plt.grid()
    save_plot("AAPL stock prices", "line chart")
    # plt.savefig("AAPL stock price line chart.jpg")
    # plt.clf()  # Clear figure
    # plt.cla()  # Clear axes
    # plt.close()

    """plot GOOG stock prices"""
    stock_GOOG_pre = stock_GOOG[["Close"]]
    plt.figure(figsize=(16, 8))
    plt.title("GOOGLE stock prices")
    plt.ylabel("GOOG stock prices")
    plt.xlabel("Datetime")
    plt.plot(stock_GOOG_pre, label="GOOG: Close Price history")
    plt.legend()
    plt.grid()
    save_plot("GOOG stock prices", "line chart")
    # plt.savefig("GOOG stock price line chart.jpg")
    # plt.clf()  # Clear figure
    # plt.cla()  # Clear axes
    # plt.close()

    """plot covid data"""
    plt.figure(figsize=(16, 8))
    plt.title("Covid Data")
    plt.ylabel("People Get Confirmed--daily")
    plt.xlabel("Datetime")
    plt.plot(covid_daily, label="daily confirmed")
    plt.legend()
    plt.grid()
    save_plot("Covid data", "line chart")
    # plt.savefig("covid data line chart.jpg")
    # plt.clf()  # Clear figure
    # plt.cla()  # Clear axes
    # plt.close()

    """Plot Weather Data"""
    plt.figure(figsize=(8, 10))
    plt.subplot(5, 1, 1)
    plt.suptitle("Weather Data")
    plt.plot(weather_data_pre[["tavg"]], linestyle="-", marker="", c="b")
    plt.ylabel("tavg")
    plt.subplot(5, 1, 2)
    plt.plot(weather_data_pre[["tmin"]], linestyle="-", marker="", c="b")
    plt.ylabel("tmin")
    plt.subplot(5, 1, 3)
    plt.plot(weather_data_pre[["tmax"]], linestyle="-", marker="", c="b")
    plt.ylabel("tmax")
    plt.subplot(5, 1, 4)
    plt.plot(weather_data_pre[["prcp"]], linestyle="-", marker="", c="b")
    plt.ylabel("prcp")
    plt.subplot(5, 1, 5)
    plt.plot(weather_data_pre[["wspd"]], linestyle="-", marker="", c="b")
    plt.ylabel("wspd")
    save_plot("Weather data", "line chart")
    # plt.savefig("weather data line chart.jpg")
    # plt.clf()  # Clear figure
    # plt.cla()  # Clear axes
    # plt.close()
    return

line_chart_plot(stock_AAPL, stock_GOOG, covid_daily, weather_data_pre)

#-------zplot----------
def z_score_plot(covid_daily, stock_AAPL, stock_GOOG):
    """Covid z-score"""
    covid_np = covid_daily[["daily_confirmed"]].to_numpy().flatten()
    covid_time = covid_daily.index.values
    z_covid = np.abs(stats.zscore(covid_np))
    print("z score of the (filled) covid dataset is:\r\n", z_covid)
    plt.figure(figsize=(16, 7))
    plt.plot(covid_time, z_covid)
    plt.grid()
    plt.ylabel("Z score")
    plt.xlabel("Datetime")
    plt.title("Covid daily confirmed-Z score")
    save_plot("Covid_daily", "Z-score")

    """stock AAPL"""
    AAPL_np = stock_AAPL[["Close"]].to_numpy().flatten()
    AAPL_time = stock_AAPL.index.values
    z_AAPL = np.abs(stats.zscore(AAPL_np))
    print("z score of the AAPL dataset is:\r\n", z_AAPL)
    plt.figure(figsize=(16, 7))
    plt.plot(AAPL_time, z_AAPL)
    plt.grid()
    plt.ylabel("Z score")
    plt.xlabel("Datetime")
    plt.title("stock AAPL -Z score")
    save_plot("Stock AAPL", "Z-score")

    """stock GOOG"""
    GOOG_np = stock_GOOG[["Close"]].to_numpy().flatten()
    GOOG_time = stock_GOOG.index.values
    z_GOOG = np.abs(stats.zscore(GOOG_np))
    print("z score of the GOOG dataset is:\r\n", z_GOOG)
    plt.figure(figsize=(16, 7))
    plt.plot(GOOG_time, z_GOOG)
    plt.grid()
    plt.ylabel("Z score")
    plt.xlabel("Datetime")
    plt.title("stock GOOG-Z score")
    save_plot("Stock GOOG", "Z-score")

    return z_covid, z_AAPL, z_GOOG

z_covid, z_AAPL, z_GOOG = z_score_plot(covid_daily, stock_AAPL, stock_GOOG)

#----scatter plot---------
def scatter_plot_covid(covid_daily):
    # set a threshold and find the location where the value meets our condition(s)

    covid_np = covid_daily[["daily_confirmed"]].to_numpy().flatten()
    covid_time = covid_daily.index.values

    threshold = 6
    covid_outlier_loc = np.where(z_covid > threshold)
    covid_time = covid_daily.index.values
    # find the outlier value given its index
    covid_outlier_by_Z_Score = z_covid[covid_outlier_loc]
    print("the data classified as outlier by z score:\r\n",
          covid_outlier_by_Z_Score)
    print("the datetime of the outlier is:\r\n", covid_time[covid_outlier_loc])

    plt.figure(figsize=(16, 7))
    plt.scatter(covid_time, covid_np)
    plt.xlabel("Date")
    plt.ylabel("Covid")
    plt.grid()
    save_plot("Covid data", "Scatter Plot")

    capped_outlier_dataset = np.copy(covid_np)
    capped_outlier_dataset[covid_outlier_loc] = np.mean(covid_np)

    # Plot and compare, before and after the outlier is capped.
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 5))
    ax1.set_title("Before Cap")
    ax1.scatter(covid_time, covid_np)
    ax1.scatter(covid_time[covid_outlier_loc],
                covid_np[covid_outlier_loc], c="r")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("People Confirmed")
    # Spacing of axis tickers.
    ax1.xaxis.set_major_locator(MultipleLocator(465))

    ax2.set_title("After Cap")
    ax2.scatter(covid_time, capped_outlier_dataset)
    ax2.scatter(
        covid_time[covid_outlier_loc], capped_outlier_dataset[covid_outlier_loc], c="r"
    )
    ax2.set_xlabel("Date")
    # Spacing of axis tickers.
    ax2.xaxis.set_major_locator(MultipleLocator(465))

    save_plot("Covid Data", "Outlier Cap")
    return

scatter_plot_covid(covid_daily)

covid_np = covid_daily[["daily_confirmed"]].to_numpy().flatten()
covid_time = covid_daily.index.values

def capped_outlier(covid_np):
    capped_outlier_dataset = np.copy(covid_np)
    capped_outlier_dataset[covid_outlier_loc] = np.mean(covid_np)

    # Plot and compare, before and after the outlier is capped.
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 5))
    ax1.set_title("Before Cap")
    ax1.scatter(covid_time, covid_np)
    ax1.scatter(covid_time[covid_outlier_loc],
                covid_np[covid_outlier_loc], c="r")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("People Confirmed")
    # Spacing of axis tickers.
    ax1.xaxis.set_major_locator(MultipleLocator(465))

    ax2.set_title("After Cap")
    ax2.scatter(covid_time, capped_outlier_dataset)
    ax2.scatter(
        covid_time[covid_outlier_loc], capped_outlier_dataset[covid_outlier_loc], c="r"
    )
    ax2.set_xlabel("Date")
    # Spacing of axis tickers.
    ax2.xaxis.set_major_locator(MultipleLocator(465))

    save_plot("Covid Data", "Outlier Cap")

#----box plot-------
def box_plot(stock_AAPL, covid_daily, stock_GOOG):
    AAPL_np = stock_AAPL[["Close"]].to_numpy().flatten()
    # AAPL_time=stock_AAPL_pre.index.values
    sns.boxplot(AAPL_np)
    plt.ylabel(r"AAPL Stock Price")
    save_plot("AAPL stock price", "box plot")

    covid_np = covid_daily[["daily_confirmed"]].to_numpy().flatten()
    # covid_time=covid_daily.index.values
    sns.boxplot(covid_np)
    plt.ylabel(r"Covid Data")
    save_plot("Covid Data", "box plot")

    GOOG_np = stock_GOOG[["Close"]].to_numpy().flatten()
    sns.boxplot(GOOG_np)
    plt.ylabel(r"GOOG Stock Price")
    save_plot("GOOG stock price", "box plot")
    return

box_plot(stock_AAPL_pre, covid_daily, stock_GOOG_pre)

#---merging----
def merge(stock_AAPL_pre, stock_GOOG_pre):
    stock_AAPL_merge = stock_AAPL_pre.rename(columns={"Close": "AAPL Close"})
    stock_GOOG_merge = stock_GOOG_pre.rename(columns={"Close": "GOOG Close"})
    pd_merged = pd.concat(
        [
            stock_AAPL_merge,
            stock_GOOG_merge,
            covid_confirmed_pre,
            covid_daily_pre,
            weather_data_pre,
        ],
        axis=1,
        join="outer",
    )

    norm_aux = pd.concat(
        [stock_GOOG_merge, covid_confirmed_pre, covid_daily_pre, weather_data_pre],
        axis=1,
        join="outer",
    )
    return pd_merged, norm_aux, stock_AAPL_merge, stock_GOOG_merge

pd_merged, norm_aux, stock_AAPL_merge, stock_GOOG_merge = merge(
    stock_AAPL_pre, stock_GOOG_pre
)

#---------Data Exploration------------
# normalize the auxiliary data
norm_aux = (norm_aux - norm_aux.mean()) / (norm_aux.std())
norm_aux

def heatmap_plot(pd_merged):
    plt.figure(dpi=250)
    sns.heatmap(
        pd_merged.corr(),
        cmap=sns.diverging_palette(20, 220, n=200),
        annot=True,
        fmt=".2f",
        center=0,
    )
    plt.tight_layout()
    save_plot("Merged Data", "Heat Map")

heatmap_plot(pd_merged)

#-----seasonality/trend/random noise components--------
import statsmodels.api as sm

decomposition = sm.tsa.seasonal_decompose(
    stock_AAPL["Close"], model="multiplicate", period=7
)
decomposition.plot()
save_plot("AAPL Close stock price-period=7", "Seasonal, Trend, Resid")

#------Hypothesis Test---------
from numpy import sqrt, abs, round
from scipy.stats import norm

def stat_cal(var):
    var_max = var.max()
    var_std = var.std()
    var_min = var.min()
    var_mean = var.mean()
    var_median = var.median()
    var_size = var.shape[0]
    return var_max, var_std, var_min, var_mean, var_median, var_size

def two_samples_t_test(mean1, mean2, std1, std2, size1, size2, significance_level=0.05):
    """
    Executes a two sample T-test with the statistic properties passed.
    mean1: mean of the first sample.
    mean2: mean of the second sample.
    std1: standard deviation of the first sample.
    std2: standard deviation of the second sample.
    size1: size of the first sample.
    size2: size of the second sample.
    the default value of the significance_level = 0.05
    """
    overall_std = sqrt(std1**2 / size1 + std2**2 / size2)
    z_statistic = round((mean1 - mean2) / overall_std, 5)
    # Two tails -> H0:x1=x2 H1:x1!=x2
    print("Two Tails test. H0 is x1=x2 and H1 is x1!=x2")
    p_value = round(2 * (1 - norm.cdf(abs(z_statistic))), 5)
    # Reject or not the Null hypothesis
    if p_value < significance_level:
        print(
            f"Statistic:{z_statistic} - P-value:{p_value} - Reject Null Hypothesis")
    else:
        print(
            f"Statistic:{z_statistic} - P-value:{p_value} - Do Not Reject Null Hypothesis"
        )

# Google

goog_max, goog_std, goog_min, goog_mean, goog_median, goog_size = stat_cal(
    stock_GOOG["Close"]
)
aapl_max, aapl_std, aapl_min, aapl_mean, aapl_median, aapl_size = stat_cal(
    stock_AAPL["Close"]
)
two_samples_t_test(goog_mean, aapl_mean, goog_std,
                   aapl_std, goog_size, aapl_size)

covid_max, covid_std, covid_min, covid_mean, covid_median, covid_size = stat_cal(
    stock_GOOG["Close"]
)
aapl_max, aapl_std, aapl_min, aapl_mean, aapl_median, aapl_size = stat_cal(
    covid_data["confirmed"]
)
two_samples_t_test(covid_mean, aapl_mean, covid_std,
                   aapl_std, covid_size, aapl_size)

#------store on MongodB---------
client = pymongo.MongoClient(
    "mongodb+srv://xmj_jessie:Xmj010928@cluster0.ohtke6p.mongodb.net/?retryWrites=true&w=majority"
)
db = client.admin
serverStatusResult = db.command("serverStatus")
# print(serverStatusResult)
db = client["daps_final"]
mycol = db["Apple Stock Forecasting"]
x = mycol.insert_many(json.loads(pd_merged.T.to_json()).values())

#----Model Inference------------
from fbprophet import Prophet
def prepare_data(data, target_feature):
    """
    prepare the data for ingestion by fbprophet:
    see: https://facebook.github.io/prophet/docs/quick_start.html
    """
    new_data = data.copy()
    new_data.reset_index(inplace=True)
    new_data = new_data.rename(
        {"index": "ds", "{}".format(target_feature): "y"}, axis=1
    )

    return new_data

stock_aapl = prepare_data(data=stock_AAPL_pre, target_feature="Close")
stock_aapl

#-----train test split and model fit-------
def train_test_split(data):

    train = data.set_index("ds").loc[:"2022-04-30", :].reset_index()
    test = data.set_index("ds").loc["2022-05-01":, :].reset_index()

    return train, test

train, test = train_test_split(data=stock_aapl)

m = Prophet(
    seasonality_mode="multiplicative",
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
)

m.fit(train)

future = m.make_future_dataframe(periods=30, freq="1D")
#future
forecast = m.predict(future)
#forecast

f = m.plot_components(forecast, figsize=(12, 16))
f.savefig("Multiplicative forecasting-with Prophet-2")
f.clf()

def make_predictions_df(forecast, data_train, data_test):
    """
    Function to convert the output Prophet dataframe to a datetime index and append the actual target values at the end
    """
    forecast.index = pd.to_datetime(forecast.ds)
    data_train.index = pd.to_datetime(data_train.ds)
    data_test.index = pd.to_datetime(data_test.ds)
    data = pd.concat([data_train, data_test], axis=0)
    forecast.loc[:, "y"] = data.loc[:, "y"]

    return forecast


def plot_predictions(forecast, start_date):
    """
    Function to plot the predictions
    """
    f, ax = plt.subplots(figsize=(14, 8))

    train = forecast.loc[start_date:"2022-05-01", :]
    ax.plot(train.index, train.y, "ko", markersize=3)
    ax.plot(train.index, train.yhat, color="steelblue", lw=0.5)
    ax.fill_between(
        train.index, train.yhat_lower, train.yhat_upper, color="steelblue", alpha=0.3
    )

    test = forecast.loc["2022-05-01":, :]
    ax.plot(test.index, test.y, "ro", markersize=3)
    ax.plot(test.index, test.yhat, color="coral", lw=0.5)
    ax.fill_between(
        test.index, test.yhat_lower, test.yhat_upper, color="coral", alpha=0.3
    )
    ax.axvline(forecast.loc["2022-05-01", "ds"], color="k", ls="--", alpha=0.7)

    ax.grid(ls=":", lw=0.5)

    return f, ax

result = make_predictions_df(forecast, train, test)
result.loc[:, "yhat"] = result.yhat.clip(lower=0)
result.loc[:, "yhat_lower"] = result.yhat_lower.clip(lower=0)
result.loc[:, "yhat_upper"] = result.yhat_upper.clip(lower=0)
#result

f, ax = plot_predictions(result, "2020-04-01")
f.savefig("Multiplicative forecasting-with Prophet-20200401")
f.clf()

def metrics(result):
    result = result.dropna()
    mse = mean_squared_error(
        result.loc["2022-05-01":"2022-05-30", "y"],
        result.loc["2022-05-01":"2022-05-30", "yhat"],
    )
    print("mean squared error is: ", mse)
    mae = mean_absolute_error(
        result.loc["2022-05-01":"2022-05-30", "y"],
        result.loc["2022-05-01":"2022-05-30", "yhat"],
    )
    print("mean absolute error is :", mae)
    rmse = np.sqrt(mse)  # mse**(0.5)
    print("root mean squared error is: ", rmse)
    r2 = r2_score(
        result.loc["2022-05-01":"2022-05-30", "y"],
        result.loc["2022-05-01":"2022-05-30", "yhat"],
    )
    print("R-squared is: ", r2)
    return mse, mae, rmse, r2
    
mse, mae, rmse, r2 = metrics(result)

#------extra regressors are added----------
m = Prophet(
    seasonality_mode="multiplicative",
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
)

m.add_regressor("confirmed", mode="multiplicative")
m.add_regressor("GOOG Close", mode="multiplicative")

pd_merged_2 = pd.concat(
    [stock_AAPL_merge, stock_GOOG_merge, covid_confirmed_pre],
    axis=1,
    join="outer",
)
pd_merged_2 = pd_merged_2.reset_index()
pd_merged_2 = pd_merged_2.rename(columns={"index": "ds", "AAPL Close": "y"})
pd_merged_2 = pd_merged_2.fillna(0)
train_extra, test_extra = train_test_split(pd_merged_2)
#train_extra

m.fit(train_extra)

future = m.make_future_dataframe(periods=len(test_extra), freq="1D")
futures = pd_merged_2.drop(["y"], axis=1)
forecast = m.predict(futures)

result = make_predictions_df(forecast, train_extra, test_extra)
result.loc[:, "yhat"] = result.yhat.clip(lower=0)
result.loc[:, "yhat_lower"] = result.yhat_lower.clip(lower=0)
result.loc[:, "yhat_upper"] = result.yhat_upper.clip(lower=0)

f, ax = plot_predictions(result, "2021-04-01")
f.savefig("Multiplicative forecasting-with Prophet&AUX-4")
f.clf()

mse, mae, rmse, r2 = metrics(result)










