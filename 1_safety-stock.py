# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/safety-stock. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/safety-stock.

# COMMAND ----------

# MAGIC %md ##Safety Stock Calculations for Inventory Management
# MAGIC 
# MAGIC Periodically, we need to order product to replenish our inventory. When we do this, we have in mind a future period for which we are attempting to address demand along with an estimate the demand in that period.
# MAGIC 
# MAGIC Our estimates are never perfectly accurate.  Instead, when we use modern forecasting techniques, we are estimating the *mean* demand across a range of potential demand values. Improvements in model accuracy narrow the range of variability around this predicted mean, but we still expect to be below the mean 50% of the time and above it the other 50% (as that's simply the nature of a mean value). 
# MAGIC 
# MAGIC When actual demand exceeds our forecasts, we run the risk of a stockout (out of stock) situation with its associated potential loss of sales and reduced customer satisfaction. To avoid this, we often include additional units of stock, above the forecasted demand, in our replenishment  orders. The amount of this *safety stock* depends on our estimates of variability in the demand for this upcoming period and the percentage of time we are willing to risk an out of stock situation.  
# MAGIC 
# MAGIC For a more in-depth examination of safety stock calculations, please refer to the [blog](https://www.databricks.com/blog/2020/04/22/how-a-fresh-approach-to-safety-stock-analysis-can-optimize-inventory.html) associated with this notebook.  The purpose of the information provided here is to examine how forecast data can be employed to effectively calculate safety stock requirements leveraging standard formulas.

# COMMAND ----------

# MAGIC %md ###Step 1: Generate the Forecasts
# MAGIC 
# MAGIC Using the [Facebook Prophet](https://facebook.github.io/prophet/) library and an anonymized but [real-world dataset](https://www.kaggle.com/c/demand-forecasting-kernels-only/data) consisting of 5 years of daily sales data for 50 items sold across 10 stores, we will generate store-item specific forecasts.  The details on how this data should be loaded into this environment and how the 500 models will be trained in a timely manner in order to generate the required forecasts are addressed in the notebook associated with [this blog post](https://databricks.com/blog/2020/01/27/time-series-forecasting-prophet-spark.html). We strongly suggest you review that blog post and its associated notebook before proceeding.
# MAGIC 
# MAGIC Differing from the patterns established in that notebook, we will incorporate holiday event information into our forecast.  The use of holiday event data along with external regressors was addressed in [another blog post](https://databricks.com/blog/2020/03/26/new-methods-for-improving-supply-chain-demand-forecasting.html) that may be worth reviewing should you have questions about that aspect of this work.
# MAGIC 
# MAGIC Given that the bulk of the code in this section is examined in more detail in the previously referenced materials, we will proceed with environment setup and forecast generation with minimal additional explanation before diving into the safety stock calculations that build upon the output of these steps: 

# COMMAND ----------

# MAGIC %run "./config/Data Extract"

# COMMAND ----------

# import the required libraries ignore the error message 
# "Importing plotly failed. Interactive plots will not work."

import numpy as np
import pandas as pd

import holidays

from pyspark.sql.types import *
from pyspark.sql.functions import lit, min, max, pandas_udf, PandasUDFType

import mlflow
import mlflow.sklearn
import shutil

from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

import logging
logging.getLogger('py4j').setLevel(logging.ERROR)

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# COMMAND ----------

# load the historical dataset

history_schema = StructType([
  StructField('date', TimestampType()),
  StructField('store', IntegerType()),
  StructField('item', IntegerType()),
  StructField('sales', IntegerType())
  ])

history = spark.read.csv(
  '/tmp/solacc/demand_forecast/train/train.csv', 
  header=True, 
  schema=history_schema
  ).cache()

history.createOrReplaceTempView('history_tmp')

# COMMAND ----------

# generate holiday information for years in the historical dataset and 
# the following year should forecasts extend into that year

first_date, last_date = (
    history
      .agg(
        min('date').alias('first_date'),
        max('date').alias('last_date')
        )
    ).collect()[0]

holidays_df = (
  spark.createDataFrame(
    holidays.UnitedStates(years=range(first_date.year, 2 + last_date.year)).items(), 
    ['date','holiday']
    )
    .orderBy('date')
  )

holidays_pd = holidays_df.toPandas()
holidays_broadcast = sc.broadcast(holidays_pd)

# COMMAND ----------

# define function to generate forecasts

def get_forecast(keys, history_pd):
  
  # read keys associated with grouped data
  store = keys[0]
  item = keys[1]
  days_to_forecast = keys[2]
  
  # prepare grouped data for training
  history_pd = history_pd.dropna()
  history_pd.rename(columns={'date':'ds', 'sales':'y'}, inplace=True)
  
  # acquire holiday dataset
  holidays_pd=holidays_broadcast.value
  holidays_pd.rename(columns={'date':'ds'}, inplace=True)
  
  # instantiate and configure prophet model
  model = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative',
    holidays=holidays_pd
  ) 
  
  # train the model
  model.fit(history_pd[['ds','y']])
  
  # save fitted model for later utilization (during evaluation)
  model_path = '/dbfs/tmp/forecast/{0}_{1}'.format(store, item) #TODO: dbfs:
  shutil.rmtree(model_path, ignore_errors=True)
  mlflow.sklearn.save_model( model, model_path)
  
  # make forecast dataset
  future_pd = model.make_future_dataframe(
    periods=days_to_forecast, 
    freq='d', 
    include_history=True
    )
  
  # generate forecast
  forecast_pd = model.predict( future_pd )
  
  # merge history and forecast datasets
  forecast_pd = forecast_pd.merge(history_pd, on='ds', how='left')
  
  # assign store and item to results
  forecast_pd['store']=store
  forecast_pd['item']=item
  
  # return results
  forecast_pd.rename(
    columns={
      'ds':'date', 'y':'sales', 
      'yhat':'sales_pred_mean', 'yhat_lower':'sales_pred_lower', 'yhat_upper':'sales_pred_upper'
      }, inplace=True
    )
  return forecast_pd[ ['store', 'item', 'date', 'sales', 'sales_pred_mean', 'sales_pred_lower', 'sales_pred_upper'] ]

# COMMAND ----------

# define function to generate forecast-wide evaluation metrics

def evaluate_forecast( keys, forecast_pd ):
  
  # read keys associated with grouped data
  forecast_date = keys[0]
  store = int(keys[1])
  item = int(keys[2])
  
  # calculate MSE & RMSE
  mse = mean_squared_error( forecast_pd['sales'], forecast_pd['sales_pred_mean'] )
  rmse = sqrt(mse)
  
  # calculate MAE
  mae = mean_absolute_error(forecast_pd['sales'], forecast_pd['sales_pred_mean'])
  
  # calculate MAPE 
  mape = np.mean(np.abs((forecast_pd['sales'] - forecast_pd['sales_pred_mean']) /forecast_pd['sales'])) * 100
  
  # assemble result set
  results = {'forecast_date':[forecast_date], 'store':[store], 'item':[item], 'mse':[mse], 'rmse':[rmse], 'mae':[mae], 'mape':[mape]}
  return pd.DataFrame( data=results )

# COMMAND ----------

# MAGIC %md Everything up to this point is a minor variation of code explained in the previously referenced blog posts and their associated notebooks.  However, this next cell introduces something a bit new: FBProphet's [model cross-validation](https://facebook.github.io/prophet/docs/diagnostics.html).
# MAGIC 
# MAGIC The structure of the user-defined function is not terribly different from that of the previously defined functions.  However, its goal is to calculate evaluation metrics using a cross-validation technique.  This will require the model trained in the *get_forecast* function to be retrieved.  In that function, we used mlflow to persist the model to a known location. In this function, we use mlflow to retrieve that model from that location for further evaluation. 
# MAGIC (Please note that saving a larger number of models to */tmp* in the Databricks workspace is not recommended. Instead, use a [cloud storage container]((https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs) mounted to your Databricks environment to provide you access to greater storage capacity and better I/O performance.)
# MAGIC 
# MAGIC With the trained model in hand, we call the cross_validation function, asking it to consider the initial three years (1,095 days) of the 5 years of data used to train the original model.  Leveraging insights gathered from this initial period, the model is used to generate a forecast for a specified number of days leading up to a maximum number of days representing the forecast horizon. This work is repeated over and over as the initial dataset is moved forward in time a number of days equal to half of the horizon duration until the end of the historical dataset on which the model was originally trained is reached.
# MAGIC 
# MAGIC The end result of this work is a dataframe within which the actual and forecasted values so many days out from the cutoff of the training data set are known.  This dataset is then passed to the performance_metrics function to calculate standard error metrics for the model for each day moving towards the specified horizon.  With these metrics, we can evaluate our model performs as we predict further out from the end of the historical dataset on which it was trained:

# COMMAND ----------

# define function to generate forecast cross-validation evaluation metrics

def evaluate_forecast_cv( keys, forecast_pd ):
  
  # read keys associated with grouped data
  forecast_date = keys[0]
  store = int(keys[1])
  item = int(keys[2])
  days_to_forecast = int(keys[3])
  
  # retrieve trained model
  model_path = '/dbfs/tmp/forecast/{0}_{1}'.format(store, item)
  model = mlflow.sklearn.load_model(model_path)
  
  # calculate cv performance metrics
  crossval_pd = cross_validation(
    model, 
    initial='1095 days',
    horizon='{0} days'.format(days_to_forecast)
    )
  perf_pd = performance_metrics(
    crossval_pd, 
    metrics=['mse', 'rmse', 'mae', 'mape', 'mdape', 'coverage']
    )  
  
  # prepare results
  perf_pd['forecast_date'] = forecast_date
  perf_pd['store'] = store
  perf_pd['item'] = item
  perf_pd['horizon'] = perf_pd['horizon'].apply(lambda h: h.days)
  
  # return metrics
  return perf_pd[['forecast_date', 'store', 'item', 'horizon', 'mse', 'rmse', 'mae', 'mape', 'mdape', 'coverage']]

# COMMAND ----------

# MAGIC %md With all the required elements in place, we can now generate our forecasts and evaluation metrics. To assist us in evaluating the performance of our safety stock calculations, we will limit forecasting to a period towards the end of the dataset for which we will have historical data available to compare to our forecasted values.  As our dataset terminates at December 31, 2017, we will generate a 30-day forecast from December 1, 2017:

# COMMAND ----------

forecast_date = '2017-12-01'
spark.conf.set('spark.sql.shuffle.partitions', 500 ) 

# generate forecast for this data
forecasts = (
  history
  .where(history.date < forecast_date) # limit training data to prior to our forecast date
  .groupBy('store', 'item', lit(30).alias('days_to_forecast'))
    .applyInPandas(get_forecast, "store integer, item integer, date timestamp, sales float, sales_pred_mean float, sales_pred_lower float, sales_pred_upper float")
    .withColumn('forecast_date', lit(forecast_date).cast(TimestampType())) 
    ).cache()

forecast_evals = (
  forecasts      
    .select('forecast_date', 'store', 'item', 'sales', 'sales_pred_mean')
    .where(forecasts.date < forecasts.forecast_date)
    .groupBy('forecast_date', 'store', 'item')
    .applyInPandas(evaluate_forecast, "forecast_date timestamp, store integer, item integer, mse float, rmse float, mae float, mape float")
    )

forecast_evals_cv = (
  forecasts      
    .select('forecast_date', 'store', 'item', 'sales', 'sales_pred_mean')
    .where(forecasts.date < forecasts.forecast_date)
    .groupBy('forecast_date', 'store', 'item', lit(30).alias('days_to_forecast'))
    .applyInPandas(evaluate_forecast_cv, "forecast_date timestamp, store integer, item integer, horizon integer, mse float, rmse float, mae float, mape float, mdape float, coverage float")
    )

forecasts.createOrReplaceTempView('forecasts_tmp')
forecast_evals.createOrReplaceTempView('forecast_evals_tmp')
forecast_evals_cv.createOrReplaceTempView('forecast_evals_cv_tmp')

# COMMAND ----------

# MAGIC %sql -- persist forecasts for later use
# MAGIC -- the join to historical ensures we have
# MAGIC -- historical data married to forecasts 
# MAGIC -- across december 2017
# MAGIC CREATE DATABASE IF NOT EXISTS solacc_safety_stock;
# MAGIC USE solacc_safety_stock;
# MAGIC DROP TABLE IF EXISTS forecasts;
# MAGIC 
# MAGIC CREATE TABLE forecasts
# MAGIC USING delta
# MAGIC AS 
# MAGIC   SELECT 
# MAGIC     a.forecast_date,
# MAGIC     a.store, 
# MAGIC     a.item,
# MAGIC     a.date,
# MAGIC     b.sales,
# MAGIC     a.sales_pred_mean,
# MAGIC     a.sales_pred_lower,
# MAGIC     a.sales_pred_upper
# MAGIC   FROM forecasts_tmp a
# MAGIC   INNER JOIN history_tmp b
# MAGIC     ON a.store=b.store AND a.item=b.item AND a.date=b.date;

# COMMAND ----------

# MAGIC %sql -- persist forecast metrics for later use
# MAGIC 
# MAGIC DROP TABLE IF EXISTS forecast_evals;
# MAGIC 
# MAGIC CREATE TABLE forecast_evals
# MAGIC USING delta
# MAGIC AS SELECT * FROM forecast_evals_tmp;

# COMMAND ----------

# MAGIC %sql -- persist cross-validation metrics for later use
# MAGIC 
# MAGIC -- NOTE this step is time-intensive
# MAGIC 
# MAGIC DROP TABLE IF EXISTS forecast_evals_cv;
# MAGIC 
# MAGIC CREATE TABLE forecast_evals_cv
# MAGIC USING delta
# MAGIC AS SELECT * FROM forecast_evals_cv_tmp;

# COMMAND ----------

# remove cached objects from memory to free up resources

history.unpersist()
forecasts.unpersist()
holidays_broadcast.unpersist()

# COMMAND ----------

# MAGIC %md ###Step 2: Perform Safety Stock Calculations
# MAGIC 
# MAGIC We now have access to forecasted and actual demand for all 500 store-item combinations. Let's quickly visualize forecasted and actual demand for the combination of item 1 in store 1.  We will limit the visualization to data in calendar year 2017 for ease of interpretation:

# COMMAND ----------

# MAGIC %sql -- actuals vs. forecast for store 1/item 1 in 2017
# MAGIC 
# MAGIC SELECT
# MAGIC   date,
# MAGIC   sales,
# MAGIC   sales_pred_mean
# MAGIC FROM forecasts a
# MAGIC WHERE 
# MAGIC   store=1 AND 
# MAGIC   item=1 AND
# MAGIC   YEAR(date)= 2017;

# COMMAND ----------

# MAGIC %md As explained at the top of the notebook, the forecast is not expected to perfectly predict demand.  Instead, it provides a mean estimate around which actual demand varies. Some days we will be below the forecasted demand and some days we will below it.  We can see this in summary statistics calculated for each of our 500 store-item forecasts:

# COMMAND ----------

# MAGIC %sql -- percent days below and above forecast by model
# MAGIC 
# MAGIC SELECT
# MAGIC   store,
# MAGIC   item,
# MAGIC   COUNT(*) as dates_total,
# MAGIC   SUM(CASE WHEN sales < sales_pred_mean THEN 1 ELSE 0 END) as dates_below_forecast,
# MAGIC   SUM(CASE WHEN sales > sales_pred_mean THEN 1 ELSE 0 END) as dates_above_forecast,
# MAGIC   FORMAT_NUMBER(SUM(CASE WHEN sales < sales_pred_mean THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as dates_below_forecast_pct,
# MAGIC   FORMAT_NUMBER(SUM(CASE WHEN sales > sales_pred_mean THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as dates_above_forecast_pct
# MAGIC FROM forecasts
# MAGIC GROUP BY store, item
# MAGIC ORDER BY store, item;

# COMMAND ----------

# MAGIC %md Given the variability around our forecast, we can now explore how we might place orders to replenish inventory and avoid frequent stockouts.  To do this, let's consider a simple hypothetical scenario within which each store in our dataset places a replenishment order to its distributor on Sunday evening at the close of business.  The products ordered are expected to arrive from the distributor Wednesday afternoon and made available for sale to customers by the opening of business on Thursday.  The units ordered are expected to last until the close of business hours on Wednesday the following week. At that time, product units from the next replenishment order should become available.
# MAGIC 
# MAGIC This scenario gives us a *performance cycle* of 7 days (from Thursday to Wednesday) and a *lead time* of 3 days (Monday to Wednesday).  While individual stores typically face some measure of variability and uncertainty around lead times, we will assume no lead time variability or uncertainty in this scenario in order to keep focused on the variability in demand.
# MAGIC 
# MAGIC With this scenario in mind, we can align each date with a particular inventory cycle and the replenishment order date that affects its stocking levels:

# COMMAND ----------

# MAGIC %sql -- calculate the cycle with which each date is associated
# MAGIC 
# MAGIC SELECT
# MAGIC   store,
# MAGIC   item,
# MAGIC   date,
# MAGIC   next_day(date_add(date, -7), 'thursday') as cycle_start,
# MAGIC   next_day(date_add(date, -1), 'wednesday') as cycle_end,
# MAGIC   date_add(next_day(date_add(date, -7), 'thursday'), -4) as order_date
# MAGIC FROM forecasts
# MAGIC ORDER BY store, item, date

# COMMAND ----------

# MAGIC %md On Sunday evening, just prior to orders being placed, we can sum the total demand expected for the 7-day period that represents the upcoming Thursday through Wednesday performance cycle. This quantity is known as our *cycle stock* and represents the quantity of product needed to address mean demand for the period:
# MAGIC 
# MAGIC NOTE: We will exclude cycles that span the head and tail of our dataset from further analysis to focus on cycles with which complete historical datasets are available.

# COMMAND ----------

# MAGIC %sql -- calculate cycle_stock
# MAGIC 
# MAGIC SELECT
# MAGIC   x.store,
# MAGIC   x.item,
# MAGIC   y.cycle_start,
# MAGIC   y.cycle_end,
# MAGIC   SUM(x.sales_pred_mean) as cycle_stock
# MAGIC FROM forecasts x
# MAGIC INNER JOIN (
# MAGIC   SELECT DISTINCT -- cycles
# MAGIC     store,
# MAGIC     item,
# MAGIC     --date,
# MAGIC     next_day(date_add(date, -7), 'thursday') as cycle_start,
# MAGIC     next_day(date_add(date, -1), 'wednesday') as cycle_end
# MAGIC   FROM forecasts
# MAGIC   ) y 
# MAGIC   ON 
# MAGIC     x.store=y.store AND 
# MAGIC     x.item=y.item AND 
# MAGIC     x.date BETWEEN y.cycle_start AND y.cycle_end
# MAGIC WHERE YEAR(y.cycle_start) >= 2013 AND YEAR(y.cycle_end) <= 2017  -- only use cycles completely within historical dataset
# MAGIC GROUP BY
# MAGIC   x.store,
# MAGIC   x.item,
# MAGIC   y.cycle_start,
# MAGIC   y.cycle_end
# MAGIC ORDER BY store, item, cycle_start

# COMMAND ----------

# MAGIC %md With total predicted mean demand for each cycle known, we now can turn our attention to variability.  Because we have historical data for all our predicted periods, we will **cheat** and use the actual, historical data for each cycle to derive cycle-specific demand variability.  This is a completely bogus way to do our calculations as we would never have actual demand variability for a future period of time, but we can use this here to explore how safety stock is calculated and verify that the safety stock formula would allow us to meet our service level expectations.
# MAGIC 
# MAGIC We will calculate safety stock as *(the number of days in the performance cycle)* *x* *(the standard deviation of the daily demand for that period)* *x* *(the z-score aligned with our service level expectation)*. If we set our service level expectation to 95%, a commonly employed service level expectation, our *z-score* is 1.6449. Given we know our cycle stock, we can add our safety stock to our cycle stock to determine the number of product units required to meet demand over the coming period given our service level goal:

# COMMAND ----------

# MAGIC %sql -- calculate safety stock
# MAGIC 
# MAGIC SELECT
# MAGIC   p.store,
# MAGIC   p.item,
# MAGIC   p.cycle_start,
# MAGIC   p.cycle_stock,
# MAGIC   p.zscore * SQRT(p.cycle_days) * p.cycle_stddev as safety_stock,
# MAGIC   p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.cycle_stddev) as required_stock,
# MAGIC   p.cycle_sales
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     a.store,
# MAGIC     a.item,
# MAGIC     a.cycle_start,
# MAGIC     DATEDIFF(a.cycle_end, a.cycle_start)+1 as cycle_days,
# MAGIC     1.6449 as zscore, -- z-score for 95% SLE
# MAGIC     a.cycle_stock,
# MAGIC     a.cycle_stddev,
# MAGIC     a.cycle_sales
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       x.store,
# MAGIC       x.item,
# MAGIC       y.cycle_start,
# MAGIC       y.cycle_end,
# MAGIC       SUM(x.sales_pred_mean) as cycle_stock,
# MAGIC       SUM(x.sales) as cycle_sales,
# MAGIC       STDDEV(x.sales) as cycle_stddev
# MAGIC     FROM forecasts x
# MAGIC     INNER JOIN (
# MAGIC       SELECT DISTINCT -- cycles
# MAGIC         store,
# MAGIC         item,
# MAGIC         next_day(date_add(date, -7), 'thursday') as cycle_start,
# MAGIC         next_day(date_add(date, -1), 'wednesday') as cycle_end
# MAGIC       FROM forecasts
# MAGIC       ) y 
# MAGIC       ON x.store=y.store AND x.item=y.item AND x.date BETWEEN y.cycle_start AND y.cycle_end
# MAGIC     WHERE YEAR(y.cycle_start) >= 2013 AND YEAR(y.cycle_end) <= 2017
# MAGIC     GROUP BY
# MAGIC       x.store,
# MAGIC       x.item,
# MAGIC       y.cycle_start,
# MAGIC       y.cycle_end
# MAGIC     ) a
# MAGIC   ) p
# MAGIC ORDER BY store, item, cycle_start

# COMMAND ----------

# MAGIC %md To examine the relationship between cycle stock, safety stock, required stock and actual demand, let's return to item 1 in store 1 for the years 2016 through 2017 and graph these components:

# COMMAND ----------

# MAGIC %sql -- stocking levels, store 1 and item 1 for >= 2016
# MAGIC 
# MAGIC SELECT
# MAGIC   p.store,
# MAGIC   p.item,
# MAGIC   p.cycle_start,
# MAGIC   p.cycle_stock,
# MAGIC   p.zscore * SQRT(p.cycle_days) * p.cycle_stddev as safety_stock,
# MAGIC   p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.cycle_stddev) as required_stock,
# MAGIC   p.cycle_sales
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     a.store,
# MAGIC     a.item,
# MAGIC     a.cycle_start,
# MAGIC     DATEDIFF(a.cycle_end, a.cycle_start)+1 as cycle_days,
# MAGIC     1.6449 as zscore, -- z-score for 95% SLE
# MAGIC     a.cycle_stock,
# MAGIC     a.cycle_stddev,
# MAGIC     a.cycle_sales
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       x.store,
# MAGIC       x.item,
# MAGIC       y.cycle_start,
# MAGIC       y.cycle_end,
# MAGIC       SUM(x.sales_pred_mean) as cycle_stock,
# MAGIC       SUM(x.sales) as cycle_sales,
# MAGIC       STDDEV(x.sales) as cycle_stddev
# MAGIC     FROM forecasts x
# MAGIC     INNER JOIN (
# MAGIC       SELECT DISTINCT -- cycles
# MAGIC         store,
# MAGIC         item,
# MAGIC         next_day(date_add(date, -7), 'thursday') as cycle_start,
# MAGIC         next_day(date_add(date, -1), 'wednesday') as cycle_end
# MAGIC       FROM forecasts
# MAGIC       ) y 
# MAGIC       ON x.store=y.store AND x.item=y.item AND x.date BETWEEN y.cycle_start AND y.cycle_end
# MAGIC     WHERE YEAR(y.cycle_start) >= 2013 AND YEAR(y.cycle_end) <= 2017
# MAGIC     GROUP BY
# MAGIC       x.store,
# MAGIC       x.item,
# MAGIC       y.cycle_start,
# MAGIC       y.cycle_end
# MAGIC     ) a
# MAGIC   ) p
# MAGIC WHERE p.store=1 AND p.item=1 AND YEAR(p.cycle_start)>=2016
# MAGIC ORDER BY store, item, cycle_start

# COMMAND ----------

# MAGIC %md In this graph, we can see our cycle stock running right down the middle of the actual demand.  Our safety stock provides us a buffer on the variability in that demand so that only occasionally does actual demand exceed our required stocking levels. If we examine all our forecasts, we should see that these occasional exceptions keep us right around the 95% service level expectation we set in deriving our safety stock quantities:

# COMMAND ----------

# MAGIC %sql -- calculate per model service levels
# MAGIC 
# MAGIC SELECT
# MAGIC   q.store,
# MAGIC   q.item,
# MAGIC   SUM(CASE WHEN q.cycle_sales > q.cycle_stock THEN 1 ELSE 0 END) as cycles_above_forecast,
# MAGIC   SUM(CASE WHEN q.cycle_sales > q.required_stock THEN 1 ELSE 0 END) as cycles_abovebufferedforecast,
# MAGIC   FORMAT_NUMBER(SUM(CASE WHEN q.cycle_sales > q.cycle_stock THEN 1 ELSE 0 END) / COUNT(*), '##.#%') as cycles_above_forecast_pct,
# MAGIC   FORMAT_NUMBER(SUM(CASE WHEN q.cycle_sales > q.required_stock THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as cycles_above_bufferedforecast_pct,
# MAGIC   FORMAT_NUMBER(1.0 - SUM(CASE WHEN q.cycle_sales > q.required_stock THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as service_level
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     p.store,
# MAGIC     p.item,
# MAGIC     p.cycle_start,
# MAGIC     p.cycle_stock,
# MAGIC     p.zscore * SQRT(p.cycle_days) * p.cycle_stddev as safety_stock,
# MAGIC     p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.cycle_stddev) as required_stock,
# MAGIC     p.cycle_sales
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       a.store,
# MAGIC       a.item,
# MAGIC       a.cycle_start,
# MAGIC       DATEDIFF(a.cycle_end, a.cycle_start)+1 as cycle_days,
# MAGIC       1.6449 as zscore, -- z-score for 95% SLE
# MAGIC       a.cycle_stock,
# MAGIC       a.cycle_stddev,
# MAGIC       a.cycle_sales
# MAGIC     FROM (
# MAGIC       SELECT
# MAGIC         x.store,
# MAGIC         x.item,
# MAGIC         y.cycle_start,
# MAGIC         y.cycle_end,
# MAGIC         SUM(x.sales_pred_mean) as cycle_stock,
# MAGIC         SUM(x.sales) as cycle_sales,
# MAGIC         STDDEV(x.sales) as cycle_stddev
# MAGIC       FROM forecasts x
# MAGIC       INNER JOIN (
# MAGIC         SELECT DISTINCT -- cycles
# MAGIC           store,
# MAGIC           item,
# MAGIC           next_day(date_add(date, -7), 'thursday') as cycle_start,
# MAGIC           next_day(date_add(date, -1), 'wednesday') as cycle_end
# MAGIC         FROM forecasts
# MAGIC         ) y 
# MAGIC         ON x.store=y.store AND x.item=y.item AND x.date BETWEEN y.cycle_start AND y.cycle_end
# MAGIC       WHERE YEAR(y.cycle_start) >= 2013 AND YEAR(y.cycle_end) <= 2017
# MAGIC       GROUP BY
# MAGIC         x.store,
# MAGIC         x.item,
# MAGIC         y.cycle_start,
# MAGIC         y.cycle_end
# MAGIC       ) a
# MAGIC     ) p
# MAGIC   ) q
# MAGIC GROUP BY q.store, q.item
# MAGIC ORDER BY store, item

# COMMAND ----------

# MAGIC %md ###Step 3: Revisiting the Standard Deviation of Demand
# MAGIC 
# MAGIC As mentioned earlier, we cheated when we performed our previous safety stock calculations.  We used knowledge of the actual demand in the future period for which we were forecasting to calculate the additional units needed to meet our service level expectations. In the real world, we would never have that knowledge and would need to estimate future variability.
# MAGIC 
# MAGIC One technique we could use is to calculate the variability for a past period and apply that to a future period.  This is often the technique used in spreadsheet exercises when safety stock calculations are taught in Supply Chain Management classes. But our data, as most retailer data, has a trend and seasonal patterns of variability which makes identifying a past period over which to calculate future demand variability problematic. 
# MAGIC 
# MAGIC Still, a few researchers in the Operations Management literature are exploring means for making these estimations.  If I'm being honest, most of the math being performed in the papers they are producing is beyond my immediate grasp, and there doesn't appear to be consensus in the field as to whether the techniques presented are the ones we should be using. 
# MAGIC 
# MAGIC And so we find ourselves a little stuck. But have no fear, we can use an old trick and substitute measures of model error provided in units aligned with that of the standard deviation of demand for this value.  The most commonly used of these measures are root mean squared error (RMSE) and mean absolute error (MAE).  Let's use these values now to recalculate our safety stock requirements:
# MAGIC 
# MAGIC NOTE Safety stock and required stock calculations based on known standard deviation of demand is kept in the results and labeled as *_perfect* for comparison purposes.

# COMMAND ----------

# MAGIC %sql -- calculate service levels with safety stock derived using RMSE and MAE
# MAGIC 
# MAGIC SELECT
# MAGIC   q.store,
# MAGIC   q.item,
# MAGIC   --FORMAT_NUMBER(SUM(CASE WHEN q.cycle_sales > q.required_stock_perfect THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as cycles_above_bufferedforecast_perfect_pct,
# MAGIC   --FORMAT_NUMBER(SUM(CASE WHEN q.cycle_sales > q.required_stock_rmse THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as cycles_above_bufferedforecast_rmse_pct,
# MAGIC   --FORMAT_NUMBER(SUM(CASE WHEN q.cycle_sales > q.required_stock_mae THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as cycles_above_bufferedforecast_mae_pct,
# MAGIC   FORMAT_NUMBER(1.0 - SUM(CASE WHEN q.cycle_sales > q.required_stock_perfect THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as service_level_perfect,
# MAGIC   FORMAT_NUMBER(1.0 - SUM(CASE WHEN q.cycle_sales > q.required_stock_rmse THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as service_level_rmse,
# MAGIC   FORMAT_NUMBER(1.0 - SUM(CASE WHEN q.cycle_sales > q.required_stock_mae THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as service_level_mae
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     p.store,
# MAGIC     p.item,
# MAGIC     p.cycle_start,
# MAGIC     p.cycle_stock,
# MAGIC     p.zscore * SQRT(p.cycle_days) * p.cycle_stddev as safety_stock_perfect,
# MAGIC     p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.cycle_stddev) as required_stock_perfect,
# MAGIC     p.zscore * SQRT(p.cycle_days) * p.rmse as safety_stock_rmse,
# MAGIC     p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.rmse) as required_stock_rmse,
# MAGIC     p.zscore * SQRT(p.cycle_days) * p.mae as safety_stock_mae,
# MAGIC     p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.mae) as required_stock_mae,
# MAGIC     p.cycle_sales
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       a.store,
# MAGIC       a.item,
# MAGIC       a.cycle_start,
# MAGIC       DATEDIFF(a.cycle_end, a.cycle_start)+1 as cycle_days,
# MAGIC       1.6449 as zscore, -- z-score for 95% SLE
# MAGIC       a.cycle_stock,
# MAGIC       a.cycle_stddev,
# MAGIC       b.rmse,
# MAGIC       b.mae,
# MAGIC       a.cycle_sales
# MAGIC     FROM (
# MAGIC       SELECT
# MAGIC         x.store,
# MAGIC         x.item,
# MAGIC         y.cycle_start,
# MAGIC         y.cycle_end,
# MAGIC         SUM(x.sales_pred_mean) as cycle_stock,
# MAGIC         SUM(x.sales) as cycle_sales,
# MAGIC         STDDEV(x.sales) as cycle_stddev
# MAGIC       FROM forecasts x
# MAGIC       INNER JOIN (
# MAGIC         SELECT DISTINCT -- cycles
# MAGIC           store,
# MAGIC           item,
# MAGIC           next_day(date_add(date, -7), 'thursday') as cycle_start,
# MAGIC           next_day(date_add(date, -1), 'wednesday') as cycle_end
# MAGIC         FROM forecasts
# MAGIC         ) y 
# MAGIC         ON x.store=y.store AND x.item=y.item AND x.date BETWEEN y.cycle_start AND y.cycle_end
# MAGIC       WHERE YEAR(y.cycle_start) >= 2013 AND YEAR(y.cycle_end) <= 2017
# MAGIC       GROUP BY
# MAGIC         x.store,
# MAGIC         x.item,
# MAGIC         y.cycle_start,
# MAGIC         y.cycle_end
# MAGIC       ) a
# MAGIC    INNER JOIN forecast_evals b
# MAGIC      ON a.store=b.store AND a.item=b.item
# MAGIC     ) p
# MAGIC   ) q
# MAGIC GROUP BY q.store, q.item
# MAGIC ORDER BY store, item

# COMMAND ----------

# MAGIC %md Substituting RMSE and MAE for the standard deviation of demand, our service levels have (typically) dropped.  Examining the MAE-derived stocking levels relative to actual demand for store 1 and item 1, again, we can see the effect on service levels in a more visual manner.  For this scenario, our service level has dropped to 92.7% despite the 95% expectation:

# COMMAND ----------

# MAGIC %sql -- stocking levels, store 1 and item 1 for >= 2016
# MAGIC 
# MAGIC   SELECT
# MAGIC     p.store,
# MAGIC     p.item,
# MAGIC     p.cycle_start,
# MAGIC     p.cycle_stock,
# MAGIC     p.zscore * SQRT(p.cycle_days) * p.cycle_stddev as safety_stock_perfect,
# MAGIC     p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.cycle_stddev) as required_stock_perfect,
# MAGIC     p.zscore * SQRT(p.cycle_days) * p.rmse as safety_stock_rmse,
# MAGIC     p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.rmse) as required_stock_rmse,
# MAGIC     p.zscore * SQRT(p.cycle_days) * p.mae as safety_stock_mae,
# MAGIC     p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.mae) as required_stock_mae,
# MAGIC     p.cycle_sales as actual_sales
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       a.store,
# MAGIC       a.item,
# MAGIC       a.cycle_start,
# MAGIC       DATEDIFF(a.cycle_end, a.cycle_start)+1 as cycle_days,
# MAGIC       1.6449 as zscore, -- z-score for 95% SLE
# MAGIC       a.cycle_stock,
# MAGIC       a.cycle_stddev,
# MAGIC       b.rmse,
# MAGIC       b.mae,
# MAGIC       a.cycle_sales
# MAGIC     FROM (
# MAGIC       SELECT
# MAGIC         x.store,
# MAGIC         x.item,
# MAGIC         y.cycle_start,
# MAGIC         y.cycle_end,
# MAGIC         SUM(x.sales_pred_mean) as cycle_stock,
# MAGIC         SUM(x.sales) as cycle_sales,
# MAGIC         STDDEV(x.sales) as cycle_stddev
# MAGIC       FROM forecasts x
# MAGIC       INNER JOIN (
# MAGIC         SELECT DISTINCT -- cycles
# MAGIC           store,
# MAGIC           item,
# MAGIC           next_day(date_add(date, -7), 'thursday') as cycle_start,
# MAGIC           next_day(date_add(date, -1), 'wednesday') as cycle_end
# MAGIC         FROM forecasts
# MAGIC         ) y 
# MAGIC         ON x.store=y.store AND x.item=y.item AND x.date BETWEEN y.cycle_start AND y.cycle_end
# MAGIC       WHERE YEAR(y.cycle_start) >= 2013 AND YEAR(y.cycle_end) <= 2017
# MAGIC       GROUP BY
# MAGIC         x.store,
# MAGIC         x.item,
# MAGIC         y.cycle_start,
# MAGIC         y.cycle_end
# MAGIC       ) a
# MAGIC    INNER JOIN forecast_evals b
# MAGIC      ON a.store=b.store AND a.item=b.item
# MAGIC     ) p
# MAGIC WHERE p.store=1 AND p.item=1 AND YEAR(p.cycle_start) >= 2016
# MAGIC ORDER BY store, item

# COMMAND ----------

# MAGIC %md Comparing the standard deviation of demand to the RMSE and MAE values for one of our models, we can see why our service levels (on the whole) are dropping:

# COMMAND ----------

# MAGIC %sql -- comparing standard deviation of demand to model-wide RMSE and MAE values
# MAGIC 
# MAGIC SELECT
# MAGIC   p.store,
# MAGIC   p.item,
# MAGIC   p.cycle_start,
# MAGIC   p.cycle_stddev,
# MAGIC   q.rmse,
# MAGIC   q.mae
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     x.store,
# MAGIC     x.item,
# MAGIC     y.cycle_start,
# MAGIC     y.cycle_end,
# MAGIC     SUM(x.sales_pred_mean) as cycle_stock,
# MAGIC     SUM(x.sales) as cycle_sales,
# MAGIC     STDDEV(x.sales) as cycle_stddev
# MAGIC   FROM forecasts x
# MAGIC   INNER JOIN (
# MAGIC     SELECT DISTINCT -- cycles
# MAGIC       store,
# MAGIC       item,
# MAGIC       next_day(date_add(date, -7), 'thursday') as cycle_start,
# MAGIC       next_day(date_add(date, -1), 'wednesday') as cycle_end
# MAGIC     FROM forecasts
# MAGIC     ) y 
# MAGIC     ON x.store=y.store AND x.item=y.item AND x.date BETWEEN y.cycle_start AND y.cycle_end
# MAGIC   WHERE YEAR(y.cycle_start) >= 2013 AND YEAR(y.cycle_end) <= 2017
# MAGIC   GROUP BY
# MAGIC     x.store,
# MAGIC     x.item,
# MAGIC     y.cycle_start,
# MAGIC     y.cycle_end
# MAGIC   ) p
# MAGIC INNER JOIN forecast_evals q
# MAGIC   ON p.store=q.store AND p.item=q.item
# MAGIC WHERE p.store=1 AND p.item=1

# COMMAND ----------

# MAGIC %md That said, here are instances when the standard deviation of demand is below the values of RMSE and MAE.  This can create situations within which RMSE and MAE may suggest higher safety stock.  When this occurs and such periods see higher demand, it is possible that an out of stock could occur with the supposedly *perfect* safety stock calculation which would not occur with the RMSE or MAE surrogate. In these situations, it is possible that RMSE and MAE deliver a higher service level.  You can see an example of this for store 1 and item 1 in the results table above.

# COMMAND ----------

# MAGIC %md But on the whole, our model-wide RMSE and MAE values are lower than the standard deviation values we frequently observe in our data. And this makes sense as the standard deviation in the demand is a feature of the historical data while the error metrics are a feature of a model that attempts to account for this exact variation.  As our modeling techniques improve, more and more variability (up to the limit that represents white noise in the data) is expected to be accounted for so that error metrics should decline while standard deviation of demand is unaffected.
# MAGIC 
# MAGIC Of course, our model-wide error metrics may be a little lower than they should be given they are based on predictions made over historical data points on which our model has been trained.  To address this, we can leverage metrics derived through cross-validation over a horizon aligned with the number of days over which we are making predictions at the time we place our replenishment orders.  Given we make our orders on Sunday evening (after business hours) for a period starting the following Thursday and ending the Wednesday after that, we should examine how our models behave on a horizon of 4 to 10 days (as these days should be 4 to 10 days after that Sunday evening).  Here we can see how RMSE and MAE behave over these time horizons for item 1 in store 1:

# COMMAND ----------

# MAGIC %sql -- RMSE and MAE values derived over different horizons using cross-validation
# MAGIC 
# MAGIC SELECT
# MAGIC   horizon,
# MAGIC   rmse,
# MAGIC   mae
# MAGIC FROM forecast_evals_cv
# MAGIC WHERE 
# MAGIC   store=1 AND 
# MAGIC   item=1 AND
# MAGIC   horizon BETWEEN 4 AND 10;

# COMMAND ----------

# MAGIC %md From the chart, we can see that both values tend to rise over time.  It would be possible for us to apply these metrics to each day in the performance cycle, but we might take a bit of a shortcut and just take the maximum value over the 4-10 day cycle to make things a bit easier.  Doing this, we can calculate our safety stock requirements over the historical periods and examine the actual service level obtained when we substitute the metrics:

# COMMAND ----------

# MAGIC %sql -- calculate service levels with safety stock derived using RMSE and MAE from model-wide and cross-validation calculations
# MAGIC 
# MAGIC SELECT
# MAGIC   q.store,
# MAGIC   q.item,
# MAGIC   FORMAT_NUMBER(1.0 - SUM(CASE WHEN q.cycle_sales > q.required_stock_perfect THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as service_level_perfect,
# MAGIC   FORMAT_NUMBER(1.0 - SUM(CASE WHEN q.cycle_sales > q.required_stock_rmse THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as service_level_rmse,
# MAGIC   FORMAT_NUMBER(1.0 - SUM(CASE WHEN q.cycle_sales > q.required_stock_rmse_cv THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as service_level_rmse_cv,
# MAGIC   FORMAT_NUMBER(1.0 - SUM(CASE WHEN q.cycle_sales > q.required_stock_mae THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as service_level_mae,
# MAGIC   FORMAT_NUMBER(1.0 - SUM(CASE WHEN q.cycle_sales > q.required_stock_mae_cv THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as service_level_mae_cv
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     p.store,
# MAGIC     p.item,
# MAGIC     p.cycle_start,
# MAGIC     p.cycle_stock,
# MAGIC     p.zscore * SQRT(p.cycle_days) * p.cycle_stddev as safety_stock_perfect,
# MAGIC     p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.cycle_stddev) as required_stock_perfect,
# MAGIC     p.zscore * SQRT(p.cycle_days) * p.rmse as safety_stock_rmse,
# MAGIC     p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.rmse) as required_stock_rmse,
# MAGIC     p.zscore * SQRT(p.cycle_days) * p.mae as safety_stock_mae,
# MAGIC     p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.mae) as required_stock_mae,
# MAGIC     p.zscore * SQRT(p.cycle_days) * p.rmse_cv as safety_stock_rmse_cv,
# MAGIC     p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.rmse_cv) as required_stock_rmse_cv,
# MAGIC     p.zscore * SQRT(p.cycle_days) * p.mae_cv as safety_stock_mae_cv,
# MAGIC     p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.mae_cv) as required_stock_mae_cv,
# MAGIC     p.cycle_sales
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       a.store,
# MAGIC       a.item,
# MAGIC       a.cycle_start,
# MAGIC       DATEDIFF(a.cycle_end, a.cycle_start)+1 as cycle_days,
# MAGIC       1.6449 as zscore, -- z-score for 95% SLE
# MAGIC       a.cycle_stock,
# MAGIC       a.cycle_stddev,
# MAGIC       b.rmse,
# MAGIC       b.mae,
# MAGIC       c.rmse_cv,
# MAGIC       c.mae_cv,
# MAGIC       a.cycle_sales
# MAGIC     FROM (
# MAGIC       SELECT
# MAGIC         x.store,
# MAGIC         x.item,
# MAGIC         y.cycle_start,
# MAGIC         y.cycle_end,
# MAGIC         SUM(x.sales_pred_mean) as cycle_stock,
# MAGIC         SUM(x.sales) as cycle_sales,
# MAGIC         STDDEV(x.sales) as cycle_stddev
# MAGIC       FROM forecasts x
# MAGIC       INNER JOIN (
# MAGIC         SELECT DISTINCT -- cycles
# MAGIC           store,
# MAGIC           item,
# MAGIC           next_day(date_add(date, -7), 'thursday') as cycle_start,
# MAGIC           next_day(date_add(date, -1), 'wednesday') as cycle_end
# MAGIC         FROM forecasts
# MAGIC         ) y 
# MAGIC         ON x.store=y.store AND x.item=y.item AND x.date BETWEEN y.cycle_start AND y.cycle_end
# MAGIC       WHERE YEAR(y.cycle_start) >= 2013 AND YEAR(y.cycle_end) <= 2017
# MAGIC       GROUP BY
# MAGIC         x.store,
# MAGIC         x.item,
# MAGIC         y.cycle_start,
# MAGIC         y.cycle_end
# MAGIC       ) a
# MAGIC    INNER JOIN forecast_evals b
# MAGIC      ON a.store=b.store AND a.item=b.item
# MAGIC   INNER JOIN (
# MAGIC     SELECT
# MAGIC       store,
# MAGIC       item,
# MAGIC       MAX(rmse) as rmse_cv,
# MAGIC       MAX(mae) as mae_cv
# MAGIC     FROM forecast_evals_cv
# MAGIC     WHERE horizon BETWEEN 4 AND 10
# MAGIC     GROUP BY store, item
# MAGIC     ) c
# MAGIC     ON a.store=c.store AND a.item=c.item
# MAGIC     ) p
# MAGIC   ) q
# MAGIC GROUP BY q.store, q.item
# MAGIC ORDER BY store, item

# COMMAND ----------

# MAGIC %md As before, most store and item combinations see a drop in actual service level as the max RMSE and MAE values in these periods are not much higher than that the model-wide values calculated earlier. But whether RMSE or MAE are above or below the variability in our data, measures of error and measures of variation (fluctuation) are different things. Substitution, while convenient, isn't necessarily consistent with intent of the safety stock formula.

# COMMAND ----------

# MAGIC %md Finally, users of Facebook Prophet are likely familiar with the fact that forecasted values are returned with upper and lower values representing a prediction interval for the mean value given variability in the average and trend components that make up the time series forecast.  If we calculate these lower and upper level values for a prediction interval aligned with our service level expectations, what kind of service level do we achieve?:

# COMMAND ----------

# MAGIC %sql -- calculate service levels with safety stock derived using RMSE and MAE from model-wide and cross-validation calculations
# MAGIC 
# MAGIC SELECT
# MAGIC   q.store,
# MAGIC   q.item,
# MAGIC   FORMAT_NUMBER(1.0 - SUM(CASE WHEN q.cycle_sales > q.required_stock_perfect THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as service_level_perfect,
# MAGIC   FORMAT_NUMBER(1.0 - SUM(CASE WHEN q.cycle_sales > q.required_stock_pi THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as service_level_pi,  
# MAGIC   FORMAT_NUMBER(1.0 - SUM(CASE WHEN q.cycle_sales > q.required_stock_rmse THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as service_level_rmse,
# MAGIC   FORMAT_NUMBER(1.0 - SUM(CASE WHEN q.cycle_sales > q.required_stock_rmse_cv THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as service_level_rmse_cv,
# MAGIC   FORMAT_NUMBER(1.0 - SUM(CASE WHEN q.cycle_sales > q.required_stock_mae THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as service_level_mae,
# MAGIC   FORMAT_NUMBER(1.0 - SUM(CASE WHEN q.cycle_sales > q.required_stock_mae_cv THEN 1 ELSE 0 END)/COUNT(*), '##.#%') as service_level_mae_cv
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     p.store,
# MAGIC     p.item,
# MAGIC     p.cycle_start,
# MAGIC     p.cycle_stock,
# MAGIC     p.zscore * SQRT(p.cycle_days) * p.cycle_stddev as safety_stock_perfect,
# MAGIC     p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.cycle_stddev) as required_stock_perfect,
# MAGIC     p.required_stock_pi,    
# MAGIC     p.zscore * SQRT(p.cycle_days) * p.rmse as safety_stock_rmse,
# MAGIC     p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.rmse) as required_stock_rmse,
# MAGIC     p.zscore * SQRT(p.cycle_days) * p.mae as safety_stock_mae,
# MAGIC     p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.mae) as required_stock_mae,
# MAGIC     p.zscore * SQRT(p.cycle_days) * p.rmse_cv as safety_stock_rmse_cv,
# MAGIC     p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.rmse_cv) as required_stock_rmse_cv,
# MAGIC     p.zscore * SQRT(p.cycle_days) * p.mae_cv as safety_stock_mae_cv,
# MAGIC     p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.mae_cv) as required_stock_mae_cv,
# MAGIC     p.cycle_sales
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       a.store,
# MAGIC       a.item,
# MAGIC       a.cycle_start,
# MAGIC       DATEDIFF(a.cycle_end, a.cycle_start)+1 as cycle_days,
# MAGIC       1.6449 as zscore, -- z-score for 95% SLE
# MAGIC       a.cycle_stock,
# MAGIC       a.cycle_stddev,
# MAGIC       b.rmse,
# MAGIC       b.mae,
# MAGIC       c.rmse_cv,
# MAGIC       c.mae_cv,
# MAGIC       a.cycle_sales,
# MAGIC       a.required_stock_pi
# MAGIC     FROM (
# MAGIC       SELECT
# MAGIC         x.store,
# MAGIC         x.item,
# MAGIC         y.cycle_start,
# MAGIC         y.cycle_end,
# MAGIC         SUM(x.sales_pred_mean) as cycle_stock,
# MAGIC         SUM(x.sales) as cycle_sales,
# MAGIC         STDDEV(x.sales) as cycle_stddev,
# MAGIC         SUM(x.sales_pred_upper) as required_stock_pi
# MAGIC       FROM forecasts x
# MAGIC       INNER JOIN (
# MAGIC         SELECT DISTINCT -- cycles
# MAGIC           store,
# MAGIC           item,
# MAGIC           next_day(date_add(date, -7), 'thursday') as cycle_start,
# MAGIC           next_day(date_add(date, -1), 'wednesday') as cycle_end
# MAGIC         FROM forecasts
# MAGIC         ) y 
# MAGIC         ON x.store=y.store AND x.item=y.item AND x.date BETWEEN y.cycle_start AND y.cycle_end
# MAGIC       WHERE YEAR(y.cycle_start) >= 2013 AND YEAR(y.cycle_end) <= 2017
# MAGIC       GROUP BY
# MAGIC         x.store,
# MAGIC         x.item,
# MAGIC         y.cycle_start,
# MAGIC         y.cycle_end
# MAGIC       ) a
# MAGIC    INNER JOIN forecast_evals b
# MAGIC      ON a.store=b.store AND a.item=b.item
# MAGIC   INNER JOIN (
# MAGIC     SELECT
# MAGIC       store,
# MAGIC       item,
# MAGIC       MAX(rmse) as rmse_cv,
# MAGIC       MAX(mae) as mae_cv
# MAGIC     FROM forecast_evals_cv
# MAGIC     WHERE horizon BETWEEN 4 AND 10
# MAGIC     GROUP BY store, item
# MAGIC     ) c
# MAGIC     ON a.store=c.store AND a.item=c.item
# MAGIC     ) p
# MAGIC   ) q
# MAGIC GROUP BY q.store, q.item
# MAGIC ORDER BY store, item

# COMMAND ----------

# MAGIC %md At first blush, it appears the prediction interval derived stocking level is the way to go, but let's take a look at the stocking requirements that are associated with this level vs. the other levels for one of our forecasts, item 1 & store 1:

# COMMAND ----------

# MAGIC %sql -- calculate service levels with safety stock derived using RMSE and MAE from model-wide and cross-validation calculations
# MAGIC 
# MAGIC SELECT
# MAGIC   p.store,
# MAGIC   p.item,
# MAGIC   p.cycle_start,
# MAGIC   p.cycle_stock,
# MAGIC   p.cycle_stock + (p.zscore * SQRT(p.cycle_stddev) * p.rmse) as required_stock_perfect,
# MAGIC   p.required_stock_pi,    
# MAGIC   p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.rmse) as required_stock_rmse,
# MAGIC   p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.mae) as required_stock_mae,
# MAGIC   p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.rmse_cv) as required_stock_rmse_cv,
# MAGIC   p.cycle_stock + (p.zscore * SQRT(p.cycle_days) * p.mae_cv) as required_stock_mae_cv,
# MAGIC   p.cycle_sales
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     a.store,
# MAGIC     a.item,
# MAGIC     a.cycle_start,
# MAGIC     DATEDIFF(a.cycle_end, a.cycle_start)+1 as cycle_days,
# MAGIC     1.6449 as zscore, -- z-score for 95% SLE
# MAGIC     a.cycle_stock,
# MAGIC     a.cycle_stddev,
# MAGIC     b.rmse,
# MAGIC     b.mae,
# MAGIC     c.rmse_cv,
# MAGIC     c.mae_cv,
# MAGIC     a.cycle_sales,
# MAGIC     a.required_stock_pi
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       x.store,
# MAGIC       x.item,
# MAGIC       y.cycle_start,
# MAGIC       y.cycle_end,
# MAGIC       SUM(x.sales_pred_mean) as cycle_stock,
# MAGIC       SUM(x.sales) as cycle_sales,
# MAGIC       STDDEV(x.sales) as cycle_stddev,
# MAGIC       SUM(x.sales_pred_upper) as required_stock_pi
# MAGIC     FROM forecasts x
# MAGIC     INNER JOIN (
# MAGIC       SELECT DISTINCT -- cycles
# MAGIC         store,
# MAGIC         item,
# MAGIC         next_day(date_add(date, -7), 'thursday') as cycle_start,
# MAGIC         next_day(date_add(date, -1), 'wednesday') as cycle_end
# MAGIC       FROM forecasts
# MAGIC       ) y 
# MAGIC       ON x.store=y.store AND x.item=y.item AND x.date BETWEEN y.cycle_start AND y.cycle_end
# MAGIC     WHERE YEAR(y.cycle_start) >= 2013 AND YEAR(y.cycle_end) <= 2017
# MAGIC     GROUP BY
# MAGIC       x.store,
# MAGIC       x.item,
# MAGIC       y.cycle_start,
# MAGIC       y.cycle_end
# MAGIC     ) a
# MAGIC  INNER JOIN forecast_evals b
# MAGIC    ON a.store=b.store AND a.item=b.item
# MAGIC INNER JOIN (
# MAGIC   SELECT
# MAGIC     store,
# MAGIC     item,
# MAGIC     MAX(rmse) as rmse_cv,
# MAGIC     MAX(mae) as mae_cv
# MAGIC   FROM forecast_evals_cv
# MAGIC   WHERE horizon BETWEEN 4 AND 10
# MAGIC   GROUP BY store, item
# MAGIC   ) c
# MAGIC   ON a.store=c.store AND a.item=c.item
# MAGIC   ) p
# MAGIC WHERE store=1 AND item=1
# MAGIC ORDER BY store, item

# COMMAND ----------

# MAGIC %md From the visualization we can see that using the Prophet prediction intervals results in a significant over-estimation of the required stock.  The reasons for this are not terribly important.  The main point of including this calculation in this notebook is to make sure those familiar with Facebook Prophet don't come to think of prediction intervals as a short-but to a solution when it comes to safety stock calculations.  (Remember, every unit of stock in inventory represents idle capital that's at risk of loss.)
# MAGIC 
# MAGIC So where does this leave us with regard to performing safety stock calculations? At a minimum, I hope that this exercise helps us to understand that safety stock calculations require careful consideration and that commonly adopted techniques may not deliver the results we expect. I'd encourage organizations trying to improve inventory management to review historical data and forecasting models and explore the actual service levels achieved based on calculations similar to the ones demonstrated here.  This may help organizations identify changes in their stocking practices that allow them to keep inventories manageable while still meeting their desired service levels.

# COMMAND ----------

# MAGIC %md ###Footnote: Setting Replenishment Quantities
# MAGIC 
# MAGIC We'd be remiss if we didn't point out that the stock calculations performed in this notebook do not imply that we are ordering exactly the quantities shown.  Most products that we stock will have longer than a 1-cycle shelf-life.  If demand is less than the forecasted demand 50% of the time and we are stocking for the possibility of demand to exceed that, we will be left with inventory at the end of a cycle which should carry over into the next. We need to take this into consideration as we place our replenishment orders Sunday evening.  But of course, we still have 3 days left in the current cycle so that we don't know exactly where will land come Wednesday evening.
# MAGIC 
# MAGIC To understand how we might take into consideration the possibility of carry-over inventory when we are placing orders mid-cycle, let's continue considering our 7-day Thursday through Wednesday cycle for which replenishment orders are placed on Sunday evening. At the time of order placement, we will still have 3 days of business remaining in the current cycle.  To account for the uncertainty around these three days, what we can do is recalculate forecasted demand and safety stock for the 10-day cycle which includes the Monday, Tuesday and Wednesday leading into the Thursday through Wednesday cycle. By subtracting the end of the day Sunday on-hand inventory from this 10-day required stock, we now have our replenishment order quantity.
