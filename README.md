## Safety Stock Calculations for Inventory Management

Periodically, we need to order product to replenish our inventory. When we do this, we have in mind a future period for which we are attempting to address demand along with an estimate the demand in that period.

Our estimates are never perfectly accurate.  Instead, when we use modern forecasting techniques, we are estimating the *mean* demand across a range of potential demand values. Improvements in model accuracy narrow the range of variability around this predicted mean, but we still expect to be below the mean 50% of the time and above it the other 50% (as that's simply the nature of a mean value). 

When actual demand exceeds our forecasts, we run the risk of a stockout (out of stock) situation with its associated potential loss of sales and reduced customer satisfaction. To avoid this, we often include additional units of stock, above the forecasted demand, in our replenishment  orders. The amount of this *safety stock* depends on our estimates of variability in the demand for this upcoming period and the percentage of time we are willing to risk an out of stock situation.  

For a more in-depth examination of safety stock calculations, please refer to the blog associated with this notebook.  The purpose of the information provided here is to examine how forecast data can be employed to effectively calculate safety stock requirements leveraging standard formulas.
___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

To run this accelerator, clone this repo into a Databricks workspace. Attach the RUNME notebook to any cluster running a DBR 11.0 or later runtime, and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. Execute the multi-step-job to see how the pipeline runs.

The job configuration is written in the RUNME notebook in json format. The cost associated with running the accelerator is the user's responsibility.
