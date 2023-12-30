
# House Price Forecasting

## Problem Statement


One of the key drivers in housing markets is house price. Predicting future house price
movement is critical to housing market. Due to the significant market changes from COVID19, there has been an increasing need to develop the new house price forecast model which is able to accurately forecast house prices during market turmoil, such as COVID19 pandemic. For instance, factors like unemployment and GDP have behaved differently during the Great Recession and COVID19 Pandemic. During the Great Recession of 2008-2009, the house prices rose with the decrease in unemployement, but during COVID19 pandemic, the house prices rose with the increase in unemployment rate. This induced large errors in future predictions. Multiple models failed to predict accurate results during the COVID19 pandemic. 

To address this issue, I have attempted to try different time series forecasting models on California data to identify how machine learning can impact the prediction results using different models like Support Vector Regression, ARIMA, Exponential Smoothing, Facebook Prophet and Long Short-Term Memory models. 


Dataset Link - https://github.com/nikunjachoure/House-Price-Prediction/blob/main/Master_Data.xlsx
 

## Dataset Overview

- 15 TOTAL INDICATOR COLUMNS
- 216 DATA POINTS
- YEARS ANALYZED :
    - TRAIN: 2005 TO 2019
    - TEST: 2020 TO 2022

- Data Source :

| **Source** | **Dataset Columns** |
| ------ | --------------- |
| State of California Department of Finance | Labor Force & Job Numbers |
| State of California Department of Finance | Inflation & Consumer Price Index |
| Federal Housing Finance Agency | Interest Rates for Single-Family Homes |
| Federal Housing Finance Agency | Monthly Average Loan Amounts |
| Freddie Mac’s Mortgage Interest Rates | 30-Year Fixed Mortgage Interest Rates |
| Freddie Mac’s Mortgage Interest Rates | 15-Year Fixed Mortgage Interest Rates |
| Federal Reserve Economic Data | New Private Housing Units Authorized by Building Permits for California |
| Bureau of Economic Analysis  | Annual Gross Domestic Product by State |
| Google Trends | Housing Interest Index |

## Data Cleaning

- NULL VALUES : Removed rows with consistently null values
- DATA FILTERING : Cleaned excess data columns, disregarded columns with yearly data, filtered out California data
- TIME RANGE SLICING : Sliced data to 2005-2022 time periods

## Data Preprocessing

- AGGREGATION : Converted data with weekly values to monthly values
- IMPUTATION : Filled empty values with imputed estimates using percieved trend values
- DATA TYPECASTING : Converted float & string values to Integers

## Exploratory Data Analysis

#### Consumer Price Index Behaviour over Time

![Summary_Page](https://github.com/nikunjachoure/House-Price-Prediction/blob/main/Model%20Results%20Snapshots/EDA_1.png?raw=true)

- 2008-2009 Downturn (Falling CPI): Reflects deflationary pressures amid the global financial crisis.
- 2019-2021 Stability (Constant CPI): Signifies an economically steady period with moderate inflation, providing a baseline for consistent housing market trends.
- Post-2021 Inflationary Surge (Rising CPI): Indicates sudden CPI rise, highlighting inflationary pressures.

#### Does CPI have an impact on Seasonaly adjusted House price index (NSA HPI)?

![Summary_Page](https://github.com/nikunjachoure/House-Price-Prediction/blob/main/Model%20Results%20Snapshots/EDA_2.png?raw=true)

```NSA=1.26×CPI−115.17```

- This indicates a positive correlation between CPI and NSA housing price index.
- CPI being the most important feature to predict the Housing Price Index.

#### Does unemployment affect sales of houses?

![Summary_Page](https://github.com/nikunjachoure/House-Price-Prediction/blob/main/Model%20Results%20Snapshots/EDA_3.png?raw=true)

We see a linear relationship between Unsold Inventory Index (UII) and Employment
Disruptions During Anomalies :
    - 2008 Financial Crisis
    - COVID-19 Pandemic
    - Predictive Insights

Though the relationship is inversely proportional, human intervention can change that trend.

#### Relationships between data fields

![Summary_Page](https://github.com/nikunjachoure/House-Price-Prediction/blob/main/Model%20Results%20Snapshots/EDA_4.png?raw=true)

Despite each feature holding significant theoretical weight in predicting housing prices, they reveal non-linear relationships. To comprehensively analyze this theory, I am employing diverse statistical models, that adds a nuanced statistical perspective.

## Machine Learning models

Considering the time series data, I have specifically implemented mpdels that are suitable for time series forecasting and analysed each model's performance.

### Exponential Smoothing

Suitable for time series forecasting when the data exhibits trends and seasonality.

![Summary_Page](https://github.com/nikunjachoure/House-Price-Prediction/blob/main/Model%20Results%20Snapshots/Exponential_Smoothing.png?raw=true)

#### Performance metrics 

    - MSE = 1398
    - RMSE = 39
    - MAE = 30.4
    - R-square = -.47

The model doesn't capture the patterns or trends in the data and is not providing meaningful predictions.

### Facebook Prophet Model

This model is developed by Meta for Time Series Forecasting and works best with strong seasonal effects and missing values. This model is known to address changepoints in the data, which means that it captures and adapts to different patterns, shifts and events in the time series.

Facebook Prophet predicts data only when it is in a certain format.

```y - Data to be forecasted```

```ds - Time series data```

![Summary_Page](https://github.com/nikunjachoure/House-Price-Prediction/blob/main/Model%20Results%20Snapshots/Prophet_model_1.png?raw=true)

I have tested 3 models :

- Model 1 : Prophet Univariate Model

![Summary_Page](https://github.com/nikunjachoure/House-Price-Prediction/blob/main/Model%20Results%20Snapshots/Prophet_Model1.png?raw=true)
#### Performance metrics 

    - RMSE = 32.77
    - MAE = 26.87
    - MSE = 1074

- Model 2 : Prophet Multivariate Model

![Summary_Page](https://github.com/nikunjachoure/House-Price-Prediction/blob/main/Model%20Results%20Snapshots/Prophet_Model2.png?raw=true)
#### Performance metrics 

    - RMSE = 51.22
    - MAE = 48.72
    - MSE = 2623

- Model 3 : Prophet Multivariate Model - Feature Selection

Feature selection is done using recursive feature elimination cross validation (RFECV) method. The selected features are :
    
    - y (Index_SA)
    - ds (Date)
    - Mortgage interest rates
    - Civil Labor Force
    - 15 years FRM Average
    - Sales of Existing Single Family Homes (% y-o-y change)

![Summary_Page](https://github.com/nikunjachoure/House-Price-Prediction/blob/main/Model%20Results%20Snapshots/Prophet_Model3.png?raw=true)
#### Performance metrics 

    - RMSE = 28.14
    - MAE = 22.69
    - MSE = 792  

Model 3 is the best performing prophet model. Model tries to capture the seasonal trend. With more impactful indicators, there is a possibility of better prediction results.

### Long Short-term Memory (LSTM)

This model is a type of recurrent neural network (RNN) architecture designed to capture and learn patterns in sequential data. It has four main features which enhance the predictions :
- Memory Cells : Have the ability to store information for long durations, capturing long-term dependencies in sequential data.
- Gates : Three gates (input gate, forget gate, and output gate) to control the flow of information.
- Cell State : Acts as a conveyor belt, allowing information to be carried across time steps.
- Hidden State : Output of the LSTM and contains information relevant for making predictions.

For this model, the code uses these features in the following manner

![Summary_Page](https://github.com/nikunjachoure/House-Price-Prediction/blob/main/Model%20Results%20Snapshots/LSTM_code.png?raw=true)

- Memory Cells : LSTM layer with 50 memory cells. Seq_length represents the length of the input sequence, and 6 represents the number of features for each time step.
- Gates : Handled internally by the LSTM layers. The return_sequences=True parameter: sequences are returned for each time step.
- Cell State : Handled internally by the LSTM layers. 
- Hidden State : Final LSTM layer & the subsequent dense layer contribute to the generation of the hidden state.

I tested 2 models :

- Model 1 : LSTM Model 1

![Summary_Page](https://github.com/nikunjachoure/House-Price-Prediction/blob/main/Model%20Results%20Snapshots/LSTM_Model1.png?raw=true)
#### Performance metrics 

    - RMSE = 30.65
    - MAE = 27.86
    - MSE = 939.5

- Model 2 : LSTM Model 2 - Feature Selection

The selected features are :
    
    - CPI
    - Mortgage interest rates
    - Civil Labor Force
    - 15 years FRM Average
    - Sales of Existing Single Family Homes (% y-o-y change)

![Summary_Page](https://github.com/nikunjachoure/House-Price-Prediction/blob/main/Model%20Results%20Snapshots/LSTM_Model2.png?raw=true)
#### Performance metrics 

    - RMSE = 10.46
    - MAE = 6.94
    - MSE = 109.55

Model 2 (feature selection) is the better performing LSTM model.

### ARIMA Model

This model excels in forecasting time-dependent data, capturing historical patterns for future predictions. ARIMA's interpretability and diagnostic capabilities make it a reliable choice for forecasting housing price indices over time. ARIMA models can be extended to SARIMAX (Seasonal ARIMA) to incorporate seasonality in the data.

#### Model Components

- Orders for ARIMA
    - p: Autoregressive (AR) order - The number of lag observations included in the model.
    - d: Integrated (I) order - The number of times that the raw observations are differenced (made stationary).
    - q: Moving Average (MA) order - The size of the moving average window.

- Orders for SARIMAX
    - Same orders as ARIMA with an additional order for reasonality
    - s : The number of time steps in a seasonal period.

- Order Selection
    - Model performance varies drastically with the selection of orders.
    - Orders were selected by training both the model multiple times using different values of orders and the best performing order values were finalized.

![Summary_Page](https://github.com/nikunjachoure/House-Price-Prediction/blob/main/Model%20Results%20Snapshots/Arima_Model_Order.png?raw=true)    

#### Predictions

- ARIMA

![Summary_Page](https://github.com/nikunjachoure/House-Price-Prediction/blob/main/Model%20Results%20Snapshots/Arima_Model1.png?raw=true)

#### Performance metrics 

    - RMSE = 49.13
    - MSE = 2414.34

- SARIMAX

![Summary_Page](https://github.com/nikunjachoure/House-Price-Prediction/blob/main/Model%20Results%20Snapshots/Sarimax_model1.png?raw=true)

#### Performance metrics 

    - RMSE = 29.27
    - MSE = 857.12

SARIMAX performed better than ARIMA.

## Conclusion

After implementation of machine learning and Neural Network models, there is evidence that these models are worth looking into with tuning and domain knowledge.
