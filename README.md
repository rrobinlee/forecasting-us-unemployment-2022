## Forecasting 2022 U.S. Unemployment Rate Based on Housing Supply Rate and Interest Rate Using Federal Reserve Economic Data 

*February 27, 2023*

### 1 Introduction

With the number of layoffs in the tech industry escalating throughout the United States following a nation-wide increase in interest rates, we have hypothesized that interest rate is definitely a significant factor that noticeably affects the unemployment rate—raising the question, what other circumstances influence employment? Thus, by applying time series analysis, we attempt to answer the following: *what are other potential factors that impact the unemployment rate?*

In this research assignment, in addition to interest rate, we have picked the monthly supply of new houses as another independent variable. The intuition behind this choice is that we believe when a massive layoff happens—such as the one recently in 2021—people tend to sell their real estate, leading to a significant increase in the housing supply. This is especially true and exemplified during the 2008 Financial Crisis.

As such, we are curious as to whether or not this phenomenon is a common pattern. We have chosen three time series datasets: interest rate, unemployment, and housing supply. Below is a table summarizing the time series datasets we have selected from the Federal Reserve Economic Data (FRED). Data coverage includes all major areas of macroeconomic analysis: growth, inflation, employment, interest rates, exchange rates, production and consumption, income and expenditure, savings and investment, and more

| Variable Code | Description of variable measures, frequency, source | Time Period |
|:--------------|:----------------------------------------------------|:------------|
| MSACSRNSA     | U.S. Census Bureau, Monthly<br>The months' supply is the ratio of new houses for sale to new houses sold. |1963:1-2022:12|
| UNRATENSA     | U.S. Bureau of Labor Statistics, Monthly<br>16+ years old, reside in 50 states or the District of Columbia, do not reside in institutions or are on active duty in the Armed Forces. |1948:1-2022:12|
| FEDFUNDS      | Board of Governors of the Federal Reserve System (US), Monthly<br>Interest rate at which depository institutions trade federal funds with each other overnight. |1954:7-2022:12|

Table 1: Data Summary

After importing our data into R using the Quandl API, we can display the three time series data on the same plot, as shown in Plot 1. To do this, we first need to make sure that they have the same start time and end time. For simplicity, we picked the time range from `1980-01-01` to `2022-12-01` for all of the three datasets. In this way, we are able to view our full dataset.

Later, we will use part of the full dataset as the training data.

<img width="629" alt="image" src="https://github.com/user-attachments/assets/475b72d1-3b11-44a7-86b9-c104c180eeed" />

Plot 1: How does unemployment rate vary together with housing supply and interest rate

From Plot 1, we can see that the unemployment rate has a similar trend compared to both federal funds effective rate and housing supply ratio. To conduct analysis in later parts of this project, we have extracted the last 12 data points as the testing data and used the remaining data points as the training data. That is to say, we use the data from 1980-01 to 2018-01 as the training data and that from `2018-02` to `2019-1` as the testing data.

### 2 Components Feature Analysis

Because we want to see how new housing supply and interest rate will affect unemployment, we have chosen Unemployment Rate as our dependent variable. Based on the dyplot shown earlier (Plot 1), we can see that there is seasonality and no obvious trend in the unemployment data, as indicated by the green line.

#### 2.1 Decomposition for Unemployment Training Data

That is to say, we know the seasonality does not increase with the trend. As a result, we can apply additive decomposition, instead of multiplicative decomposition, for our time series.

* The additive model is useful when the seasonal variation is relatively constant over time.
* The multiplicative model is useful when the seasonal variation increases over time.
  
In other words, the additive model is used when the variance of the time series does not change over different values of the time series. On the other hand, if the variance is higher when the time series is higher, then it often means we should use a multiplicative model.

<img width="629" alt="image" src="https://github.com/user-attachments/assets/c21d3687-fa52-4ee6-ab62-74bb94e5a759" />

Plot 2: Additive Decomposition of Unemployment Training Data

We first plot the additive decomposition of the training data (Plot 2). We see that the range of fluctuation for the random term is relatively small (from -0.4 to 0.4), implying we do not need any transformation on the raw dataset; as such, an additive decomposition is good enough.

<img width="629" alt="image" src="https://github.com/user-attachments/assets/f456ed3e-b41b-469b-af15-02def33d1895" />

Plot 2.1: Random Term of Additive Decomposition

#### 2.2 Seasonal Box Plot for Unemployment Training Data

A seasonal box plot is a graphical representation used to display the distribution of a dataset over time, specifically focusing on seasonal patterns. Each box plot within a seasonal box plot represents the distribution of the data for a particular time interval (e.g., each month of the year). The box itself represents the interquartile range (IQR), with the median marked by a line. Seasonal box plots are useful for visualizing seasonal patterns and identifying any variations or trends that occur within specific time intervals.

<img width="629" alt="image" src="https://github.com/user-attachments/assets/ead81411-9d37-4784-8b2a-a063c8d563b0" />

Plot 3: Seasonal boxplot for the unemployment data

From the boxplot (Plot 3), we can see that the unemployment data does show certain seasonality. In particular, we can see that the unemployment rate drops from January to May, before increasing again in June. However, after this growth in June, the unemployment rate begins to drop again from July to December.

### 3 Autocorrelation Feature Analysis

Next, we plot the ACF and PACF graphs for the training data. The ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots are graphical tools used in time series analysis to identify the autocorrelation structure of a time series dataset.

These plots help in determining the appropriate parameters for autoregressive integrated moving average (ARIMA) models, which are commonly used for forecasting.

<img width="629" alt="image" src="https://github.com/user-attachments/assets/be07b95b-013e-4b73-89bb-f00d3002bc32" />

Plot 4: ACF and PACF plots for the unemployment data

Both ACF and PACF plots are crucial for determining the order of autoregressive (AR) and moving average (MA) terms in an ARIMA model. The patterns observed in these plots guide the selection of appropriate model parameters (p, d, q) for fitting the time series data.

Looking at the ACF graph (Plot 4), we notice that the autocorrelations for the random term stay significant until a larger lag, but it gradually cuts off.

<img width="629" alt="image" src="https://github.com/user-attachments/assets/3e91ac5c-497d-4c18-94ca-338ae8d1f6aa" />

Plot 5: ACF plots for the AR(1) model with parameter 0.4

For the regular part, it seems like an `ARMA(4, 2)` model. By looking at the ACF graph, the peak cuts off at lag 2 (`MA(2)`), and in the PACF, the peak cuts off at lag 4 — `AR(4)`. For the seasonal part, it looks like the `MA(3)` model.
By comparing the raw time series plots and the ACF plots, we believe an autoregressive process with an order of 1 with a parameter of 0.4 might generate our dependent variables; this is because the ACF graphs are similar.

### 4 Exponential Smoothing Modeling and Forecasting

Using the raw training dependent variable—Unemployment—we fit an appropriate exponential smoothing model and forecast. As such, we decide to use seasonal Holt-Winters exponential smoothing, which fits a time series with seasonality and trend. Furthermore, we implement additive decomposition, because the model’s seasonality does not increase with the trend.

Exponential smoothing methods are widely used due to their simplicity, ease of implementation, and ability to provide reasonably accurate forecasts for a wide range of time series data.

The additive Holt-Winters prediction function for time series with period length p is defined as:

$\hat{y_{t+h}} = l_t + hb_t +s_{t+h-m}$

This function tries to find the optimal values of alpha (defined as l in the equation above), beta,
and/or gamma. We obtain the following results:

| Smoothing Parameters | Coefficients        |                     |
|:---------------------|:--------------------|:--------------------|
| `alpha` = 0.8429     | `a` = 4.07894326    | `s6` = 0.32706991   |
| `beta` = 0.0283      | `b` = -0.04468121   | `s7` = -0.01516064  |
| `gamma` = 1          | `s1` = 0.40741155   | `s8` = -0.40081732  |
|                      | `s2` = 0.25960977   | `s9` = -0.52094787  |
|                      | `s3` = 0.21651062   | `s10` = -0.52039932 |
|                      | `s4` = -0.21651062  | `s11` = -0.35669888 |
|                      | `s5` = 0.30746959   | `s12` = -0.42105674 |

Table 2: Smoothing Parameters and Coefficients of Holt-Winters Function

We forecast as many periods ahead as observations are in the test set, as exemplified below:

|      | Jan  | Feb  | Mar  | Apr  | May  | Jun  | Jul  | Aug  | Sep  | Oct  | Nov  | Dec  |
|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|
| 2018 |      | 4.44 | 4.24 | 3.72 | 3.77 | 4.16 | 4.13 | 3.78 | 3.32 | 3.15 | 3.11 | 3.23 |
| 2019 | 3.96 |      |      |      |      |      |      |      |      |      |      |      |

Table 3: Forecast of Raw Training Dependent Variable

By comparing the test set of the raw data with the forecast we just obtained, we can see that our forecast is fairly accurate at first, but deviates quite significantly after the first six months.

|      | Jan  | Feb  | Mar  | Apr  | May  | Jun  | Jul  | Aug  | Sep  | Oct  | Nov  | Dec  |
|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|
| 2018 |      | 4.4  | 4.1  | 3.7  | 3.6  | 4.2  | 4.1  | 3.9  | 3.6  | 3.5  | 3.5  | 3.7  |
| 2019 | 4.4  |      |      |      |      |      |      |      |      |      |      |      |

Table 4: Test Set of Raw Training Dependent Variable

All in all, using additive Holt-Winters on the raw training dependent variable, we forecast the number of unemployed as a percentage of the labor force from February 2018 to January 2019; this is expressed in forecasts. We compare these values to unemployment_rate_test, which display the true values of the raw data.

Comparing Table 3 and Table 4, it appears that the forecast data matches fairly well with the test set of the raw data at first, but grows further apart as the months increase.

The forecasted values are about 0.1 from the actual model in the first half of 2018, but the difference increases to more than 0.4 by January 2019.

Finally, we plot the final fitted model and forecast. Note: Using the smoothing parameters and coefficients from Table 2, we define the fitted model as:

$\hat{Y_{t+h}} = (a_t + h_t) * s_{t + 1 + (h - 1) * mod(p)}$
where:
* $a_t =   0.8428519 * (Y_t / s_{t-p}) + (1-0.8428519) (a_{t-1} + b_{t-1})$
* $b_t =   0.02833799 * (a_t - a_{t-1}) + (1-0.02833799) * b_{t-1}$
* $s_t = (1) * (Y_t / a_t) + (1-1) * s_{t-p}$

Using the raw training dependent variable as the primary focus, we can see in the exponential smoothing model (Plot 6) that the forecast predicts that the unemployment rate will continue decreasing throughout 2018 and 2019.

![image](https://github.com/user-attachments/assets/c6b54ec1-021d-4d88-9aa2-6d7267d34246)

Plot 6: Seasonal Exponential Smoothing for Unemployment Rate

The fitted model (solid red line) appears to fit the actual values (black line) fairly well, and we can see an overall negative trend from 2010 onwards. This signifies that our fitted model is accurate, and that our forecast (dotted line) is most likely accurate as well.

### 5 Polynomial Regression, Seasonal Effect, Forecasting

Polynomial regression is a type of regression analysis where the relationship between the independent variable x and the dependent variable y is modeled as an nth-degree polynomial function. Unlike simple linear regression, which fits a straight line to the data, polynomial regression can capture more complex relationships between variables by allowing for curves and bends. Higher-degree polynomials can capture more patterns but may also lead to overfitting.

In our report, we select the 4th order polynomial model after fitting the 1st, 2nd, 3rd, 4th, and 5th order polynomial models, before assessing their fit to the data. The 1st order (linear) to 3rd order polynomial models were underfitting the data while the 5th order polynomial appeared to be overfitting the data.

**Equation of fitted polynomial regression model:**

$y_t = 1.938 + 8.454(10^{-3})t - 1.287(10^{-4}t^2 + 5.054(10^{-7})t^3-5.91(10^{-10})t^4$

As mentioned earlier, we select the 4th order polynomial model. However, it is essential to interpret the results with caution, as higher-degree polynomials can lead to overfitting and may not generalize well to new data.

![image](https://github.com/user-attachments/assets/97011f28-b097-4e49-b67d-ea707dc39e45)

Plot 7: Polynomial Regression Smoother and Forecast

Plot 7 shows the plot of a 4th order polynomial model fitted to the training data. The red line represents the trend as modeled by polynomial smoother. The blue dotted line indicates where the forecast begins, with seasonality added to the trend.

![image](https://github.com/user-attachments/assets/4d74d7c1-f0a7-4c43-a9ce-5a1dff95e28a)

Plot 8: Scaled Residuals of 4th Order Polynomial Model

We determine the goodness of fit of a regression model with a standardized time plot of the residuals (Plot 8). Unfortunately, all of the polynomial models are not great models for the unemployment rate data as we can observe cyclical patterns in Plot 8. We can compare the predicted trend values, seasonal values, and predicted values with the historical data:

| Date     | t        | $\hat{T}$  | $\hat{S}$  | $\hat{x}$  |  
|:---------|:---------|:-----------|:-----------|:-----------|
|2018,2    |458       |1.357600    |0.07638777  |1.433987    |
|2018,3    |459       |1.338896    |0.04099895  |1.379895    |
|2018,4    |460       |1.319833    |0.03898064  |1.280852    |
|2018,5    |461       |1.300407    |0.03921716  |1.261189    |
|2018,6    |462       |1.280614    |0.03157355  |1.312187    |
|2018,7    |463       |1.260450    |0.03124312  |1.291693    |
|2018,8    |464       |1.239913    |0.00749424  |1.232419    |
|2018,9    |465       |1.218999    |0.03701035  |1.181989    |
|2018,10   |466       |1.197705    |0.05767968  |1.140025    |
|2018,11   |467       |1.176025    |0.04700606  |1.129019    |
|2018,12   |468       |1.153958    |0.04676550  |1.107193    |
|2019,1    |469       |1.131499    |0.09395024  |1.225450    |

Table 5: Additive decomposition with polynomial regression

Table 5 displays the additive decomposition with 4th order polynomial regression:

$\hat{T}$ is the predicted trend values, $\hat{S}$ is the predicted seasonal values, and $\hat{x}$ is the predicted value.

Polynomial regression can also be applied to time series data, where the relationship between the dependent variable (usually the observed values over time) and the independent variable (time itself) is modeled using a polynomial function. This approach can capture nonlinear trends in the time series data. However, we can see that polynomial regression tends to overfit the data.

### 6 ARIMA Modeling and Forecasting

ARIMA (AutoRegressive Integrated Moving Average) is a popular and powerful time series analysis and forecasting method. It is a combination of autoregressive (AR), differencing (I), and moving average (MA) components, hence the name ARIMA.

We have ultimately decided that we do not need to apply any pre-transformations, since we found that additive decomposition works—as indicated in Section 2. Because the variance in Unemployment Rate is stationary, no transformation is required to ensure the constant variance.

#### 6.1 Assessment of Mean Stationarity

In time series analysis, stationarity is an important property where the statistical properties of a time series (such as mean, variance, and autocorrelation) remain constant over time. Mean stationarity specifically refers to the constancy of the mean of the time series over time.

![image](https://github.com/user-attachments/assets/a5593127-5f55-4251-a05f-f57bf27853cf)

Plot 9: ACF/PACF of Regular, Seasonal, and Seasonal of the Regular Difference

As exemplified in Plot 9, there is seasonality and trend in the mean when we implement a lag of 50 in our ACF and PACF models. In the ACF and PACF of Regular and Seasonal Differencing, we analyze the ACF and PACF of the model alongside regular and/or seasonal differencing in order to identify what method minimizes trend in the mean and make the series stationary.

By seasonally differencing the regular differencing, we notice that the ACF has few significant spikes at the smaller lags before cutting off. Even though we identify a significant spike at lag = 1, we see that the lag quickly dies off. Therefore, we conclude that with the seasonal differencing of the regular differencing of the training data, we achieve mean stationarity.

#### 6.2 ARIMA Model Identification

For model selection, we break the ACF and PACF of seasonal difference of regular difference into regular and seasonal parts. From the ACF and PACF of regular and seasonal differencing (Plot 9), we see that the ACF cuts off at approximately lag 1, and the PACF cuts off at approximately lag 3. Because there is seasonality, we need to identify a model for both the stationarity at low lags and the stationarity at seasonal lags.

As we mentioned, the significant peak cuts off after the third spike in the PACF plot, which tells us that it corresponds to an AR(3). For the seasonal part, we see that the significant spike dies off in both ACF and PACF plots, so we conclude this is an ARMA(1, 1). Therefore, we have decided that the regular differencing can be expressed as an AR(3) and MA(1), while the seasonal differencing can be expressed as AR(1) and MA(1).

The model—in ARIMA notation ARIMA(p,d,q)(P,D,Q)[F]—is: **ARIMA(3,1,1)(1,1,1)[12]**

#### 6.3 ARIMA Fit and Diagnose Analysis (Ljung-Box Test)

In order to fit and diagnose our identified model, we apply the Ljung-Box test using 50 lags—as seen in the ACF (Plot 9). Given the following hypotheses:

$H_0: \rho_1 = \rho_2 = ... = \rho_{50} = 0$

$H_1:$ not all $\rho_k$ up to lag $k$ are 0

We want to test whether or not our detrended and seasonally adjusted Unemployment data are white noise. 
By conducting this test given $\alpha=0.05$, we obtain: **P-value: 0.82536**

As such, because of the P-value > $\alpha=0.05$, we do not reject the null hypothesis that the residuals are white noise. This is good, because the residuals of a well-fitting model should be white noise. Therefore, by the Ljung-Box test, we do not need to restart the model fitting process.

Furthermore, we fit two more models and compare them with our model to see which one is better by analyzing their respective AIC values. The best model is the one with the smallest AIC. The other two models we compare are ARIMA(3,1,0)(1,1,0)[12] and ARIMA(5,1,0)(1,1,0)[12]:

* Model 1: ARIMA(3,1,0)(1,1,0)[12] AIC = -130.2859
* Model 2: ARIMA(5,1,0)(1,1,0)[12] AIC = -131.879
* Model 3: ARIMA(3,1,1)(1,1,1)[12] AIC = -237.1305

By comparing the three models we have randomly selected, we can see that the best model is ARIMA(3,1,1)(1,1,1)[12], because it has the lowest AIC. The model coefficients are:

|      | ar1    | ar2    | ar3    | ma1    | ma2    | sma1   | 
|:-----|:-------|:-------|:-------|:-------|:-------|:-------|
|      | 0.6052 | 0.1334 | 0.0921 | -0.5824| 0.0495 | -0.8786| 
| se   | 0.1353 | 0.0558 | 0.0669 | 0.1293 | 0.0570 | 0.0285 |   

Table 6: Raw data, forecast from models, and RMSE

Hence, our model in the polynomial form—with all AR terms and differencing on the left hand side of the equal (=) sign and all the MA terms on the right hand side—is:

$(1-0.6052B-0.1334B^2-0.0921B^3)(1-B^{12})(1-B)y_t = (1-0.5824B+0.0495B^2)(1-0.8786B^{12})w_t$

Expanding the polynomial version of our mode, the final forecasting equation, with all the independent variables on the right hand side and only the dependent variable at time t on the left-hand-side, is:

$(1-0.6052B-0.1334B^2-0.0921B^3)(1-B^{12})(1-B)y_t = (1-0.5824B+0.0495B^2)(1-0.8786B^{12})w_t$

$y_t(-0.0921B^{16}-0.0413B^{15}-0.4718B^{14}+1.6052B^{13}-B^{12}+ 0.0921B^4+0.0413B^3+0.4718B^2-1.6052B+1) = $
$w_t(1-0.8786B^{12}-0.5824B+0.51169664B^{13}+0.0495B^2-0.0434907B^{14})$


$-0.0921y_{t-16}-0.0413y_{t-15}-0.4718y_{t-14}+1.6052y_{t-13}- y_{t-12}+0.0921y_{t-4}+0.0413y_{t-3}+0.4718y_{t-2}-1.6052y_{t-1} + y_t = $
$w_t-0.8786w_{t-12}-0.5824w_{t-1}+0.51169664w_{t-13}+ 0.0495w_{t-2}-0.0434907w_{t-14}$

**Final Model:**

$y_t = 0.0921y_{t-16}+0.0413y_{t-15}+0.4718y_{t-14}-1.6052 y_{t-13}+y_{t-12}-0.0921y_{t-4}-0.0413y_{t-3}-0.4718y_{t-2}+1.6052y_{t-1}-0.8786w_{t-12}$
$-0.5824w_{t-1}+0.51169664w_{t-13}+0.0495w_{t-2}-0.0434907w_{t-14}$

We want to check whether the model is stationary and invertible, so we find the roots of the MA and AR parts:

**Stationarity (AR roots):**

Finding the modulus of the roots of the polynomial in B: (1,-0.6052,-0.1334,-0.0921) 

=> `1.140201 3.085883 3.085883`

Because the modulus of the roots are greater than 1, the process is stationary.

**Invertibility (MA roots):**

Finding the modulus of the roots of the polynomial in B: (1, -0.5824, 0.0495) 

=> `2.0873509 9.6783056`

Because the modulus of the roots are greater than 1, the process is invertible.

In order to confirm that the coefficients generating the data are not 0, we utilize t-tests. Our hypothesis are shown as follows:

$H_0: \alpha_1 = \alpha_2 = \alpha_3 = \alpha_4 = \alpha_5 = \alpha_6 = 0$

$H_1: \alpha_1, \alpha_2, \alpha_3, \alpha_4, \alpha_5,$ and/or $\alpha_6$ not equal to 0

|      | $\alpha_1$ | $\alpha_1$ | $\alpha_1$ | $\alpha_1$ | $\alpha_1$ | $\alpha_1$ |
|:-----|:-----------|:-----------|:-----------|:-----------|:-----------|:-----------|
| value | 0.6052 | 0.1334 | 0.0921 | -0.5824 | 0.0495 | -0.8786 |
| se   | 0.1353 | 0.0558 | 0.0669 | 0.1293 | 0.0570 | 0.0285 |   
|$t_{n-k}$ | 4.47302 | 2.39068 | 1.37668 | 4.50425 | 0.86842 | 30.82807 |

Table 7: Raw data, forecast from models, and RMSE

We reject the null hypotheses in all cases, because the t-statistics are more than 2 standard errors away from the center. Thus, there is statistically evidence suggesting that the coefficients of the AR and MA models generating the data—labeled alpha—are not 0.

#### 6.4. ARIMA Forecasting

After confirming that the residuals are white noise, the model is stationary and invertible, and the model coefficients are significantly different from 0, we proceed to use the model we have selected in Section 6.2 to forecast the future values of the series. A stationary time series is one whose statistical properties—such as variance and autocorrelation—remain constant over time. An invertible model ensures that the estimated coefficients provide meaningful insights.

When fitting an ARIMA model, it is crucial to ensure that the resulting model is both stationary and invertible. This can be achieved by examining diagnostic plots, conducting statistical tests, and checking the estimated coefficients. If the model fails to meet these criteria, adjustments may need to be made. We can plot the forecast in both the testing period and the entire date range.

![image](https://github.com/user-attachments/assets/1ee2b343-f27b-4144-a45c-07c10622fcc0)

Plot 10: Entire Forecast using ARIMA(3,1,1)(1,1,1)[12]

First and foremost, we plot the forecast—first with the whole time series, before plotting just the forecast period. From Plot 10 and Plot 11, we see that the forecast (red dotted line) appears to follow the raw unemployment data (black line) fairly closely, with the upper and lower bands (blue dotted lines) providing a sufficient buffer for the prediction.

![image](https://github.com/user-attachments/assets/75c5469c-0aa4-48fc-b665-6d25e67604f7)

Plot 11: Forecast Period using ARIMA(3,1,1)(1,1,1)[12]

Next, we construct confidence intervals that inform us as to how confident we are that the actual value of the series in the future lies in the particular interval. Thus, with our final model—ARIMA(3,1,1)(1,1,1)[12]—containing normally-distributed residuals, we forecast the test period of our original raw y variable (`Unemployment`) to obtain point and prediction intervals. By doing so, we print the data frame with raw test values from `Unemployment`, the forecast, the forecast interval (CI Low, CI High) and standard error of our forecast.

| Test | CI Low         | Forecast Value | CI High        | Forecast SE    |
|:-----|:---------------|:---------------|:---------------|:---------------|
| 4.4  | 3.9885357      | 4.3393331      | 4.6901305      | 0.17897825     |
| 4.1  | 3.5827438      | 4.0845305      | 4.5863172      | 0.25601362     |
| 3.7  | 2.8913789      | 3.5396347      | 4.1878905      | 0.33074274     |
| 3.6  | 2.7654800      | 3.5691688      | 4.3728577      | 0.41004533     |
| 4.2  | 3.0167901      | 3.9750120      | 4.9332340      | 0.48888873     |
| 4.1  | 2.9065643      | 4.0181760      | 5.1297878      | 0.56714886     |
| 3.9  | 2.5231823      | 3.7867982      | 5.0504140      | 0.64470195     |
| 3.6  | 2.0609218      | 3.4743871      | 4.8878525      | 0.72115578     |
| 3.5  | 1.7703723      | 3.3311271      | 4.8918819      | 0.79630347     |
| 3.5  | 1.5895574      | 3.2947801      | 5.0000029      | 0.87001161     |
| 3.7  | 1.4740907      | 3.3207743      | 5.1674579      | 0.94218553     |
| 4.4  | 2.0286240      | 4.0136551      | 5.9986862      | 1.01277097     |

Table 8: Forecast of ARIMA(3,1,1)(1,1,1)[12]

We can see that the forecasted values tend to be rather close to the raw test values, and are all within 1 SE of the actual value. Furthermore, the test values are within the prediction interval. As with the exponential smoothing model, as the months increase, the forecast grows farther from the test value; however, the actual value of the series in the future still lies in the particular interval. In conclusion, ARIMA(3,1,1)(1,1,1)[12] provides an accurate forecast of the test data.

We measure the accuracy of our forecast using the root mean square error (RMSE) statistic. Our final calculated MSE is: **0.56367369**. Because RMSE values between 0.2 and 0.5 are able to relatively predict the data accurately, we are satisfied with the capabilities of our model—ARIMA(3,1,1)(1,1,1)[12].

Finally, besides the other assumptions we have regarding the residuals $w_t$ having mean a mean of 0, a constant variance at all $t$, and 0 correlation, we need to check that it is normally distributed in order to conduct accurate statistical inference.

![image](https://github.com/user-attachments/assets/1286b2fa-bb91-4c46-b59a-7ddd700ecd49)

Plot 12: Histogram of ARIMA(3,1,1)(1,1,1)[12] Residuals

We want to check that the residuals for our final model, ARIMA(3,1,1)(1,1,1)[12], are normally distributed; as such, we plot a histogram (Plot 12) to confirm that this is the case. Because the histogram appears to be normal, we are able to proceed with the forecast we have created.

### 7 Multiple Regression with ARMA Residuals

Multiple regression with ARMA (AutoRegressive Moving Average) residuals is a statistical modeling technique that combines elements of multiple linear regression and ARMA time series modeling. This approach is used when the residuals from a multiple regression model exhibit autocorrelation or other time series patterns that can be better captured by an ARMA model.

#### 7.1 ARMA Causal Model Fit

A causal model fit refers to the process of fitting a statistical model that attempts to establish causal relationships between variables. In a causal model, one variable (the independent or predictor variable) is hypothesized to directly influence another variable (the dependent or response variable), while controlling for other potential confounding variables.

* Overall, fitting a causal model involves a systematic process of evaluating causal relationships between variables.

To fit a multiple regression model with our variables, we first split the three time series into training and testing data. After that, we fit a multiple regression model with unemployment rate as the dependent variable, and housing supply ratio and federal funds effective rate as the two independent variables. Analyzing the autocorrelation function (ACF) and partial autocorrelation function (PACF) of the residuals of a multiple linear regression model can provide valuable insights into whether the residuals exhibit any remaining autocorrelation after fitting the model.

![image](https://github.com/user-attachments/assets/68c73298-b159-4470-a0a4-8513a67f1d51)

Plot 13: ACF and PACF of residuals of multiple linear regression model

We observe a cyclical pattern from the ACF and PACF plots (Plot 13), so we further determine that the residuals are not white noise. Hence, we attempt to fit an ARMA model which we can get the coefficients of to feed into a GLS model.

![image](https://github.com/user-attachments/assets/f3967045-b4a0-4ce2-82b4-00bb4e5202af)

Plot 14: Residuals plot and ACF plot of residuals of AR(1) model

After much trial and error, we selected the ARMA model with the best AIC value, AR(16). We find that this model is stationary when we observe the residuals to be white noise and as shown in the ACF plot (Plot 14). We then attempted to feed the coefficients we found into a GLS model, but were initially unable as not all the coefficients were less than 1. This meant that our model was not invertible, and hence could not be used for calculating a GLS model.

We repeated this process multiple times with different ARMA models that appeared to have white noise residuals, and could not determine a set of coefficients that were all less than 1. Ultimately, we only used the fact that the AR model was of order 16 as a parameter in the GLS function in R without inputting our own coefficients. The GLS function was able to calculate its own coefficients.

* Plot 15 displays the residuals of the GLS model and the ACF of the GLS residuals:

![image](https://github.com/user-attachments/assets/c1606a10-c37c-4a5f-b6b7-747a2ed6d12f)

Plot 15: Residuals and ACF plot of GLS model

The formula that we found is the following where:

* $x_{1t}$ is housing supply ratio at time $t$ and $x_{2t}$ is federal funds effective rate at time $t$:

$y_t = 6.480278 - 0.036687 x_{1t} - 0.024086 x_{2t} + e_t$  where $e_t$ is the following:

$e_t = 0.94761954 e_{t-1} + 0.136519195 e_{t-2} + 0.005222809 e_{t-3} - 0.064356943 e_{t-4}$
$+ 0.048716685 e_{t-5} - 0.133601897 e_{t-6} + 0.101303293 e_{t-7} - 0.027352147 e_{t-8}$
$- 0.149792849 e_{t-9} + 0.089271534 e_{t-10} + 0.035347094 e_{t-11} + 0.687864584 e_{t-12}$
$- 0.643179964 e_{t-13} - 0.142164182 e_{t-14} - 0.127062788 e_{t-15} + 0.217739528 e_{t-16}$

We have calculated the intercept standard error as: **0.5313121**

$x_{1t}$ standard error: **0.0142313**

$x_{2t}$ standard error: **0.0125550**

Plot 16 shows the plot of the training data and the forecasted values. 

The RMSE of the GLS model on the test data is **2.328397**.

![image](https://github.com/user-attachments/assets/0b3eec7e-9f02-4158-a7f0-7b8fc69c9159)

Plot 16: Plot of Unemployment Rate, GLS fitted values, and forecast

### 8 Vector Autoregression

Vector Autoregression (VAR) is a multivariate time series forecasting model used to analyze the dynamic relationships among multiple time series variables. Unlike traditional univariate time series models that focus on predicting a single variable, VAR models jointly model the behavior of multiple variables over time.

#### 8.1 CCF and Degree for Vector Autoregression

The cross-correlation function (CCF) is a statistical tool used to measure the relationship between two time series variables by calculating the correlation between their lagged values. It helps to identify the extent and direction of the linear relationship between two series, including any time lags in the relationship. In order to conduct CCF (cross-correlation function), we need to make sure the time series data we input are stationary. We first draw the ACF and PACF for both of our independent variables: Federal Reserve Effective Rate and Housing Supply Ratio.

However, before that, we need to split the data for independent variables into training and testing sets in order to measure the accuracy of our model. Just as what we did to the dependent variable, for each of the two independent variables, we use the data from `1980-01` to `2018-01` as the training data and that from `2018-02` to `2019-1` as the testing data.

For the unemployment training data, we have already concluded in Section 6.2 that we will apply seasonal differences and regular differences to it.

Thus, ${unemployment}_t = (1-B^{12})(1-B) {unemployment}_t$

Next, we take a look at the ACF and PACF for the housing supply ratio training data (Plot 17).

The data is obviously not mean-stationary.

![image](https://github.com/user-attachments/assets/f040b59f-e6f5-4e57-9515-c4229a1feb24)

Plot 17: ACF and PACF of Housing Supply Ratio training data

Because the data is not mean-stationary, it suggests that the mean of the time series is not constant over time, violating one of the key assumptions of time series analysis. In this case, it is important to address the non-stationarity before proceeding with any further analysis or modeling

We then try the regular difference, seasonal difference, and the seasonal difference of the regular difference of the training data:

* Regular differencing involves taking the difference between consecutive observations in the time series (effective for removing trend components from the data)
* Seasonal differencing involves taking the difference between observations separated by the seasonal period (helps remove seasonal patterns from the data)
* The seasonal difference of the regular difference combines regular differencing and seasonal differencing (data exhibits both trend and seasonality that need to be removed)

By applying these differencing techniques, non-stationary time series data can be transformed into a stationary form. We can see from Plot 18 that when we apply seasonal difference to the regular difference to the training data, we achieve stationarity.

Thus, for housing supply ratio, we decide to use the following formula:

${housing.supply.ratio}_t = (1-B^{12})(1-B) {housing.supply.ratio}_t$

![image](https://github.com/user-attachments/assets/f743bf71-787d-4969-98a8-3e1b5ce164c0)

Plot 18: ACF and PACF of all 3 differencing applied to Housing Supply Ratio training data

![image](https://github.com/user-attachments/assets/b3e40137-93c1-402a-a877-cb1ec33da9eb)

Plot 19: ACF and PACF of Differencing applied to Federal Reserve Effective Rate Training Data

Finally, for the federal reserve effective rate, from Plot 19, we believe it is sufficiently enough just to apply regular differences to achieve stationarity.

Thus, we obtain: ${fed.reserve.rate}_t = (1-B) {fed.reserve.rate}_t$

After figuring out the differencing we should apply to the training data of both our dependent and independent variables to make them stationary, next we will study their cross-correlation. Before that, we need to combine all of our differenced data into a single object.

![image](https://github.com/user-attachments/assets/f2141403-d828-44bd-b314-0355fa3e278c)

Plot 20: CCF between Unemployment and Housing Supply Ratio training data

From Plot 20, we see the cross-correlation between unemployment and housing supply ratio. Based on this, we can find the VAR model between them. Since we care about how housing supply ratio affects unemployment, we look at the top right graph in Plot 20. We see that the first significant spike occurs at lag 6 (excluding lag 0).

In addition, that significant spike decays right away, so we know unemployment at time t depends on housing supply ratio at time (t-6). Next, we see how unemployment is related to itself. We see that CCFs die away in a damped sine-wave fashion, so we know unemployment at time t depends on unemployment at time t-1 and unemployment at time t-2.

Writing this in formula form, we have:

${y_t}^* = a_1 {x_{1,t-6}}^* + a_2 {y_{t-1}}^* + a_3 {y_{t-2}}^*$

* where ${y_t}^*$ denotes the differenced unemployment data

* where ${x_{1,t}}^*$ denotes differenced housing supply ratio data

When we look at how the housing supply ratio is affected by unemployment, we focus on the lower left graph in Plot 20. We see that the first significant spike occurs at lag 1 (excluding lag 0). In addition, that significant spike decays right away, so we know housing supply ratio at time t depends on unemployment at time (t-1).

Next, we see how the housing supply ratio is related to itself. We see that CCFs die away in a damped sine-wave fashion, so we know housing supply ratio at time t depends on housing supply ratio at time t-1 and housing supply ratio at time t-2.

Writing this in formula form, we have:

${x_{1,t}}^* = a_1 {y_{t-1}}^* + a_2 {x_{1,t-1}}^* + a_3 {x_{1,t-2}}^*$

That is to say, we fit a VAR(6) model between unemployment and housing supply ratio with:

1. ${x_{1,t}}^* = a_1 {y_{t-1}}^* + a_2 {x_{1,t-1}}^* + a_3 {x_{1,t-2}}^*$   

2. ${y_t}^* = a_1 {x_{1,t-6}}^* + a_2 {y_{t-1}}^* + a_3 {y_{t-2}}^*$

![image](https://github.com/user-attachments/assets/071a2edf-fa69-45ca-93fe-48f932a301b2)

Plot 21: CCF between Unemployment and Federal Reserve Effective Rate training data

From Plot 21, we see the cross-correlation between unemployment and federal reserve effective rate. Based on this, we can find the VAR model between them. Since we care about how the federal reserve effective rate affects unemployment, we look at the top right graph in Plot 21.

We see that the first significant spike occurs at lag 1 (excluding lag 0). In addition, that significant spike decays right away, so we know unemployment at time t dependents on federal reserve effective rate at time (t-1). Next, we see how unemployment is related to itself. We see that CCFs die away in a damped sine-wave fashion, so we know unemployment at time t depends on unemployment at time t-1 and unemployment at time t-2.

Writing this in formula form, we have:

${y_t}^* = a_1 {x_{2,t-1}}^* + a_2 {y_{t-1}}^* + a_3 {y_{t-2}}^*$

When we look at how the federal reserve rate is affected by unemployment, we focus on the lower left graph in Plot 21. We see that the first significant spike occurs at lag 1 (excluding lag 0). In addition, that significant spike decays right away, so we know the reserve rate at time t depends on unemployment at time (t-1).

Next, we see how the federal reserve rate is related to itself. We see that CCFs die away in a damped sine-wave fashion, so we know federal reserve rate at time t depends on federal reserve rate at time t-1 and federal reserve rate at time t-2.

Writing this in formula form, we have:

${x_{2,t}}^* = a_1 {y_{t-1}}^* + a_2 {x_{2,t-1}}^* + a_3 {x_{2,t-2}}^*$   

* where ${x_{2,t}}^*$ refers to differenced federal reserve rate training data

This means that we need to fit a VAR(2) model between unemployment and federal reserve effective rate. By looking at both Plot 20 and Plot 21, we can see that unemployment is leading since it is significant at lag 1 in the lower left graph, while in the upper right graph, housing is not significant until lag 6.

In Plot 21, we see that in both the upper right and lower left graph, lag 1 is significant, but the lower left graph has a more significant spike than the upper right graph, meaning unemployment is more “leading” than federal reserve rate.

#### 8.2 Constructing the Vector Autoregression Model

In this section, we fit the VAR(6) and VAR(2) models:

| Unemployment Rate      | Estimate     |                         | Estimate    | 
|:-----------------------|:-------------|:------------------------|:------------|
|Unemployment.Rate.l1    | -0.057276934 | Unemployment.Rate.l4    | 0.149502297 |
|Housing.Supply.Raio.l1  | 0.036779918  | Housing.Supply.Ratio.l4 | 0.028874474 |
|Unemployment.Rate.l2    | 0.138770027  | Unemployment.Rate.l5    | 0.143440787 |
|Housing.Supply.Ratio.l2 | 0.060761740  | Housing.Supply.Ratio.l5 | 0.035424376 |
|Unemployment.Rate.l3    | 0.222121689  | Unemployment.Rate.l6    | 0.052098039 |
|Housing.Supply.Ratio.l3 | 0.040446479  | Housing.Supply.Ratio.l6 | 0.044845204 |
|constant                | 0.001448732  |                         |             |

Table 9: Raw data, forecast from models, and RMSE — VAR(6) Model

| Housing Supply Ratio   | Estimate     |                         | Estimate     | 
|:-----------------------|:-------------|:------------------------|:-------------|
|Unemployment.Rate.l1    | -0.628678932 | Unemployment.Rate.l4    | 0.061052072  |
|Housing.Supply.Raio.l1  | -0.247767338 | Housing.Supply.Ratio.l4 | -0.153412526 |
|Unemployment.Rate.l2    | -0.478534543 | Unemployment.Rate.l5    | -0.121499075 |
|Housing.Supply.Ratio.l2 | -0.188153630 | Housing.Supply.Ratio.l5 | -0.048113699 |
|Unemployment.Rate.l3    | 0.058333071  | Unemployment.Rate.l6    | -0.534840301 |
|Housing.Supply.Ratio.l3 | 0.045247464  | Housing.Supply.Ratio.l6 | -0.066892020 |
|constant                | -0.008287217 |                         |              |

Table 10: Raw data, forecast from models, and RMSE — VAR(6) Model

With the information about the model coefficients, we conclude the system is:

${x_{1,t}}^* = -0.2478 {x_{1,t-1}}^* - 0.1882 {x_{1,t-2}}^* + 0.04524 {x_{1,t-3}}^* $
$0.1534 {x_{1,t-4}}^* - 0.04811 {x_{1,t-5}}^* - 0.0669 {x_{1,t-6}}^* - 0.6287 {y_{t-1}}^* $
</br>$- 0.4785 {y_{t-2}}^* + 0.05833 {y_{t-3}}^* + 0.06105 {y_{t-4}}^* - 0.1215 {y_{t-5}}^* - 0.5348 {y_{t-6}}^*$
$-0.00829$

${y_t}^* = -0.05728 {y_{t-1}}^* + 0.13877 {y_{t-2}}^* + 0.2221 {y_{t-3}}^* + 0.1495 {y_{t-4}}^* $
$+ 0.14344 {y_{t-5}}^* + 0.05210 {y_{t-6}}^* + 0.03677 {x_{1,t-1}}^* + 0.06076 {x_{1,t-2}}^* $
</br>$+ 0.04044 {x_{1,t-3}}^* + 0.02887 {x_{1,t-4}}^* + 0.04811 {x_{1,t-5}}^* + 0.04485 {x_{1,t-6}}^*$
$+0.001449$

| Unemployment Rate      | Estimate     |  Federal Funds Effective Rate  | Estimate     | 
|:-----------------------|:-------------|:-------------------------------|:-------------|
|Unemployment.Rate.l1    | 0.034332129  | Unemployment.Rate.l1           | -0.146505553 |
|Fed.Effective.Rate.l1   | -0.068430632 | Fed.Effective.Rate.l1          | 0.535638790  |
|Unemployment.Rate.l2    | 0.162287671  | Unemployment.Rate.l2           | 0.007378826  |
|Fed.Effective.Rate.l2   | 0.041075519  | Fed.Effective.Rate.l2          | -0.175693686 |
|constant                | -0.003645138 | constant                       | -0.020028198  |

Table 11: Raw data, forecast from models, and RMSE — VAR(2) Model

With the information about the model coefficients, we conclude the system is:

* $x_{2,t} = 0.5356 x_{2,t-1} - 0.1757 x_{2,t-2} - 0.1465 y_{t-1} + 0.00738 y_{t-2} - -0.02003$

* $y_t = 0.0343 y_{t-1} + 0.1623 y_{t-2} - 0.0684 x_{2,t-1} + 0.0411 x_{2,t-2} - 0.003645$

The Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots are essential tools for understanding the autocorrelation structure of a time series. They provide insights into the dependence between observations at different lags.

![image](https://github.com/user-attachments/assets/2068d5fd-5e76-4c04-b6f0-a160967e7631)

Plot 22: ACF of Residuals for VAR(6)

By looking at the residuals of both our VAR(6) and VAR(2) models in Plot 22 and Plot 23, since the residuals are almost bivariate white noise, the assumption for the VAR model is validated.

![image](https://github.com/user-attachments/assets/c0fc1cc0-2134-4704-8e3e-995d7a357a43)

Plot 23: ACF of Residuals for VAR(2)

#### 8.3 Impulse Response Functions for Vector Autoregression Model

In this section, we construct and plot the impulse response functions to see the effect and length of the effect of a shock to the system. It allows for the decomposition of the response of endogenous variables to a shock into orthogonal components

![image](https://github.com/user-attachments/assets/cabd3a5c-5f88-4c5a-9f1a-4b6be3ae3941)

Plot 24: Orthogonal Impulse Response for Unemployment Rate

In Plot 24, we can see how Unemployment rate and Housing Supply Ratio respond to an impulse in Unemployment rate. Both have a large response, and the equilibrium is achieved after t = 4. Impulse Response Functions (IRFs) are a key tool in time series analysis, particularly in the context of vector autoregressive (VAR) models.

![image](https://github.com/user-attachments/assets/6daa1173-2b89-4467-9967-5495780ce81b)

Plot 25: Orthogonal Impulse Response for Housing Supply Ratio

In Plot 25, we can see how the Unemployment Rate and Housing Supply Ratio respond to an impulse in Housing Supply Ratio. Unemployment does not respond much to the shock, while the Housing Supply Ratio responds dramatically to the impulse. Therefore, Equilibrium is achieved after t = 4.

![image](https://github.com/user-attachments/assets/fa78e7c1-364f-4f98-8c14-1ac9cc4dbfdf)

Plot 26: Orthogonal Impulse Response from Unemployment Rate

In Plot 26, we can see how Unemployment Rate and Federal Reserve Effective Rate respond to an impulse in Unemployment rate. Clearly, the Unemployment Rate responds drastically to the impulse, but the Federal Reserve Effective Rate responds more mildly. In conclusion, both of them reach equilibrium at t = 10.

![image](https://github.com/user-attachments/assets/f1c0cee0-0229-497f-9c43-742c579e7bfe)

Plot 27: Orthogonal Impulse Response for Federal Funds Rate

Finally, in Plot 27, we can see how the Unemployment rate and Federal Reserve Effective Rate respond to an impulse in Federal Reserve Effective Rate. The Unemployment Rate does not respond much to the impulse, but the Federal Reserve Effective Rate responds quite dramatically. Both of them reach equilibrium at t = 10.

Orthogonal Impulse Response (OIR) analysis is a method used in econometrics and macroeconomic modeling to study the effects of a shock (or impulse) in a vector autoregressive (VAR) model. It allows for the decomposition of the response of endogenous variables to a shock into orthogonal components. Impulse Response Functions provide insights into the dynamic interactions between variables in the system

Impulse Response Functions are tools for analyzing the dynamic behavior of multivariate time series data and understanding the responses of variables to shocks in the system.

#### 8.4 Forecasting using the Vector Autoregression Model

After figuring out the VAR models, we can use them to make predictions. In particular, we are interested in how well the VAR(6) and VAR(2) models can predict the unemployment rate.In Plot 28 and Plot 29, we draw the prediction together with the confidence interval for the prediction for both VAR(6) and VAR(2).

![image](https://github.com/user-attachments/assets/67007b9f-febd-4d1c-8d8a-753b18e717a0)

Plot 28: Forecast of Change in Unemployment with VAR(6) of Housing Supply Ratio

If we calculate the RMSE, we have the RMSE for the VAR(6) model being 0.4184812, and the RMSE for the VAR(2) model being 0.4452308. By comparing the RMSE, we can see the VAR(6) model predicting Unemployment with Housing Supply Ratio is a better model than the VAR(2) model predicting Unemployment with Federal Reserve Effective Rate.

![image](https://github.com/user-attachments/assets/dedd46cb-f69e-47a6-8da4-b77a58ee5496)

Plot 29: Forecast of Change in Unemployment with VAR(2) of Federal Reserve Rate

### 9. Forecast Cross-Validation, Final Conclusion

In conclusion, we found in the time series that the dependent variable—unemployment rate—contains the seasonality that does not increase with the trend. These two features indicate that the seasonality of unemployment does not increase with the trend, allowing us to utilize additive decomposition rather than multiplicative decomposition. Hence, after plotting the additive decomposition outcomes, it is clear that the random term fluctuates around the value of 0—signifying that our data has constant variance.

Using seasonal Holt-Winters exponential smoothing, we were able to relatively accurately forecast data in 2019 by constructing a training dataset containing data before 2019, and comparing it with a test dataset containing only 2019 data. As such, the comparison appears to show that the forecast data matches fairly well with the test set of the raw data.

The RMSE for the exponential smoothing model is 0.2632173, the RMSE for the polynomial regression model is 1.222444, the RMSE for the GLS model is 2.328397, the RMSE for the ARIMA model is 0.5637, the RMSE for the VAR model is 0.41848, and the average forecast RMSE is 0.263217 (Table 12). We can say that the exponential smoothing model performed the best so far in fitting the unemployment rate data and the multiple linear regression model (GLS) the worst so far.

Exponential smoothing is particularly effective for data with a trend and/or seasonal patterns. It can adapt to changes in the underlying data more easily than polynomial regression, which might require adjustments to the degree of the polynomial or the inclusion of additional terms. Exponential smoothing is generally more robust to outliers compared to polynomial regression.

Notice that we did not include the VAR models in Plot 26. The reason is that we have applied different differencing methods to our dependent variable and independent variables to make them stationary before fitting the VAR models. That is why Plot 28 and Plot 29 look different from the raw time plot in Plot 2 even though they are all time plots for unemployment rate. The only difference is that Plot 28 and Plot 29 show the differenced unemployment rate. That is to say, the testing data for dependent variables and independent variables have been differenced.

![image](https://github.com/user-attachments/assets/015f9155-eef8-4177-ab67-6df9efa7e284)

One should notice that the Diff Raw Data column in Table 12 are completely different from the Raw Data Values since the former one is the differenced version of the latter one. That is why we choose not to plot VAR in Plot 34. If we really want to do so, we have to undo the differencing, which is quite laborious. In Table 13, we have attached the VAR prediction from our VAR(6) since it has a smaller RMSE than the VAR(2) model.











