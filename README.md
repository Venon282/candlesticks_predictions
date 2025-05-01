"# candlesticks_predictions" 

## Model parameters for bellow tests
Model:
| num_layers | num_heads | dff | d_model | dropout_rate |
| ---------- | --------- | --- | ------- | ------------ |

Training:
| es_patience | batch_size | epochs | warmup_steps |
| ----------- | ---------- | ------ | ------------ |


Model size:
| encoder | decoder | final | parameter number |
| ------- | ------- | ----- | ---------------- |

Datas:
| train | val | test |
| ----- | --- | ---- |

## Raw Datas:

### Choice
| Market                    | Description                                                                                                          | 👍 Advantages                                                                                              | 👎 Disadvantages                                                                                              | Prediction Suitability |
|:--------------------------|:---------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------|:----------------------:|
| **Equities (Stocks)**     | Shares of individual companies, driven by earnings, M&A, corporate news                                             | – Long daily history<br>– Sector trends<br>– High data volume                                             | – Abrupt jumps on earnings/news<br>– Idiosyncratic shocks                                                     | ✓                      |
| **Forex (FX)**            | Currency pairs traded 24/5, influenced by central banks, macro data, algos                                         | – Very high liquidity<br>– Continuous trading → fewer gaps<br>– Repeatable carry/momentum patterns         | – High HFT noise<br>– Sudden regime shifts on policy announcements                                           | ✓✓                     |
| **Commodities**           | Raw materials (oil, gold, grains), subject to supply/demand, weather, geopolitics                                  | – Seasonal cycles (agri, energy)<br>– Macro correlations (USD, inflation)                                 | – Geopolitical shocks<br>– Non‑linear spikes<br>– Limited history for some metals                              | ✗                      |
| **Cryptocurrencies**      | Decentralized digital assets (BTC, ETH), driven by sentiment, network effects, regulation                          | – 24/7 data → dense history<br>– Clear bull/bear cycles<br>– Sentiment‑based algorithmic patterns        | – Extreme volatility<br>– Forks/regulatory bans                                                              | ✗                      |
| **Index Futures**         | Futures on baskets (S&P 500, DAX), reflecting broad market trends and fund flows                                   | – Less idiosyncratic noise than single stocks<br>– Clear long‑term trends<br>– High liquidity            | – Sensitive to macro releases (inflation, rates)                                                            | ✓✓✓                    |



| Field   | Definition                                                  | Technical Detail / Example                                                  |
| ------- | ----------------------------------------------------------- | --------------------------------------------------------------------------- |
| date    | Date of the candlestick                                     | Example: 2025-04-22                                                         |
| time    | Time of the candlestick                                     | Example: 14:30:00 — often in UTC or platform-local time                     |
| open    | Opening price of the time period                            | First recorded price within the interval                                    |
| high    | Highest price reached during the time period                | May indicate a spike or breakout attempt                                    |
| low     | Lowest price reached during the time period                 | Opposite of high, helps measure range                                       |
| close   | Closing price of the time period                            | Last recorded price in the interval                                         |
| tickvol | Number of price changes (ticks) during the time period      | If the price changed 150 times → tickvol = 150                              |
| vol     | Actual trading volume during the time period (if available) | Example: 20,000 shares or contracts, depending on the asset                 |
| spred   | Spread between bid (buy) and ask (sell) prices              | Typically: ask - bid, reflects market liquidity and hidden transaction cost |

**Notes:**
- open, high, low, close = price structure used to build candlesticks.
- tickvol ≠ vol: one counts activity (number of ticks), the other measures actual volume traded.
- spread can be:
    - Fixed (some brokers),
    - Variable (common in Forex and crypto),
    - Widens during illiquid or volatile conditions.

**Processing**
Date and time are drop. They are not usefull as the time step is regular



















## Tests


In my datas, i have in informations: 

Date and time is not usefull to know as the step is always the same so drop.  
Volume is mainly 0 due to the fact that my datas are forex one. So the volume is not know.
Tick

    
The prediction is made on only one candle because the indicator in this transformer model are predicted.  
A best and futur way, once selected, they will be recalculate inside the model.  

The datas use

The complexity model use:

single test
none | without_volume | without_spread |rsi | macd | bollinger

duo test
rsi_macd | rsi_bollinger | macd_bollinger

triple test
rsi_macd_bollinger