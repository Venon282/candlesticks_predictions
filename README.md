"# candlesticks_predictions"  
  
# Presentation  
  
This model have for objectif to predict the n futur candlesticks based on the m previous.  
A lot of paper already exist on the subject.  
Since the arrival of the transformers, this approach has been called into question.  
The results were never really interesting. But have Transformers changed the way things are done ?  
I'm not a researcher looking for publications, just a slightly curious engineer with no great talent for writing, as you'll discover below.  
My code is public and bad results will be mentioned.  
  
# Data preparation  
The features use will be study in a next part. Here the study focus on raw preparation.  
  
## Dataset  
For each market timestep is process:  
- Drop undesire cols/features  
- Ensure the data type of each columns  
- Add desire features (indicators, time notions, etc)  
- Some of this features need historic for be calculate. As they add nan value at the start, drop this rows.  
  
## Data  
**sequences**  
The build transformer can accept different size of input and output size to generate.  
For create the sequence we can choose a number of candle minimal and maximal for an input and output sequence.  
With a specific step (automatic or define), it allow the model to see different input/output sequence len.  
The reproductibility of the sequence is guarantee with a seed.  
  
Once done the differents markets are split into the legendary three sets (train, val, test).  
Then still for each markets, fit a StandardGlobalScaler on the train sets and transform them all with it.  
  
## StandardScaler  
A StandardScaler is a data‐preprocessing tool that performs z-score normalization.  
It transforms each feature so that it center the datas around 0 with a standard deviation of 1.  
  
$x_{scaled} = \frac{x-\mu}{\sigma}$  
where $\mu$ is the mean of that feature over the fitting data, and $\sigma$ is its standard deviation.  
### Pros  
**Numerical Stability**  
Centering each feature at zero mean and unit variance reduces the risk of exploding or vanishing gradients, especially important when computing attention scores.  
  
**Faster Convergence**  
Optimizers like Adam and SGD typically converge more quickly when inputs are zero-centered and similarly scaled.  
  
**Consistent Attention Magnitudes**  
Transformers rely on dot-product attention; having inputs on a consistent scale prevents some heads from dominating and stabilizes learning.  
  
### Cons  
**Sensitivity to Outliers**  
Since it uses the dataset’s global mean and standard deviation, extreme spikes (e.g., flash crashes) can skew the scaling.  
  
**Non-Stationarity Challenges**  
If the market regime shifts (bull vs. bear), a scaler computed on past data may become suboptimal. Will need to re-fit or use a rolling/windowed approach.  
  
| **Scaler**     | **Effect on Input**     | **Transformer Impact**                                | **Best When…**                                 | **Drawbacks**                                       |
| -------------- | ----------------------- | ----------------------------------------------------- | ---------------------------------------------- | --------------------------------------------------- |
| StandardScaler | Z-score → mean 0, var 1 | Stable dot-product scales; faster, balanced gradients | Features ≈ Gaussian (e.g. log-returns, deltas) | Outliers can skew μ/σ                               |
| MinMaxScaler   | Linear → \[0,1]         | Small uniform magnitudes; may need LR tuning          | Data has fixed bounds, few new extremes        | Evolving min/max; extreme sensitivity to outliers   |
| RobustScaler   | Median/IQR centering    | Attention less influenced by spikes; robust to tails  | Data with frequent fat tails or spikes         | Slower fine-pattern learning on Gaussian-like data  |
| None           | Raw units               | Unbalanced attention; unstable gradients              | Quick prototyping only                         | Poor convergence; dominates by large-scale features |
  
## What is StandardGlobalScaler ?  
It's a scaler that instead fit on each columns, it will process a global fit.  
The cons is that it need more than one scaler for each markets.  
The pros is that i can regroup columns to have the same fit as the price (open, high, low, close) or indicators which maintain coherence between columns  

# Loss

# Learning Rate Scheduler Noam
The Noam learning rate scheduler is commonly used with Transformer models and is defined by:  
$$
\text{lr}(step) = d_{\text{model}}^{-0.5} \cdot \min\left(step^{-0.5}, \text{step} \cdot \text{warmup\_steps}^{-1.5} \right)
$$

This schedule has two main phases:
* **Warm-up phase**: The learning rate increases linearly for a number of `warmup_steps`.
* **Decay phase**: After the warm-up, the learning rate decays proportionally to $step^{-0.5}$.

This behavior helps stabilize training early on and encourages faster convergence later.

### Learning Rate as a Function of Training Steps

<div style="display: flex; justify-content: center;">
  <div style="text-align: center; margin: 10px;">
    <img src="./img/n_inputs_train=2603825_epochs=50_batch_size=32.png" /><br/>
  </div>
</div>

#### Observations from the Plot:

* **Effect of `warmup_steps`**:

  * Increasing the number of warm-up steps **extends** the rising phase.
  * A higher `warmup_steps` ratio leads to a **higher peak learning rate** but delays when it's reached.
  * Eventually, all configurations decay to similar values as training progresses.

* **Effect of `d_model` (model dimensionality)**:

  * A higher `d_model` results in a **lower overall learning rate**, since it's scaled by $d_{\text{model}}^{-0.5}$.
  * This makes training more stable but possibly slower in convergence if not properly tuned.

* **Impact of training setup (`batch_size`, `epochs`)**:

  * Increasing the **number of epochs** spreads the training steps over a longer period, allowing for finer convergence and a lower final learning rate.
  * Increasing the **batch size** effectively reduces the number of optimization steps per epoch, indirectly increasing the learning rate due to faster accumulation of gradients.

#### Takeaway:

Proper tuning of `d_model`, `warmup_steps`, `batch_size`, and `epochs` is **crucial**. These parameters strongly influence the learning rate schedule, which in turn has a **direct impact on model performance and convergence stability**.
  
# Model parameters for bellow tests  
Model:  
| num_layers | num_heads | dff  | d_model | dropout_rate |
| ---------- | --------- | ---- | ------- | ------------ |
| 8          | 8         | 1024 | 512     | 0.1          |
  
Training:  
| es_patience | batch_size | epochs | warmup_steps |
| ----------- | ---------- | ------ | ------------ |
| 10          | 128        | 50     | 20345        |
  
Model size:  
| encoder    | decoder    | final | parameter number |
| ---------- | ---------- | ----- | ---------------- |
| 16,825,344 | 25,238,528 | 2,565 | 42,066,437       |
  
Datas:  
| -           | train            | val             | test           |
| ----------- | ---------------- | --------------- | -------------- |
| **input**   | (2604163, 40, -) | (139259, 40, -) | (41795, 40, -) |
| **outputs** | (2604163, 1, -)  | (139259, 1, -)  | (41795, 1, -)  |
The prediction is made on 1 candle because in autoregressif mode, it will take the previous model results for predict the next sequence one. As for now we don't know what feature is important for the model, we make it predict them. So if an indicator is add, the indicator value for the predict candle, is not calculate but predicted which is really not terrible. In that purpose, we verify the predictions on only one candle. Once we have defined the interesting features, we could add the calculs to the model directly.  
  
# Raw Datas:  
  
## Choice  
| Market                | Description                                                                               | 👍 Advantages                                                                                       | 👎 Disadvantages                                                                   | Prediction Suitability |
| :-------------------- | :---------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- | :--------------------: |
| **Equities (Stocks)** | Shares of individual companies, driven by earnings, M&A, corporate news                   | – Long daily history<br>– Sector trends<br>– High data volume                                      | – Abrupt jumps on earnings/news<br>– Idiosyncratic shocks                         |           ✓            |
| **Forex (FX)**        | Currency pairs traded 24/5, influenced by central banks, macro data, algos                | – Very high liquidity<br>– Continuous trading → fewer gaps<br>– Repeatable carry/momentum patterns | – High HFT noise<br>– Sudden regime shifts on policy announcements                |           ✓✓           |
| **Commodities**       | Raw materials (oil, gold, grains), subject to supply/demand, weather, geopolitics         | – Seasonal cycles (agri, energy)<br>– Macro correlations (USD, inflation)                          | – Geopolitical shocks<br>– Non‑linear spikes<br>– Limited history for some metals |           ✗            |
| **Cryptocurrencies**  | Decentralized digital assets (BTC, ETH), driven by sentiment, network effects, regulation | – 24/7 data → dense history<br>– Clear bull/bear cycles<br>– Sentiment‑based algorithmic patterns  | – Extreme volatility<br>– Forks/regulatory bans                                   |           ✗            |
| **Index Futures**     | Futures on baskets (S&P 500, DAX), reflecting broad market trends and fund flows          | – Less idiosyncratic noise than single stocks<br>– Clear long‑term trends<br>– High liquidity      | – Sensitive to macro releases (inflation, rates)                                  |          ✓✓✓           |
  
  
  
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
| spread  | Spread between bid (buy) and ask (sell) prices              | Typically: ask - bid, reflects market liquidity and hidden transaction cost |
  
**Notes:**  
- open, high, low, close = price structure used to build candlesticks.  
- tickvol != vol: one counts activity (number of ticks), the other measures actual volume traded.  
- spread can be:  
    - Fixed (some brokers),  
    - Variable (common in Forex and crypto),  
    - Widens during illiquid or volatile conditions.  
  
## Tests  

### Base features
First step is to know the model performance with only the price as information and does the aditional feature already present in our datas bring an usefull information ? 

| -         | -               | TickVolume      | Volume           | Spread           |
| --------- | --------------- | --------------- | ---------------- | ---------------- |
| **Price** | 0.01/2.396/0.54 | 0.016/3.125/0.5 | 0.014/2.526/0.52 | 0.014/2.481/0.52 |
*Table giving the mae/mse/directional_accuracy for each feature in the model*

**Results explanation**
- Volume: mainly 0 because generally not available in interbank forex.  
- Tick: more ticks usually mean more trading interest (higher liquidity, volatility). But different brokers/data vendors sample differently—tick counts can be noisy or inconsistent across pairs or time zones. So better to not keep it.  
- Spread: wider spreads typically signal lower liquidity or higher uncertainty—often accompanying big moves. Information in the price. Mainly usefull if the model have to enter order on the markets which is not the case there.
  
No difference is observe between the differents tests.  
Some settings/weights are probably need to be adjust in the custom loss because only buying candles are predicted.  
The candles size are coherents but not their positions:

<div style="display: flex; justify-content: center;">
  <div style="text-align: center; margin: 10px;">
    <img src="./img/candles_15_a.png" /><br/>
    <sub>Consistent prediction</sub>
  </div>
  <div style="text-align: center; margin: 10px;">
    <img src="./img/candles_0_a.png"/><br/>
    <sub>Unconsistent prediction</sub>
  </div>
</div>

### Indicators
| -             | rsi | macd | bollinger |
| ------------- | --- | ---- | --------- |
| **rsi**       |     |      |           |
| **macd**      |     |      |           |
| **bollinger** |     |      |           |

**rsi**
The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements on a scale from 0 to 100. It’s typically calculated over 14 periods as:  
$RSI=100-\frac{100}{1+\frac{Average Gain}{Average Loss}}$  
Readings above 70 suggest the asset may be overbought (potentially due for a pullback), while readings below 30 suggest it may be oversold (potentially due for a bounce). Traders also watch for divergences between RSI and price—for example, when price makes a new high but RSI does not, indicating weakening momentum.  

**macd**
The Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship between two exponential moving averages (EMAs) of price. The standard MACD line is the difference between the 12-period EMA and the 26-period EMA, and the signal line is a 9-period EMA of the MACD line. Key signals include:
- Crossover: When the MACD line crosses above the signal line, it’s bullish; when it crosses below, it’s bearish.
- Histogram: The bars represent the distance between the MACD and its signal line, helping visualize the strength of the trend.
- Zero line: Movement above zero suggests upward momentum; below zero suggests downward momentum.

**bollinger**
Bollinger Bands consist of a middle n-period simple moving average (SMA) (commonly 20 periods) and two bands plotted above and below it at k standard deviations (commonly ±2σ) of price. They adapt to volatility:
- Band width expands when volatility rises and contracts when volatility falls (“the squeeze”).
- Touches of the upper band can signal overbought conditions; touches of the lower band can indicate oversold conditions.
- Traders often use band breakouts (price moving outside the bands) as the start of strong directional moves, confirming with volume or other indicators.

# Todo
- add market + time infos  
- add indicators  

  
# Related papers  
- [Transformer-based forecasting for intraday trading in the Shanghai crude oil market: Analyzing open-high-low-close prices](https://www.sciencedirect.com/science/article/pii/S0140988323006047)  
- [Transformer-Based Deep Learning Model for Stock Price Prediction: A Case Study on Bangladesh Stock Market](https://arxiv.org/abs/2208.08300#:~:text=investors,based%20on%20their%20historical%20daily)  
- [Stock and market index prediction using Informer network](https://arxiv.org/pdf/2305.14382#:~:text=the%20convergence%20of%20networks%20result,stamp%20mechanism%20has%20significantly%20lower)  
- [Enhanced LFTSformer: A Novel Long-Term Financial Time Series Prediction Model Using Advanced Feature Engineering and the DS Encoder Informer Architecture](https://arxiv.org/html/2310.01884v2#:~:text=This%20study%20presents%20a%20groundbreaking,itself%20through%20several%20significant%20innovations)  
- [Transformer Based Time-Series Forecasting For Stock](https://arxiv.org/html/2502.09625v1#:~:text=In%20order%20to%20show%20the,with%20the%20traditional%20LSTM%20model)  
- [Exploring the Advantages of Transformers for High-Frequency Trading](https://arxiv.org/abs/2302.13850#:~:text=,are%20assessed%20and%20results%20indicate)  
- [TRANSFORMERS VERSUS LSTMS FOR ELECTRONIC TRADING](https://openreview.net/pdf/06f4232517e3c80aef7d6c683719114e1f037413.pdf#:~:text=A%20novel%20LSTM,as%20price%20differences%20and%20movements)  
- [LSTM versus Transformers: A Practical Comparison of Deep Learning Models for Trading Financial Instruments](https://www.scitepress.org/Papers/2024/129811/129811.pdf#:~:text=On%20experiment%202%2C%20the%20combination,of%20Deep%20Learning%20Models%20for)  
- [Learning Stock Price Signals from Candlestick Chart Via Vision Transformer](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5224805#:~:text=This%20study%20introduces%20and%20evaluates,power%20beyond%20traditional%20market%20characteristics)  # todo print once review
- [BreakGPT: Leveraging Large Language Models for Predicting Asset Price Surges](https://arxiv.org/html/2411.06076v1#:~:text=This%20paper%20introduces%20BreakGPT%2C%20a,local%20and%20global%20temporal%20dependencies)  
- [FinBHARAT - Stock Prediction Using Transformer](https://www.researchgate.net/publication/388955649_FinBHARAT_-_Stock_Prediction_Using_Transformer#:~:text=In%20traditional%20quantitative%20trading%2C%20effectively,market%20trends%20and%20price%20movements)  
- [Modality-aware Transformer for Financial Time series Forecasting](https://arxiv.org/html/2310.01232v2#:~:text=to%20predict%20the%20target%20time,modal)  
- [Stock Market Prediction Based on Cnn with Attention Mechanism](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5000005#:~:text=Forecasting%20stock%20prices%20is%20essential,and%20feature%20extraction%20compared%20to)  
- [Comparing Different Transformer Model Structures for Stock Prediction](https://www.arxiv.org/abs/2504.16361#:~:text=variant%20is%20most%20suitable%20for,performance%20in%20almost%20all%20cases)  
- [inancial Time Series Forecasting using CNN and Transformer](https://ar5iv.labs.arxiv.org/html/2304.04912#:~:text=Our%20contributions%20is%20the%20following%3A,2020)  