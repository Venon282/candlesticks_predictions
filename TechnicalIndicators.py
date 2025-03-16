import pandas as pd
import numpy as np

class TechnicalIndicators:
    def __init__(self, indicators=None, params=None):
        """
        Initializes the technical indicator system.

        Args:
          indicators: A list of strings representing the indicators to compute.
                      For example: ['rsi', 'macd', 'bollinger']
          params: A dictionary with parameters for each indicator. Example:
                      {
                        'rsi': {'window': 14},
                        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                        'bollinger': {'window': 20, 'std': 2}
                      }
        """
        self.indicators = indicators if indicators is not None else []
        self.params = params if params is not None else {}
    
    def compute_RSI(self, series, window=14):
        """
        Compute the Relative Strength Index (RSI) for a price series.
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_MACD(self, series, fast=12, slow=26, signal=9):
        """
        Compute the MACD (Moving Average Convergence Divergence) for a price series.
        Returns: macd_line, signal_line, macd_histogram.
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - signal_line
        return macd_line, signal_line, macd_hist

    def compute_bollinger(self, series, window=20, std=2):
        """
        Compute Bollinger Bands for a price series.
        Returns: sma (moving average), upper_band, lower_band.
        """
        sma = series.rolling(window=window, min_periods=window).mean()
        rstd = series.rolling(window=window, min_periods=window).std()
        upper_band = sma + std * rstd
        lower_band = sma - std * rstd
        return sma, upper_band, lower_band

    def add_indicators(self, df, inplace=False):
        """
        Computes and appends the selected technical indicators to the DataFrame.
        The DataFrame is assumed to have at least the following columns:
          - 'open', 'high', 'low', 'close', 'volume'
        Returns:
          A new DataFrame with additional columns for each computed indicator.
        """
        df_result = df if inplace else df.copy()
        
        # Add RSI computed on the close price.
        if 'rsi' in self.indicators:
            window = self.params.get('rsi', {}).get('window', 14)
            df_result['rsi'] = self.compute_RSI(df_result['close'], window)
        
        # Add MACD computed on the close price.
        if 'macd' in self.indicators:
            fast = self.params.get('macd', {}).get('fast', 12)
            slow = self.params.get('macd', {}).get('slow', 26)
            signal = self.params.get('macd', {}).get('signal', 9)
            macd_line, signal_line, macd_hist = self.compute_MACD(df_result['close'], fast, slow, signal)
            df_result['macd_line'] = macd_line
            df_result['macd_signal'] = signal_line
            df_result['macd_hist'] = macd_hist
        
        # Add Bollinger Bands computed on the close price.
        if 'bollinger' in self.indicators:
            window = self.params.get('bollinger', {}).get('window', 20)
            std = self.params.get('bollinger', {}).get('std', 2)
            sma, upper_band, lower_band = self.compute_bollinger(df_result['close'], window, std)
            df_result['bollinger_sma'] = sma
            df_result['bollinger_upper'] = upper_band
            df_result['bollinger_lower'] = lower_band
        
        return df_result
    
# === Example Usage ===
# if __name__ == '__main__':
#     # Sample DataFrame with candlestick data
#     data = {
#         'open': np.random.uniform(100, 200, 100),
#         'high': np.random.uniform(200, 300, 100),
#         'low': np.random.uniform(50, 100, 100),
#         'close': np.random.uniform(100, 200, 100),
#         'volume': np.random.uniform(1000, 5000, 100)
#     }
#     df = pd.DataFrame(data)
    
#     # Define which indicators to compute and any custom parameters
#     indicators_to_add = ['rsi', 'macd', 'bollinger']
#     indicator_params = {
#         'rsi': {'window': 14},
#         'macd': {'fast': 12, 'slow': 26, 'signal': 9},
#         'bollinger': {'window': 20, 'std': 2}
#     }
    
#     tech_ind = TechnicalIndicators(indicators=indicators_to_add, params=indicator_params)
#     df_with_indicators = tech_ind.add_indicators(df)
    
#     print(df_with_indicators.head())