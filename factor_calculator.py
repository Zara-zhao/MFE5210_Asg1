import numpy as np
import pandas as pd
from scipy import stats
import os
import pickle
from tqdm import tqdm

class FactorCalculator:
    def __init__(self, data_dir='data', lookback=20):
        """
        初始化因子计算器
        
        参数:
        -----------
        data_dir : str
            数据文件所在目录
        lookback : int
            因子计算的回看期（默认20天）
        """
        self.data_dir = data_dir
        self.lookback = lookback
        self.price_data = self._load_price_data()
        
    def _load_price_data(self):
        """从pickle文件加载价格数据并转换为所需格式"""
        try:
            # 读取pickle文件
            with open(os.path.join(self.data_dir, 'stock_data.pkl'), 'rb') as f:
                df = pickle.load(f)
            
            # 确保日期列为datetime类型
            df['END_DATE'] = pd.to_datetime(df['END_DATE'])
            
            # 将数据转换为宽格式（pivot）
            price_data = {}
            for field in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL']:
                # 创建透视表，行为日期，列为股票代码
                pivot_df = df.pivot(index='END_DATE', columns='STOCK_CODE', values=field)
                # 将列名转换为小写以保持一致性
                price_data[field.lower()] = pivot_df
            
            # 重命名VOL为volume以保持一致性
            price_data['volume'] = price_data.pop('vol')
            
            return price_data
            
        except Exception as e:
            raise Exception(f"加载数据失败: {str(e)}")
        
    def calculate_factor1(self):
        """计算因子1：交易量加权价格动量（带方向性）"""
        close = self.price_data['close']
        volume = self.price_data['volume']
        
        returns = close / close.shift(1) - 1
        up_move = (returns > 0).astype(float)
        down_move = (returns < 0).astype(float)
        
        up_component = up_move * returns.pow(2) * volume
        down_component = down_move * returns.pow(2) * volume
        
        up_ema = up_component.ewm(span=self.lookback).mean()
        down_ema = down_component.ewm(span=self.lookback).mean()
        volume_ema = volume.ewm(span=self.lookback).mean()
        
        return (up_ema / volume_ema) - (down_ema / volume_ema)
    
    def calculate_factor2(self):
        """计算因子2：收益率-交易量相关性"""
        close = self.price_data['close']
        volume = self.price_data['volume']
        
        returns = close / close.shift(1) - 1
        volume_change = volume / volume.shift(1) - 1
        
        component = returns * volume_change
        return component.ewm(span=self.lookback).mean() / returns.rolling(self.lookback).std()
        
    def calculate_factor3(self):
        """计算因子4：交易量符号收益率"""
        close = self.price_data['close']
        volume = self.price_data['volume']
        
        returns = close / close.shift(1) - 1
        volume_change = volume / volume.shift(1) - 1
        
        component = returns * np.sign(volume_change)
        return component.ewm(span=self.lookback).mean() / returns.rolling(self.lookback).std()
        
    def calculate_all_factors(self):
        """计算所有因子并返回DataFrame"""
        print("开始计算因子...")
        factors = pd.DataFrame()
        
        # 使用tqdm创建进度条
        factor_methods = [
            ('Factor1', self.calculate_factor1),
            ('Factor2', self.calculate_factor2),
            ('Factor3', self.calculate_factor3)
        ]
        
        for factor_name, method in tqdm(factor_methods, desc="计算因子进度"):
            try:
                factors[factor_name] = method().stack()
                print(f"\n{factor_name} 计算完成")
            except Exception as e:
                print(f"\n{factor_name} 计算失败: {str(e)}")
                factors[factor_name] = np.nan
        
        print("\n所有因子计算完成")
        return factors 