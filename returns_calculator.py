import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm

class ReturnsCalculator:
    def __init__(self, data_dir='data'):
        """
        初始化收益率计算器
        
        参数:
        -----------
        data_dir : str
            数据文件所在目录
        """
        self.data_dir = data_dir
        self.price_data, self.adj_factor, self.is_trade = self._load_price_data()
        
    def _load_price_data(self):
        """从pickle文件加载价格数据和复权因子，并转换为所需格式"""
        try:
            print("加载价格数据和复权因子...")
            # 读取pickle文件
            with open(os.path.join(self.data_dir, 'stock_data.pkl'), 'rb') as f:
                df = pickle.load(f)
            
            # 确保日期列为datetime类型
            df['END_DATE'] = pd.to_datetime(df['END_DATE'])
            
            # 将数据转换为宽格式（pivot）
            price_data = {}
            for field in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL', 'FACTOR', 'ISTRADE']:
                # 创建透视表，行为日期，列为股票代码
                pivot_df = df.pivot(index='END_DATE', columns='STOCK_CODE', values=field)
                # 将列名转换为小写以保持一致性
                price_data[field.lower()] = pivot_df
            
            # 重命名VOL为volume以保持一致性
            price_data['volume'] = price_data.pop('vol')
            
            # 提取复权因子和交易日标记
            adj_factor = price_data.pop('factor')
            is_trade = price_data.pop('istrade')
            
            print("价格数据和复权因子加载完成")
            return price_data, adj_factor, is_trade
            
        except Exception as e:
            raise Exception(f"加载数据失败: {str(e)}")
    
    def _get_adjusted_prices(self, price_type='close'):
        """
        获取复权后的价格
        
        参数:
        -----------
        price_type : str
            价格类型，可选 'open', 'high', 'low', 'close'
            
        返回:
        --------
        pd.DataFrame
            复权后的价格数据
        """
        # 获取原始价格
        prices = self.price_data[price_type]
        
        # 计算复权价格
        adjusted_prices = prices * self.adj_factor
        
        return adjusted_prices
    
    def calculate_returns(self, periods=[1, 5, 10, 20]):
        """
        计算不同周期的未来收益率（使用复权价格）
        当当天或未来那天 ISTRADE 为 0 时，收益率设为空值
        
        参数:
        -----------
        periods : list
            收益率周期列表，默认为[1, 5, 10, 20]天
            
        返回:
        --------
        dict
            包含不同周期未来收益率的字典
        """
        print("开始计算未来收益率...")
        # 获取复权后的收盘价
        close = self._get_adjusted_prices('close')
        returns_dict = {}
        
        for period in tqdm(periods, desc="计算不同周期未来收益率"):
            # 计算未来收益率
            forward_returns = close.shift(-period) / close - 1
            
            # 处理非交易日
            # 检查当天和未来那天的交易日标记
            # 当天不是交易日
            forward_returns[self.is_trade == 0] = np.nan
            # 未来那天不是交易日
            forward_returns[self.is_trade.shift(-period) == 0] = np.nan
            
            returns_dict[f'return_{period}d'] = forward_returns
        
        # 保存收益率数据
        self._save_returns(returns_dict)
        
        return returns_dict
    
    def _save_returns(self, returns_dict):
        """保存未来收益率数据到pickle文件"""
        try:
            print("保存未来收益率数据...")
            # 保存为pickle文件
            # returns_path = os.path.join(self.data_dir, 'returns.pkl')
            # with open(returns_path, 'wb') as f:
            #     pickle.dump(returns_dict, f)
            # print(f"未来收益率数据已保存至 {returns_path}")

            # 保存为pickle文件（每个周期一个文件）
            for period, returns in returns_dict.items():
                pkl_path = os.path.join(self.data_dir, f'returns_{period}.pkl')
                with open(pkl_path, 'wb') as f:
                    pickle.dump(returns, f)
                print(f"未来收益率数据已保存至 {pkl_path}")
                
        except Exception as e:
            print(f"保存未来收益率数据失败: {str(e)}")

