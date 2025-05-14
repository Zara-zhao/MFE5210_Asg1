import pandas as pd
import numpy as np
import os
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class FactorMetrics:
    def __init__(self, data_dir='data'):
        """初始化因子评估器"""
        self.data_dir = data_dir
    
    def load_data(self, start_date='2015-01-01', end_date='2024-12-31'):
        """
        加载并合并因子数据和收益率数据
        
        参数:
        --------
        start_date : str
            起始日期
        end_date : str
            结束日期
        
        返回:
        --------
        tuple: (因子数据DataFrame, 收益率数据DataFrame)
        """
        # 加载因子数据
        print("加载因子数据...")
        factors_path = os.path.join(self.data_dir, 'factors.pkl')
        with open(factors_path, 'rb') as f:
            factors = pickle.load(f)
        # print(f"加载的因子数据包含 {factors.shape[1]} 个因子")
        
        # 加载收益率数据
        print("加载收益率数据...")
        returns_path = os.path.join(self.data_dir, 'returns_return_1d.pkl')
        with open(returns_path, 'rb') as f:
            returns = pickle.load(f)
        # print(f"收益率数据形状: {returns.shape}")
        
        # 转换日期索引
        factors.index = pd.MultiIndex.from_tuples(factors.index, names=['END_DATE', 'STOCK_CODE'])
        factors.index = factors.index.set_levels(pd.to_datetime(factors.index.levels[0]), level=0)
        # print(f"因子数据日期范围: {factors.index.get_level_values(0).min()} 到 {factors.index.get_level_values(0).max()}")
        
        # 转换收益率数据索引
        returns.index = pd.to_datetime(returns.index)
        # print(f"收益率数据日期范围: {returns.index.min()} 到 {returns.index.max()}")
        
        # 找到共同日期
        common_dates = factors.index.get_level_values(0).unique().intersection(returns.index)
        # print(f"共同日期数量: {len(common_dates)}")
        
        # 截取指定时间范围内的数据
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        common_dates = common_dates[(common_dates >= start_date) & (common_dates <= end_date)]
        # print(f"指定时间范围内的日期数量: {len(common_dates)}")
        
        # 使用共同日期重新索引数据
        factors = factors.loc[common_dates]
        returns = returns.loc[common_dates]
        
        # 删除缺失值
        factors = factors.dropna(how='all')
        returns = returns.dropna(how='all')
        
        print(f"\n数据预处理完成。")
        # print(f"时间范围: {factors.index.get_level_values(0).min()} 到 {factors.index.get_level_values(0).max()}")
        # print(f"有效交易日数: {len(factors.index.get_level_values(0).unique())}")
        # print(f"有效股票数: {len(factors.index.get_level_values(1).unique())}")
        # print(f"因子数据形状: {factors.shape}")
        # print(f"收益率数据形状: {returns.shape}")
        
        return factors, returns
    
    def calculate_ic(self, factor, returns, factor_direction=-1):
        """
        计算信息系数（IC）
        
        参数:
        --------
        factor : pd.DataFrame
            因子数据，MultiIndex为[END_DATE, STOCK_CODE]
        returns : pd.DataFrame
            收益率数据，索引为日期，列为股票代码
        factor_direction : int
            因子方向，1表示因子值越大越好，-1表示因子值越小越好
            
        返回:
        --------
        tuple: (IC序列, IC统计信息字典)
        """
        ic_series = []
        
        # 确保factor是MultiIndex DataFrame
        if not isinstance(factor.index, pd.MultiIndex):
            raise ValueError("Factor data must have a MultiIndex with 'END_DATE' and 'STOCK_CODE' levels")
            
        # 确保returns的索引是日期
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)
            
        # 获取所有日期
        dates = factor.index.get_level_values('END_DATE').unique()
        
        # 根据因子方向调整因子值
        if factor_direction == -1:
            factor = -factor
        
        for date in dates:
            try:
                if date in returns.index:
                    # 获取当前日期的数据
                    curr_factors = factor.xs(date, level='END_DATE')
                    curr_returns = returns.loc[date]
                    
                    # 确保curr_factors的索引是股票代码
                    if not isinstance(curr_factors.index, pd.Index):
                        curr_factors = pd.Series(curr_factors, index=curr_factors.index)
                    
                    # 先删除收益率为nan的股票
                    valid_returns = curr_returns.notna()
                    if valid_returns.sum() > 0:
                        curr_returns = curr_returns[valid_returns]
                        
                        # 找到共同股票
                        common_stocks = curr_factors.index.intersection(curr_returns.index)
                        if len(common_stocks) > 0:
                            factor_values = curr_factors[common_stocks]
                            return_values = curr_returns[common_stocks]
                            
                            # 删除因子值为nan的股票
                            valid_factors = factor_values.notna()
                            if valid_factors.sum() > 0:
                                factor_values = factor_values[valid_factors]
                                return_values = return_values[valid_factors]
                                
                                # 确保有足够的有效数据点
                                if len(factor_values) > 10:  # 设置最小样本量要求
                                    ic = stats.spearmanr(factor_values, return_values)[0]
                                    if not np.isnan(ic):  # 确保IC不是NaN
                                        ic_series.append((date, ic))
            except Exception as e:
                print(f"计算日期 {date} 的IC时出错: {str(e)}")
                continue
        
        if not ic_series:
            return pd.Series(dtype=float), {}
            
        ic = pd.Series([x[1] for x in ic_series], index=[x[0] for x in ic_series])
        
        # 计算IC统计信息
        ic_stats = {
            'IC_mean': ic.mean(),
            'IC_std': ic.std(),
            'IC_IR': ic.mean() / ic.std() * np.sqrt(252) if ic.std() != 0 else np.nan,
            'IC_positive_ratio': (ic > 0).mean(),
            # 'IC_negative_ratio': (ic < 0).mean(),
            # 'IC_abs_mean': ic.abs().mean(),
            # 'IC_skew': ic.skew(),
            # 'IC_kurt': ic.kurtosis(),
            # 'IC_t_stat': (ic.mean() / (ic.std() / np.sqrt(len(ic)))) if len(ic) > 0 else np.nan,
            # 'IC_win_rate': (ic > 0).mean() if factor_direction == 1 else (ic < 0).mean()
        }
        
        return ic, ic_stats
    
    def calculate_all_factors_ic(self, factors, returns, factor_directions=None):
        """
        计算所有因子的IC
        
        参数:
        --------
        factors : pd.DataFrame
            因子数据，MultiIndex为[END_DATE, STOCK_CODE]，列为因子名称
        returns : pd.DataFrame
            收益率数据，索引为日期，列为股票代码
        factor_directions : dict, optional
            因子方向字典，key为因子名称，value为1或-1
            1表示因子值越大越好，-1表示因子值越小越好
            如果不提供，默认所有因子方向为-1
            
        返回:
        --------
        tuple: (IC序列DataFrame, IC统计信息DataFrame)
        """
        print("计算所有因子的IC...")
        
        # 初始化结果存储
        all_ic_series = {}
        all_ic_stats = {}
        
        # 获取所有因子名称
        factor_names = factors.columns
        
        # 设置默认因子方向
        if factor_directions is None:
            factor_directions = {name: -1 for name in factor_names}
        
        # 计算每个因子的IC
        for factor_name in factor_names:
            print(f"\n计算因子 {factor_name} 的IC...")
            
            # 获取因子数据
            factor_data = factors[factor_name]
            
            # 获取因子方向
            direction = factor_directions.get(factor_name, -1)
            
            # 计算IC
            ic_series, ic_stats = self.calculate_ic(
                factor_data, 
                returns, 
                factor_direction=direction
            )
            
            # 存储结果
            all_ic_series[factor_name] = ic_series
            all_ic_stats[factor_name] = ic_stats
            
            # 打印IC统计信息
            print(f"{factor_name} IC统计信息:")
            print(f"IC均值: {ic_stats['IC_mean']:.4f}")
            print(f"IC标准差: {ic_stats['IC_std']:.4f}")
            print(f"IC_IR: {ic_stats['IC_IR']:.4f}")
            print(f"IC为正的比例: {ic_stats['IC_positive_ratio']:.2%}")
        
        # 转换为DataFrame
        ic_series_df = pd.DataFrame(all_ic_series)
        ic_stats_df = pd.DataFrame(all_ic_stats).T
        
        # 添加因子方向信息
        ic_stats_df['factor_direction'] = pd.Series(factor_directions)
        
        # 按IC_IR降序排序
        ic_stats_df = ic_stats_df.sort_values('IC_IR', ascending=False)
        
        print("\n所有因子IC计算完成")
 
        return ic_series_df, ic_stats_df
    
    def plot_ic_heatmap(self, ic_series_df, save_path=None):
        """
        绘制IC热力图
        
        参数:
        --------
        ic_series_df : pd.DataFrame
            IC序列数据，行为日期，列为因子名称
        save_path : str, optional
            图表保存路径
        """
        # 计算IC相关性矩阵
        ic_corr = ic_series_df.corr()
        
        # 绘制热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(ic_corr, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f',
                   square=True)
        plt.title('Factor IC Correlation Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"IC相关性热力图已保存至: {save_path}")
        
        plt.close()
    
    def plot_ic_ts(self, ic_series_df, factor_names=None, save_path=None):
        """
        绘制IC时间序列图
        
        参数:
        --------
        ic_series_df : pd.DataFrame
            IC序列数据，行为日期，列为因子名称
        factor_names : list, optional
            要绘制的因子名称列表，如果不提供则绘制所有因子
        save_path : str, optional
            图表保存路径
        """
        if factor_names is None:
            factor_names = ic_series_df.columns
        
        # 计算滚动IC（20日）
        rolling_ic = ic_series_df[factor_names].rolling(window=20).mean()
        
        # 绘制时间序列图
        plt.figure(figsize=(15, 8))
        for factor in factor_names:
            plt.plot(rolling_ic.index, rolling_ic[factor], 
                    label=factor, alpha=0.7)
        
        plt.title('Rolling IC (20-day)')
        plt.xlabel('日期')
        plt.ylabel('IC')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"IC时间序列图已保存至: {save_path}")
        
        plt.close()
    
    def plot_factor_corr(self, factors, returns, save_path=None):
        """
        计算因子相关性矩阵并绘制热力图
        
        参数:
        --------
        factors : pd.DataFrame
            因子数据，MultiIndex为[END_DATE, STOCK_CODE]，列为因子名称
        returns : pd.DataFrame
            收益率数据，索引为日期，列为股票代码
        save_path : str, optional
            图表保存路径
        """
        # 获取所有日期
        dates = factors.index.get_level_values('END_DATE').unique()
        valid_factors = []
        
        # 对每个日期处理数据
        for date in dates:
            if date in returns.index:
                # 获取当前日期的数据
                curr_factors = factors.xs(date, level='END_DATE')
                curr_returns = returns.loc[date]
                
                # 删除收益率为0或nan的股票
                valid_returns = (curr_returns != 0) & (curr_returns.notna())
                if valid_returns.sum() > 0:
                    # 找到共同股票
                    common_stocks = curr_factors.index.intersection(curr_returns[valid_returns].index)
                    if len(common_stocks) > 0:
                        # 获取有效股票的因子值
                        valid_factor_values = curr_factors.loc[common_stocks]
                        valid_factors.append(valid_factor_values)
        
        if not valid_factors:
            raise ValueError("没有有效的因子数据")
            
        # 合并所有日期的数据
        all_factors = pd.concat(valid_factors)
        
        # 计算相关性矩阵
        corr_matrix = all_factors.corr()
        
        # 创建新的图表
        plt.clf()  # 清除当前图表
        plt.figure(figsize=(15, 12))  # 创建新的图表窗口
        
        # 绘制热力图
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f',
                   square=True)
        plt.title('Factor Correlation Heatmap (Excluding Zero Returns)')
        
        # 调整布局
        plt.tight_layout()
        
        # 显示图表
        plt.show()
        
        # 如果需要保存
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"因子相关性热力图已保存至: {save_path}")
        
        plt.close()
        
        return corr_matrix
    
    def calculate_group_returns(self, factor, returns, n_groups=5, factor_direction=-1):
        """
        计算因子分组收益
        
        参数:
        --------
        factor : pd.Series
            因子数据，MultiIndex为[END_DATE, STOCK_CODE]
        returns : pd.DataFrame
            收益率数据，索引为日期，列为股票代码
        n_groups : int
            分组数量
        factor_direction : int
            因子方向，1表示因子值越大越好，-1表示因子值越小越好
            
        返回:
        --------
        tuple: (分组收益率DataFrame, 评估指标字典)
        """
        print(f"计算{n_groups}分组收益...")
        
        # 确保factor是MultiIndex Series
        if not isinstance(factor.index, pd.MultiIndex):
            raise ValueError("Factor data must have a MultiIndex with 'END_DATE' and 'STOCK_CODE' levels")
            
        # 确保returns的索引是日期
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)
            
        # 获取所有日期
        dates = factor.index.get_level_values('END_DATE').unique()
        group_returns = []
        
        # 根据因子方向调整因子值
        if factor_direction == -1:
            factor = -factor
        
        # 对每个日期进行分组
        for date in dates:
            if date in returns.index:
                # 获取当前日期的数据
                curr_factors = factor.xs(date, level='END_DATE')
                curr_returns = returns.loc[date]
                
                # 删除收益率为0或nan的股票
                valid_returns = (curr_returns != 0) & (curr_returns.notna())
                if valid_returns.sum() > 0:
                    curr_returns = curr_returns[valid_returns]
                    
                    # 找到共同股票
                    common_stocks = curr_factors.index.intersection(curr_returns.index)
                    if len(common_stocks) > 0:
                        factor_values = curr_factors[common_stocks]
                        return_values = curr_returns[common_stocks]
                        
                        # 删除因子值为nan的股票
                        valid_factors = factor_values.notna()
                        if valid_factors.sum() > 0:
                            factor_values = factor_values[valid_factors]
                            return_values = return_values[valid_factors]
                            
                            # 确保有足够的有效数据点
                            if len(factor_values) >= n_groups:
                                # 计算分位数
                                quantiles = pd.qcut(factor_values, n_groups, labels=False)
                                
                                # 计算每组的平均收益
                                group_ret = pd.Series(return_values.values, index=quantiles).groupby(level=0).mean()
                                group_ret.name = date
                                group_returns.append(group_ret)
        
        if not group_returns:
            raise ValueError("没有足够的数据计算分组收益")
            
        # 合并所有日期的分组收益
        group_returns_df = pd.DataFrame(group_returns)
        
        # 计算累积收益
        cum_returns = (1 + group_returns_df).cumprod()
        
        # 计算评估指标
        metrics = self._calculate_performance_metrics(group_returns_df)
        
        return group_returns_df, cum_returns, metrics
    
    def _calculate_performance_metrics(self, returns_df):
        """
        计算评估指标
        
        参数:
        --------
        returns_df : pd.DataFrame
            分组收益率数据，行为日期，列为分组编号
            
        返回:
        --------
        dict: 评估指标字典
        """
        # 计算年化收益率
        annual_returns = returns_df.mean() * 252
        
        # 计算年化波动率
        annual_vol = returns_df.std() * np.sqrt(252)
        
        # 计算夏普比率
        sharpe_ratio = annual_returns / annual_vol
        
        # 计算最大回撤
        cum_returns = (1 + returns_df).cumprod()
        max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
        
        # 计算胜率
        win_rate = (returns_df > 0).mean()
        
        # 计算收益风险比
        return_risk_ratio = annual_returns / annual_vol
        
        # 整理指标
        metrics = pd.DataFrame({
            'Annual Return': annual_returns,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Return/Risk Ratio': return_risk_ratio
        })
        
        return metrics
    
    def plot_group_returns(self, cum_returns, metrics, save_path=None):
        """
        绘制分组收益曲线
        
        参数:
        --------
        cum_returns : pd.DataFrame
            累积收益率数据，行为日期，列为分组编号
        metrics : pd.DataFrame
            评估指标数据（此参数保留但不再使用）
        save_path : str, optional
            图表保存路径
        """
        # 创建图表
        plt.figure(figsize=(15, 8))
        
        # 绘制累积收益曲线
        for group in cum_returns.columns:
            plt.plot(cum_returns.index, cum_returns[group], 
                    label=f'Group {group+1}', alpha=0.7)
        
        plt.title('Cumulative Returns by Factor Group')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 如果需要保存，先保存再显示
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"分组收益曲线已保存至: {save_path}")
        
        # 显示图表
        plt.show()
        
        plt.close()
    
    def analyze_factor_returns(self, factor, returns, n_groups=5, factor_direction=-1, 
                             save_path=None):
        """
        分析因子分组收益并生成报告
        
        参数:
        --------
        factor : pd.Series
            因子数据，MultiIndex为[END_DATE, STOCK_CODE]
        returns : pd.DataFrame
            收益率数据，索引为日期，列为股票代码
        n_groups : int
            分组数量
        factor_direction : int
            因子方向
        save_path : str, optional
            图表保存路径
            
        返回:
        --------
        tuple: (分组收益率DataFrame, 累积收益率DataFrame, 评估指标DataFrame)
        """
        # 计算分组收益
        group_returns, cum_returns, metrics = self.calculate_group_returns(
            factor, 
            returns,
            n_groups=n_groups,
            factor_direction=factor_direction
        )
        
        # 绘制分析图
        self.plot_group_returns(cum_returns, metrics, save_path)
        
        # # 打印评估指标
        # print("\n因子分组收益评估指标：")
        # print(metrics)
        
        return group_returns, cum_returns, metrics
    
    def plot_factor_group_returns(self, factor, returns, n_groups=5, factor_direction=-1, save_path=None):
        """
        绘制因子分组收益曲线
        
        参数:
        --------
        factor : pd.Series
            因子数据，MultiIndex为[END_DATE, STOCK_CODE]
        returns : pd.DataFrame
            收益率数据，索引为日期，列为股票代码
        n_groups : int
            分组数量
        factor_direction : int
            因子方向，1表示因子值越大越好，-1表示因子值越小越好
        save_path : str, optional
            图表保存路径
        """
        # 计算分组收益
        group_returns, cum_returns, _ = self.calculate_group_returns(
            factor, 
            returns,
            n_groups=n_groups,
            factor_direction=factor_direction
        )
        
        # 创建图表
        plt.figure(figsize=(15, 8))
        
        # 绘制累积收益曲线
        for group in cum_returns.columns:
            plt.plot(cum_returns.index, cum_returns[group], 
                    label=f'Group {group+1}', alpha=0.7)
        
        plt.title('Factor Group Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 显示图表
        plt.show()
        
        # 如果需要保存
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"分组收益曲线已保存至: {save_path}")
        
        plt.close()
        
        return cum_returns
    
