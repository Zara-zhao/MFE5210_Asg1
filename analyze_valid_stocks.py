import pickle
import os
import pandas as pd
import numpy as np
from datetime import datetime
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def analyze_valid_stocks(factors, returns):
    """
    分析每个因子在每个日期的有效股票数量
    
    参数:
    -----------
    factors : dict
        因子数据字典
    returns : pd.DataFrame
        收益率数据
    """
    # 获取所有日期
    all_dates = sorted(returns.index.unique())
    
    # 创建结果DataFrame
    results = pd.DataFrame(index=all_dates, columns=factors.keys())
    
    # 创建每日统计DataFrame
    daily_stats = pd.DataFrame(index=all_dates, 
                             columns=['总股票数', '有效股票数', '有效率(%)'])
    
    # 对每个因子进行分析
    for factor_name, factor_data in factors.items():
        print(f"\n分析 {factor_name} 的有效股票数量...")
        
        # 获取因子数据的所有日期
        factor_dates = factor_data.index.get_level_values(0).unique()
        
        # 对每个日期进行分析
        for date in all_dates:
            if date in factor_dates:
                try:
                    # 获取当前日期的因子值
                    curr_factors = factor_data.xs(date, level=0)
                    # 获取当前日期的收益率
                    curr_returns = returns.loc[date]
                    
                    # 找到共同的股票代码
                    common_stocks = curr_factors.index.intersection(curr_returns.index)
                    
                    if len(common_stocks) > 0:
                        # 获取有效数据
                        factor_values = curr_factors[common_stocks]
                        returns_values = curr_returns[common_stocks]
                        
                        # 计算有效股票数量
                        valid_mask = factor_values.notna() & returns_values.notna()
                        valid_count = valid_mask.sum()
                        
                        # 存储结果
                        results.loc[date, factor_name] = valid_count
                        
                        # 更新每日统计
                        daily_stats.loc[date, '总股票数'] = len(common_stocks)
                        daily_stats.loc[date, '有效股票数'] = valid_count
                        daily_stats.loc[date, '有效率(%)'] = valid_count/len(common_stocks)*100
                        
                except Exception as e:
                    print(f"处理日期 {date} 时出错: {str(e)}")
                    continue
    
    # 计算统计信息
    stats = pd.DataFrame(index=factors.keys(), 
                        columns=['平均有效股票数', '最小有效股票数', '最大有效股票数', 
                                '标准差', '平均有效率(%)', '最小有效率(%)', '最大有效率(%)'])
    
    for factor_name in factors.keys():
        valid_counts = results[factor_name].dropna()
        total_counts = daily_stats['总股票数'].dropna()
        efficiency = (valid_counts / total_counts * 100).dropna()
        
        stats.loc[factor_name, '平均有效股票数'] = valid_counts.mean()
        stats.loc[factor_name, '最小有效股票数'] = valid_counts.min()
        stats.loc[factor_name, '最大有效股票数'] = valid_counts.max()
        stats.loc[factor_name, '标准差'] = valid_counts.std()
        stats.loc[factor_name, '平均有效率(%)'] = efficiency.mean()
        stats.loc[factor_name, '最小有效率(%)'] = efficiency.min()
        stats.loc[factor_name, '最大有效率(%)'] = efficiency.max()
    
    return results, stats, daily_stats

def main():
    print("加载因子数据...")
    with open(os.path.join('data', 'factors.pkl'), 'rb') as f:
        factors = pickle.load(f)

    print("加载收益率数据...")
    with open(os.path.join('data', 'returns_return_1d.pkl'), 'rb') as f:
        returns = pickle.load(f)

    # 运行分析
    print("\n开始分析有效股票数量...")
    results, stats, daily_stats = analyze_valid_stocks(factors, returns)

    # 显示统计信息
    print("\n因子有效股票数量统计:")
    print("\n" + "="*100)
    print(stats.to_string())
    print("="*100)

    # 显示每日统计信息
    print("\n每日统计信息 (前5个交易日):")
    print("\n" + "="*100)
    print(daily_stats.head().to_string())
    print("="*100)

    print("\n每日统计信息 (最后5个交易日):")
    print("\n" + "="*100)
    print(daily_stats.tail().to_string())
    print("="*100)

    # 保存结果到CSV文件
    results.to_csv('valid_stocks_analysis.csv')
    stats.to_csv('valid_stocks_stats.csv')
    daily_stats.to_csv('daily_stats.csv')
    print("\n分析结果已保存到:")
    print("- valid_stocks_analysis.csv (每个因子的每日有效股票数)")
    print("- valid_stocks_stats.csv (因子统计信息)")
    print("- daily_stats.csv (每日统计信息)")

if __name__ == "__main__":
    main() 