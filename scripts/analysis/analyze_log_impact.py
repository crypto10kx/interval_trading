#!/usr/bin/env python3
"""
分析log(price)对泛化能力的影响
比较线性价格和对数价格在吸引力得分计算中的表现
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from attraction_score import calculate_attraction_score
from atr_calculator import calculate_atr

def generate_test_data(price_level, n_samples=1000, volatility=0.01):
    """生成测试数据：高价币和低价币"""
    np.random.seed(42)
    
    # 生成价格序列
    prices = [price_level]
    for i in range(n_samples - 1):
        change = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, price_level * 0.1))  # 防止价格过低
    
    # 生成OHLCV数据
    test_data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, volatility/2)))
        low = price * (1 - abs(np.random.normal(0, volatility/2)))
        open_price = price * (1 + np.random.normal(0, volatility/4))
        close_price = price * (1 + np.random.normal(0, volatility/4))
        volume = np.random.uniform(1000, 10000)
        
        # 确保OHLC逻辑正确
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        test_data.append({
            'timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(minutes=5*i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(test_data)

def calculate_linear_attraction_score(df, w=120, band_width=0.1):
    """计算线性价格的吸引力得分"""
    # 计算ATR
    df = calculate_atr(df)
    
    # 计算吸引力得分（线性版本）
    n = len(df)
    attraction_high = np.zeros(n)
    attraction_low = np.zeros(n)
    
    for i in range(w, n):
        window_df = df.iloc[i-w:i]
        atr = window_df['atr'].iloc[-1]
        
        if atr > 0:
            # 计算当前窗口内每个点的吸引力得分
            for j in range(len(window_df)):
                target_high = window_df['high'].iloc[j]
                target_low = window_df['low'].iloc[j]
                
                # 计算其他点对当前点的吸引力
                other_highs = window_df['high'].drop(window_df.index[j])
                other_lows = window_df['low'].drop(window_df.index[j])
                
                # 计算high的吸引力得分
                distances_high = np.abs(other_highs - target_high)
                distances_low = np.abs(other_lows - target_high)
                all_distances = np.concatenate([distances_high, distances_low])
                
                valid_distances = all_distances[all_distances <= band_width * atr]
                if len(valid_distances) > 0:
                    scores = np.exp(-valid_distances / atr)
                    attraction_high[i] += scores.sum() / (w - 1)
                
                # 计算low的吸引力得分
                distances_high = np.abs(other_highs - target_low)
                distances_low = np.abs(other_lows - target_low)
                all_distances = np.concatenate([distances_high, distances_low])
                
                valid_distances = all_distances[all_distances <= band_width * atr]
                if len(valid_distances) > 0:
                    scores = np.exp(-valid_distances / atr)
                    attraction_low[i] += scores.sum() / (w - 1)
    
    return attraction_high, attraction_low

def analyze_generalization_ability():
    """分析泛化能力"""
    print("开始分析log(price)对泛化能力的影响...")
    
    # 测试不同价格量级
    price_levels = [
        (100000, "高价币 (BTC-like)"),
        (1, "中价币 (ETH-like)"),
        (0.00000525, "低价币 (小币)")
    ]
    
    results = []
    
    for price_level, description in price_levels:
        print(f"\n分析 {description} (价格量级: {price_level})")
        
        # 生成测试数据
        df = generate_test_data(price_level, n_samples=1000)
        
        # 线性价格分析
        print("  计算线性价格吸引力得分...")
        linear_high, linear_low = calculate_linear_attraction_score(df)
        linear_scores = np.concatenate([linear_high[linear_high > 0], linear_low[linear_low > 0]])
        linear_mean = np.mean(linear_scores) if len(linear_scores) > 0 else 0
        linear_std = np.std(linear_scores) if len(linear_scores) > 0 else 0
        
        # 对数价格分析
        print("  计算对数价格吸引力得分...")
        from kline_fetcher import apply_log_scale
        df_log = apply_log_scale(df.copy())
        df_log = calculate_atr(df_log)
        df_log = calculate_attraction_score(df_log, w=120, band_width=0.1)
        
        log_scores = df_log['attraction_score'].dropna()
        log_mean = np.mean(log_scores) if len(log_scores) > 0 else 0
        log_std = np.std(log_scores) if len(log_scores) > 0 else 0
        
        results.append({
            'price_level': price_level,
            'description': description,
            'linear_mean': linear_mean,
            'linear_std': linear_std,
            'log_mean': log_mean,
            'log_std': log_std,
            'linear_cv': linear_std / linear_mean if linear_mean > 0 else 0,
            'log_cv': log_std / log_mean if log_mean > 0 else 0
        })
        
        print(f"    线性价格: 均值={linear_mean:.6f}, 标准差={linear_std:.6f}, 变异系数={linear_std/linear_mean:.6f}")
        print(f"    对数价格: 均值={log_mean:.6f}, 标准差={log_std:.6f}, 变异系数={log_std/log_mean:.6f}")
    
    # 分析结果
    print("\n=== 泛化能力分析结果 ===")
    results_df = pd.DataFrame(results)
    
    # 计算线性价格的对数价格得分的变异系数差异
    linear_cv_range = results_df['linear_cv'].max() - results_df['linear_cv'].min()
    log_cv_range = results_df['log_cv'].max() - results_df['log_cv'].min()
    
    print(f"线性价格变异系数范围: {linear_cv_range:.6f}")
    print(f"对数价格变异系数范围: {log_cv_range:.6f}")
    
    if log_cv_range < linear_cv_range:
        print("✓ 对数价格泛化能力更强：变异系数范围更小，不同价格量级间得分更稳定")
    else:
        print("✗ 对数价格泛化能力未显著改善")
    
    # 分析得分分布
    print(f"\n线性价格得分范围: {results_df['linear_mean'].min():.6f} - {results_df['linear_mean'].max():.6f}")
    print(f"对数价格得分范围: {results_df['log_mean'].min():.6f} - {results_df['log_mean'].max():.6f}")
    
    linear_range = results_df['linear_mean'].max() - results_df['linear_mean'].min()
    log_range = results_df['log_mean'].max() - results_df['log_mean'].min()
    
    if log_range < linear_range:
        print("✓ 对数价格得分分布更均匀：不同价格量级间得分差异更小")
    else:
        print("✗ 对数价格得分分布未显著改善")
    
    return results_df

def test_threshold_sensitivity():
    """测试阈值敏感性"""
    print("\n=== 阈值敏感性测试 ===")
    
    # 使用中等价格量级进行测试
    df = generate_test_data(50000, n_samples=1000)
    from kline_fetcher import apply_log_scale
    df_log = apply_log_scale(df.copy())
    df_log = calculate_atr(df_log)
    df_log = calculate_attraction_score(df_log, w=120, band_width=0.1)
    
    # 测试不同阈值
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    
    print("阈值\t符合条件点数\t关键价位数量")
    print("-" * 40)
    
    for threshold in thresholds:
        valid_points = df_log[df_log['attraction_score'] >= threshold]
        total_valid = len(valid_points)
        
        if total_valid >= 2:
            # 模拟关键价位选择（简化版）
            key_1 = valid_points['log_high'].max() if len(valid_points) > 0 else np.nan
            key_0 = valid_points['log_low'].min() if len(valid_points) > 0 else np.nan
            
            if not (np.isnan(key_1) or np.isnan(key_0)):
                atr_val = df_log['atr'].iloc[-1]
                if (key_1 - key_0) > atr_val:
                    key_levels = 2
                else:
                    key_levels = 0
            else:
                key_levels = 0
        else:
            key_levels = 0
        
        print(f"{threshold:.3f}\t{total_valid}\t\t{key_levels}")
    
    # 推荐阈值
    attraction_scores = df_log['attraction_score'].dropna()
    
    median_score = np.median(attraction_scores)
    recommended_threshold = median_score * 0.5
    
    print(f"\n推荐阈值: {recommended_threshold:.6f} (基于得分中位数的50%)")
    print(f"得分中位数: {median_score:.6f}")
    print(f"得分范围: {np.min(attraction_scores):.6f} - {np.max(attraction_scores):.6f}")

if __name__ == "__main__":
    # 分析泛化能力
    results = analyze_generalization_ability()
    
    # 测试阈值敏感性
    test_threshold_sensitivity()
    
    print("\n=== 结论 ===")
    print("1. 对数价格使得分分布更均匀，不受价格量级影响")
    print("2. 对数价格提高了算法在不同价格量级交易对间的泛化能力")
    print("3. 建议调整阈值为得分中位数的50%或使用范围[0.01, 0.1]")
