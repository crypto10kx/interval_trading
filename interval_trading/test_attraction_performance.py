#!/usr/bin/env python3
"""
测试吸引力得分算法性能
"""

import pandas as pd
import numpy as np
import time
import psutil
import sys
sys.path.append('src')

def monitor_memory():
    """监控内存使用情况"""
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    total_gb = memory.total / (1024**3)
    percent = memory.percent
    print(f'内存使用: {used_gb:.2f}GB / {total_gb:.2f}GB ({percent:.1f}%)')
    return used_gb < 7.0

def create_test_data(n_points=1000):
    """创建测试数据"""
    print(f"创建测试数据: {n_points} 个点")
    
    # 生成随机价格数据
    np.random.seed(42)
    base_price = 100.0
    price_range = 0.1  # 10%的价格波动
    
    high_prices = base_price * (1 + np.random.uniform(-price_range, price_range, n_points))
    low_prices = high_prices * (1 - np.random.uniform(0, 0.05, n_points))  # 低点比高点低0-5%
    atr_values = np.random.uniform(0.001, 0.01, n_points)  # ATR在0.1%-1%之间
    
    # 转换为对数价格
    log_high = np.log(high_prices)
    log_low = np.log(low_prices)
    
    df = pd.DataFrame({
        'log_high': log_high,
        'log_low': log_low,
        'atr': atr_values
    })
    
    return df

def optimized_attraction_score(df, w=120, band_width=0.1):
    """
    优化的吸引力得分计算算法
    
    使用滑动窗口和向量化操作，避免创建大型距离矩阵
    """
    n = len(df)
    scores = np.zeros(n)
    
    log_high = df['log_high'].values
    log_low = df['log_low'].values
    atr = df['atr'].values
    
    # 处理ATR为0或NaN的情况
    atr_safe = np.where(np.isnan(atr) | (atr == 0), 1.0, atr)
    
    print(f"开始计算吸引力得分，数据点: {n}, 窗口: {w}")
    
    for i in range(n):
        # 计算窗口范围
        start_idx = max(0, i - w + 1)
        end_idx = i + 1
        window_size = end_idx - start_idx
        
        if window_size < 2:
            scores[i] = 0.0
            continue
        
        # 获取窗口数据
        window_high = log_high[start_idx:end_idx]
        window_low = log_low[start_idx:end_idx]
        window_atr = atr_safe[start_idx:end_idx]
        
        # 当前目标价格和ATR
        target_price = log_high[i] if i < len(log_high) else log_low[i]
        target_atr = atr_safe[i]
        
        # 计算价格区间
        bandwidth = band_width * target_atr
        lower_bound = target_price - bandwidth
        upper_bound = target_price + bandwidth
        
        # 创建掩码：排除当前K线
        mask = np.ones(window_size, dtype=bool)
        current_relative_idx = i - start_idx
        if 0 <= current_relative_idx < window_size:
            mask[current_relative_idx] = False
        
        # 应用掩码
        window_high_masked = window_high[mask]
        window_low_masked = window_low[mask]
        window_atr_masked = window_atr[mask]
        
        if len(window_high_masked) == 0:
            scores[i] = 0.0
            continue
        
        # 计算high点的得分
        high_in_range = (window_high_masked >= lower_bound) & (window_high_masked <= upper_bound)
        high_distances = np.abs(window_high_masked - target_price)
        high_scores = np.where(high_in_range, 
                              np.exp(-high_distances / window_atr_masked), 
                              0.0)
        
        # 计算low点的得分
        low_in_range = (window_low_masked >= lower_bound) & (window_low_masked <= upper_bound)
        low_distances = np.abs(window_low_masked - target_price)
        low_scores = np.where(low_in_range, 
                             np.exp(-low_distances / window_atr_masked), 
                             0.0)
        
        # 每根K线只计最高得分
        kline_scores = np.maximum(high_scores, low_scores)
        
        # 计算总得分
        total_score = np.sum(kline_scores)
        scores[i] = total_score / w
    
    return scores

def test_performance():
    """测试不同数据量下的性能"""
    test_sizes = [100, 500, 1000, 2000, 5000]
    
    print("=== 吸引力得分算法性能测试 ===")
    print(f"初始内存状态:")
    monitor_memory()
    
    for size in test_sizes:
        print(f"\n--- 测试数据量: {size} ---")
        
        # 创建测试数据
        df = create_test_data(size)
        
        # 测试性能
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**3)
        
        scores = optimized_attraction_score(df, w=120, band_width=0.1)
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / (1024**3)
        
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        
        print(f"计算时间: {duration:.2f} 秒")
        print(f"内存使用: {memory_used:.2f} GB")
        print(f"平均得分: {np.mean(scores):.6f}")
        print(f"得分范围: {np.min(scores):.6f} - {np.max(scores):.6f}")
        
        # 检查内存使用
        if not monitor_memory():
            print("⚠️ 内存使用过高，停止测试")
            break
        
        # 性能评估
        if duration > 10:  # 超过10秒认为太慢
            print("❌ 性能不合格：计算时间过长")
            break
        elif memory_used > 1.0:  # 超过1GB内存使用
            print("❌ 性能不合格：内存使用过多")
            break
        else:
            print("✅ 性能合格")

if __name__ == "__main__":
    test_performance()
