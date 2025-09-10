#!/usr/bin/env python3
"""
简化测试脚本 - 避免卡顿问题
"""

import psutil
import sys
import os
import time
sys.path.append('src')

def monitor_memory():
    """监控内存使用情况"""
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    total_gb = memory.total / (1024**3)
    percent = memory.percent
    print(f'内存使用: {used_gb:.2f}GB / {total_gb:.2f}GB ({percent:.1f}%)')
    return used_gb < 8.0

def main():
    print('=== 简化测试脚本 - 避免卡顿 ===')
    print(f'初始内存状态:')
    monitor_memory()

    # 使用预定义交易对，跳过验证
    print('\n=== 使用预定义交易对 ===')
    test_symbols = [
        {'symbol': 'BTC/USDT:USDT', 'volume': 1000000},
        {'symbol': 'ETH/USDT:USDT', 'volume': 800000},
        {'symbol': 'BNB/USDT:USDT', 'volume': 600000}
    ]

    print(f'使用预定义交易对: {len(test_symbols)} 个')
    for symbol in test_symbols:
        print(f'  - {symbol["symbol"]}')

    monitor_memory()

    # 测试单个交易对下载
    print('\n=== 测试单个交易对下载 ===')
    from kline_fetcher import fetch_kline_data

    symbol = test_symbols[0]['symbol']
    print(f'测试交易对: {symbol}')
    print('下载少量数据（1天）...')

    try:
        start_time = time.time()
        df = fetch_kline_data(symbol, '2024-12-01', '2024-12-01')  # 使用更近的时间
        end_time = time.time()
        
        print(f'下载完成: {len(df)} 条记录，耗时: {end_time - start_time:.2f}秒')
        print(f'时间范围: {df["timestamp"].min()} 到 {df["timestamp"].max()}')
        print(f'列名: {list(df.columns)}')
        
        monitor_memory()
        
        # 快速测试峰谷算法
        print('\n=== 快速测试峰谷算法 ===')
        from peak_valley import identify_high_low_points
        
        print('执行峰谷识别...')
        hl_df = identify_high_low_points(df, g=3)
        print(f'峰谷识别完成，数据点: {len(hl_df)}')
        print(f'高点数量: {hl_df["is_high"].sum()}')
        print(f'低点数量: {hl_df["is_low"].sum()}')
        
        monitor_memory()
        
        # 测试ATR计算
        print('\n=== 测试ATR计算 ===')
        from atr_calculator import calculate_atr
        
        print('计算ATR...')
        atr_df = calculate_atr(df, period=20)  # 使用较小的周期
        print(f'ATR计算完成')
        print(f'ATR范围: {atr_df["atr"].min():.6f} - {atr_df["atr"].max():.6f}')
        
        monitor_memory()
        
    except Exception as e:
        print(f'测试失败: {e}')
        import traceback
        traceback.print_exc()

    print('\n✅ 简化测试完成')
    print(f'最终内存状态:')
    monitor_memory()

if __name__ == "__main__":
    main()
