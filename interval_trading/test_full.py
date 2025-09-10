#!/usr/bin/env python3
"""
完整测试脚本 - 执行模块1-3，监控内存使用
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
    print('=== 完整测试脚本 - 执行模块1-3 ===')
    print(f'初始内存状态:')
    monitor_memory()

    # 模块1: 数据获取
    print('\n=== 模块1: 数据获取 ===')
    from kline_fetcher import fetch_kline_data

    # 使用预定义交易对
    test_symbols = [
        {'symbol': 'BTC/USDT:USDT', 'volume': 1000000},
        {'symbol': 'ETH/USDT:USDT', 'volume': 800000},
        {'symbol': 'BNB/USDT:USDT', 'volume': 600000}
    ]

    print(f'使用预定义交易对: {len(test_symbols)} 个')
    for symbol in test_symbols:
        print(f'  - {symbol["symbol"]}')

    monitor_memory()

    # 测试第一个交易对下载（1周数据）
    print('\n=== 测试BTC/USDT:USDT下载（1周数据）===')
    symbol = test_symbols[0]['symbol']
    print(f'测试交易对: {symbol}')

    try:
        start_time = time.time()
        df = fetch_kline_data(symbol, '2024-12-01', '2024-12-07')  # 下载1周数据
        end_time = time.time()
        
        print(f'下载完成: {len(df)} 条记录，耗时: {end_time - start_time:.2f}秒')
        print(f'时间范围: {df["timestamp"].min()} 到 {df["timestamp"].max()}')
        print(f'列名: {list(df.columns)}')
        
        monitor_memory()
        
        # 模块2: 峰谷算法
        print('\n=== 模块2: 峰谷算法 ===')
        from peak_valley import identify_high_low_points, deduplicate_peaks_valleys
        
        print('执行峰谷识别...')
        hl_df = identify_high_low_points(df, g=3)
        print(f'峰谷识别完成，数据点: {len(hl_df)}')
        print(f'高点数量: {hl_df["is_high"].sum()}')
        print(f'低点数量: {hl_df["is_low"].sum()}')
        
        # 峰谷去重
        print('执行峰谷去重...')
        peak_valley_df = deduplicate_peaks_valleys(hl_df)
        print(f'峰谷去重完成，峰谷点: {len(peak_valley_df)}')
        
        monitor_memory()
        
        # 模块3: 关键价位
        print('\n=== 模块3: 关键价位 ===')
        from atr_calculator import calculate_atr
        from attraction_score import calculate_attraction_score
        from key_levels import filter_key_levels
        from extended_levels import calculate_extended_levels
        
        # ATR计算
        print('计算ATR...')
        atr_df = calculate_atr(df, period=120)
        print(f'ATR计算完成')
        print(f'ATR范围: {atr_df["atr"].min():.6f} - {atr_df["atr"].max():.6f}')
        
        monitor_memory()
        
        # 吸引力得分计算
        print('计算吸引力得分...')
        attraction_df = calculate_attraction_score(atr_df, w=120, band_width=0.1)
        print(f'吸引力得分计算完成')
        print(f'吸引力得分范围: {attraction_df["attraction_score"].min():.6f} - {attraction_df["attraction_score"].max():.6f}')
        
        monitor_memory()
        
        # 关键价位筛选
        print('筛选关键价位...')
        key_levels_df = filter_key_levels(attraction_df, threshold=0.1)
        print(f'关键价位筛选完成')
        
        # 检查是否有关键价位
        if 'key_1' in key_levels_df.columns:
            key_count = key_levels_df["key_1"].notna().sum()
            print(f'关键价位数量: {key_count}')
            
            if key_count > 0:
                # 扩展价位计算
                print('计算扩展价位...')
                extended_df = calculate_extended_levels(key_levels_df, key_1_col='key_1', key_0_col='key_0')
                print(f'扩展价位计算完成')
                print(f'扩展价位数量: {extended_df["level_2"].notna().sum()}')
            else:
                print('没有找到关键价位，跳过扩展价位计算')
                extended_df = key_levels_df
        else:
            print('关键价位列不存在，跳过扩展价位计算')
            extended_df = key_levels_df
        
        monitor_memory()
        
        print('\n=== 执行结果摘要 ===')
        print(f'数据点总数: {len(extended_df)}')
        print(f'峰谷点数量: {len(peak_valley_df)}')
        if 'key_1' in extended_df.columns:
            print(f'关键价位数量: {extended_df["key_1"].notna().sum()}')
        if 'level_2' in extended_df.columns:
            print(f'扩展价位数量: {extended_df["level_2"].notna().sum()}')
        
        # 保存结果
        print('\n=== 保存结果 ===')
        os.makedirs('data', exist_ok=True)
        extended_df.to_csv('data/test_results.csv', index=False)
        print('结果已保存到 data/test_results.csv')
        
    except Exception as e:
        print(f'测试失败: {e}')
        import traceback
        traceback.print_exc()

    print('\n✅ 完整测试完成')
    print(f'最终内存状态:')
    monitor_memory()

if __name__ == "__main__":
    main()
