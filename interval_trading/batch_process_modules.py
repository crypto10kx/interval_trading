#!/usr/bin/env python3
"""
批量处理模块1-3：基于本地CSV执行HL识别、ATR计算、吸引力得分、关键价位和扩展价位
"""

import os
import sys
import pandas as pd
import numpy as np
import psutil
import time
from datetime import datetime

# 添加src路径
sys.path.append('src')

def monitor_memory():
    """监控内存使用情况"""
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    total_gb = memory.total / (1024**3)
    percent = memory.percent
    print(f'内存使用: {used_gb:.2f}GB / {total_gb:.2f}GB ({percent:.1f}%)')
    return used_gb < 8.0

def process_single_symbol(symbol, csv_file):
    """处理单个交易对的所有模块"""
    print(f'\n=== 处理 {symbol} ===')
    print(f'文件: {csv_file}')
    
    start_time = time.time()
    
    try:
        # 1. 加载CSV数据
        print('  - 加载CSV数据...')
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f'  - 数据量: {len(df)} 条记录')
        print(f'  - 时间范围: {df["timestamp"].min()} 到 {df["timestamp"].max()}')
        
        # 检查是否已有对数价格列
        if 'log_high' not in df.columns:
            print('  - 警告: 未找到对数价格列，跳过此文件')
            return False, 0, 0.0
        
        # 2. 模块1: HL识别与去重
        print('  - 执行模块1: HL识别与去重...')
        from peak_valley import identify_high_low_points, deduplicate_peaks_valleys
        
        # HL识别
        hl_df = identify_high_low_points(df, g=3)
        print(f'    - 识别到 {hl_df["is_high"].sum()} 个高点, {hl_df["is_low"].sum()} 个低点')
        
        # HL去重
        dedup_df = deduplicate_peaks_valleys(hl_df)
        print(f'    - 去重后: {dedup_df["is_high"].sum()} 个高点, {dedup_df["is_low"].sum()} 个低点')
        
        # 3. 模块2: ATR计算
        print('  - 执行模块2: ATR计算...')
        from atr_calculator import calculate_atr
        atr_df = calculate_atr(dedup_df, period=14)
        print(f'    - ATR计算完成，平均ATR: {atr_df["atr"].mean():.6f}')
        
        # 4. 模块3: 吸引力得分计算
        print('  - 执行模块3: 吸引力得分计算...')
        from attraction_score import calculate_attraction_score
        score_df = calculate_attraction_score(atr_df, w=120, band_width=0.1)
        print(f'    - 吸引力得分计算完成')
        
        # 5. 关键价位识别
        print('  - 执行关键价位识别...')
        from key_levels import filter_key_levels
        key_levels_df = filter_key_levels(score_df, threshold=0.1)
        
        if len(key_levels_df) > 0 and 'key_1' in key_levels_df.columns:
            print(f'    - 识别到 {len(key_levels_df)} 个关键价位')
            if not key_levels_df['key_1'].isna().all():
                print(f'    - 关键价位范围: {key_levels_df["key_1"].min():.6f} 到 {key_levels_df["key_1"].max():.6f}')
            else:
                print('    - 关键价位数据为空')
        else:
            print('    - 未识别到关键价位')
        
        # 6. 扩展价位计算
        if len(key_levels_df) > 0 and 'key_1' in key_levels_df.columns and not key_levels_df['key_1'].isna().all():
            print('  - 执行扩展价位计算...')
            from extended_levels import calculate_extended_levels
            extended_df = calculate_extended_levels(key_levels_df)
            print(f'    - 扩展价位计算完成，生成 {len(extended_df)} 个扩展价位')
        else:
            print('  - 跳过扩展价位计算（无关键价位）')
            extended_df = key_levels_df
        
        # 7. 保存结果
        output_file = csv_file.replace('.csv', '_processed.csv')
        extended_df.to_csv(output_file, index=False)
        print(f'  - 结果已保存到: {output_file}')
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f'✅ {symbol} 处理完成:')
        print(f'  - 耗时: {duration:.2f} 秒')
        print(f'  - 输出文件: {output_file}')
        
        return True, len(extended_df), duration
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f'❌ {symbol} 处理失败: {e}')
        print(f'  - 耗时: {duration:.2f} 秒')
        return False, 0, duration

def main():
    print('=== 批量处理模块1-3 ===')
    print(f'开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'初始内存状态:')
    monitor_memory()
    
    # 获取所有CSV文件
    data_dir = 'data'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_5m.csv')]
    csv_files.sort()
    
    print(f'\n发现 {len(csv_files)} 个CSV文件:')
    for i, file in enumerate(csv_files):
        print(f'  {i+1}. {file}')
    
    if not csv_files:
        print('❌ 未找到CSV文件')
        return
    
    # 统计变量
    total_processed = 0
    total_duration = 0
    successful_processes = 0
    failed_processes = 0
    
    # 逐个处理
    for i, csv_file in enumerate(csv_files):
        print(f'\n{"="*60}')
        print(f'进度: {i+1}/{len(csv_files)} - {csv_file}')
        print(f'{"="*60}')
        
        # 监控内存
        if not monitor_memory():
            print('⚠️ 内存使用过高，停止处理')
            break
        
        # 提取交易对名称
        symbol = csv_file.replace('_5m.csv', '').replace('_', '/')
        
        # 处理单个交易对
        success, records, duration = process_single_symbol(symbol, os.path.join(data_dir, csv_file))
        
        # 更新统计
        if success:
            successful_processes += 1
            total_processed += records
        else:
            failed_processes += 1
        
        total_duration += duration
        
        # 显示进度
        print(f'\n📊 当前进度统计:')
        print(f'  - 已完成: {i+1}/{len(csv_files)} 个交易对')
        print(f'  - 成功: {successful_processes} 个')
        print(f'  - 失败: {failed_processes} 个')
        print(f'  - 总处理记录: {total_processed:,} 条')
        print(f'  - 总耗时: {total_duration:.2f} 秒')
        print(f'  - 平均耗时: {total_duration/(i+1):.2f} 秒/交易对')
        
        # 短暂休息
        if i < len(csv_files) - 1:
            print(f'  - 等待0.5秒后继续...')
            time.sleep(0.5)
    
    # 最终统计
    print(f'\n{"="*60}')
    print(f'🎉 批量处理完成!')
    print(f'{"="*60}')
    print(f'总交易对数: {len(csv_files)}')
    print(f'成功处理: {successful_processes} 个')
    print(f'处理失败: {failed_processes} 个')
    print(f'总处理记录: {total_processed:,} 条')
    print(f'总耗时: {total_duration:.2f} 秒')
    print(f'平均耗时: {total_duration/len(csv_files):.2f} 秒/交易对')
    print(f'结束时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    # 检查输出文件
    print(f'\n📁 输出文件检查:')
    processed_files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')]
    print(f'处理结果文件数量: {len(processed_files)}')
    for file in processed_files:
        file_path = os.path.join(data_dir, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        print(f'  - {file}: {file_size:.2f} MB')
    
    print(f'\n最终内存状态:')
    monitor_memory()

if __name__ == "__main__":
    main()

