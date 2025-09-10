#!/usr/bin/env python3
"""
动态Range计算 - 考虑key_1和key_0的动态更新
"""

import sys
sys.path.append('src')

from key_levels import filter_key_levels
from extended_levels import calculate_extended_levels
import pandas as pd
import numpy as np
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_dynamic_range_count(df, w=60, band_width=0.1, threshold=0.075):
    """
    计算动态Range数量
    
    正确的逻辑：
    1. 观察窗口跟随K线移动，每根新K线都会重新计算key_1和key_0
    2. 当扩展价位被触及时，key_1和key_0停止更新，Range计数+1
    3. Range计数后，需要等待w根K线，然后重新开启观察窗口
    """
    try:
        # 1. 计算吸引力得分（不计算关键价位，因为我们要动态计算）
        from attraction_score import calculate_attraction_score
        result_df = calculate_attraction_score(df, w=w, band_width=band_width)
        
        # 2. 计算ATR
        from atr_calculator import calculate_atr
        result_df = calculate_atr(result_df, period=120)
        
        range_count = 0
        current_window_start = 0
        range_active = False
        range_completed = False  # Range是否已完成（key_1和key_0停止更新）
        wait_count = 0  # 等待计数器
        
        for i in range(w, len(result_df)):
            # 如果Range已完成，需要等待w根K线
            if range_completed:
                wait_count += 1
                if wait_count >= w:
                    # 等待完成，重新开启观察窗口
                    range_completed = False
                    range_active = False
                    current_window_start = i
                    wait_count = 0
                    logger.info(f"等待完成，重新开启观察窗口 at index {i}")
                continue
            
            # 检查当前窗口是否有足够的候选点
            window_df = result_df.iloc[current_window_start:i]
            
            # 获取符合条件的候选点
            qualified_points = window_df[window_df['attraction_score'] >= threshold]
            
            if len(qualified_points) < 2:
                continue
            
            # 构建候选点列表
            candidates = []
            for idx, row in qualified_points.iterrows():
                candidates.append({
                    'idx': idx,
                    'log_price': row['log_high'],
                    'price': np.exp(row['log_high']),
                    'score': row['attraction_score'],
                    'type': 'high'
                })
                candidates.append({
                    'idx': idx,
                    'log_price': row['log_low'],
                    'price': np.exp(row['log_low']),
                    'score': row['attraction_score'],
                    'type': 'low'
                })
            
            if len(candidates) < 2:
                continue
            
            # 两阶段筛选算法
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # 第一阶段：选择得分最高且高于阈值的点
            first_candidate = None
            for candidate in candidates:
                if candidate['score'] >= threshold:
                    first_candidate = candidate
                    break
            
            if first_candidate is None:
                continue
            
            # 获取ATR用于区间排除
            last_atr = result_df['atr'].iloc[i]
            
            # 区间排除：排除掉在区间[候选关键价位-ATR, 候选关键价位+ATR]内的所有候选点
            first_price = first_candidate['price']
            remaining_candidates = []
            
            for candidate in candidates:
                if candidate['idx'] == first_candidate['idx']:
                    continue
                
                if abs(candidate['price'] - first_price) > last_atr:
                    remaining_candidates.append(candidate)
            
            if len(remaining_candidates) == 0:
                continue
            
            # 第二阶段：从区间外的剩余候选点中选择得分最高且高于阈值的点
            remaining_candidates.sort(key=lambda x: x['score'], reverse=True)
            second_candidate = remaining_candidates[0]
            
            if second_candidate['score'] < threshold:
                continue
            
            # 确定key_1和key_0
            if first_candidate['price'] >= second_candidate['price']:
                key_1_log = first_candidate['log_price']
                key_0_log = second_candidate['log_price']
            else:
                key_1_log = second_candidate['log_price']
                key_0_log = first_candidate['log_price']
            
            if not range_active:
                # 开始一个新的range
                range_active = True
                logger.info(f"Range 在索引 {i} 处开始")
            
            # 动态计算扩展价位
            log_price_diff = key_1_log - key_0_log
            level_minus_2 = key_0_log + log_price_diff * -2
            level_minus_1 = key_0_log + log_price_diff * -1
            level_2 = key_0_log + log_price_diff * 2
            level_3 = key_0_log + log_price_diff * 3
            
            # 检查扩展价位是否被触及
            current_high = result_df.iloc[i]['log_high']
            current_low = result_df.iloc[i]['log_low']
            
            extended_touched = False
            extended_levels = [level_minus_2, level_minus_1, level_2, level_3]
            level_names = ['level_-2', 'level_-1', 'level_2', 'level_3']
            
            for level_name, level_value in zip(level_names, extended_levels):
                # 检查是否被触及（价格在扩展价位附近）
                if (current_high >= level_value - 0.01 and current_high <= level_value + 0.01) or \
                   (current_low >= level_value - 0.01 and current_low <= level_value + 0.01):
                    extended_touched = True
                    logger.info(f"扩展价位 {level_name} = {level_value:.6f} 在索引 {i} 处被触及")
                    break
            
            if extended_touched:
                # 扩展价位被触及，计数一次range并停止更新key_1和key_0
                range_count += 1
                range_active = False
                range_completed = True
                wait_count = 0
                logger.info(f"Range {range_count} 在索引 {i} 处完成，key_1和key_0停止更新，等待{w}根K线")
        
        return range_count
        
    except Exception as e:
        logger.error(f"计算动态Range数量失败: {e}")
        return 0

def main():
    """测试动态Range计算"""
    # 测试所有交易对
    symbols = ['BTC_USDT_USDT', 'ETH_USDT_USDT', 'SOL_USDT_USDT', 'LTC_USDT_USDT', 'XRP_USDT_USDT']
    
    print('动态Range计算测试:')
    print('交易对 | Range数量')
    print('-' * 30)
    
    for symbol in symbols:
        df = pd.read_csv(f'data/{symbol}_5m_processed.csv')
        range_count = calculate_dynamic_range_count(df, w=60, threshold=0.05)
        print(f'{symbol:12s} | {range_count:8d}')

if __name__ == "__main__":
    main()
