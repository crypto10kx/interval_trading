#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析15个交易对的range统计
使用w=120, band_width=0.1, threshold=0.075
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
from datetime import datetime

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from atr_calculator import calculate_atr
from attraction_score import calculate_attraction_score
from key_levels import filter_key_levels
from extended_levels import calculate_extended_levels

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_single_symbol(symbol: str, data_dir: str = "data") -> Dict:
    """
    分析单个交易对的range统计
    
    Args:
        symbol: 交易对符号
        data_dir: 数据目录
        
    Returns:
        包含range统计信息的字典
    """
    logger.info(f"开始分析: {symbol}")
    
    try:
        # 加载数据
        file_path = os.path.join(data_dir, f"{symbol}_5m_processed.csv")
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        logger.info(f"数据形状: {df.shape}")
        
        # 检查必要的列
        required_columns = ['log_open', 'log_high', 'log_low', 'log_close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"缺少必要列: {missing_columns}")
            return None
        
        # 检查是否已经处理过
        if 'key_level' in df.columns and 'atr' in df.columns and 'attraction_score' in df.columns:
            logger.info(f"使用已处理的数据: {symbol}")
        else:
            # 如果没有volume列，使用volume_zscore或创建虚拟volume
            if 'volume' not in df.columns:
                if 'volume_zscore' in df.columns:
                    df['volume'] = df['volume_zscore'] * 1000
                else:
                    df['volume'] = 1000
            
            # 参数设置
            w = 120
            band_width = 0.1
            threshold = 0.075
            
            # 计算ATR
            df = calculate_atr(df)
            
            # 计算吸引力得分
            df = calculate_attraction_score(df, w=w, band_width=band_width)
            
            # 筛选关键价位
            df = filter_key_levels(df, threshold=threshold)
            
            # 计算扩展价位
            df = calculate_extended_levels(df)
        
        # 统计range数量
        key_levels = df['key_level'].dropna()
        range_count = len(key_levels[key_levels.isin(['key_1', 'key_0'])]) // 2
        
        # 统计有效扩展价位
        extended_levels = ['log_level_-2', 'log_level_-1', 'log_level_2', 'log_level_3']
        valid_extended = {}
        for level in extended_levels:
            if level in df.columns:
                valid_count = df[level].notna().sum()
                valid_extended[level] = valid_count
            else:
                valid_extended[level] = 0
        
        # 获取价格范围
        price_min = np.exp(df['log_low'].min())
        price_max = np.exp(df['log_high'].max())
        price_range = price_max - price_min
        
        result = {
            'symbol': symbol,
            'total_klines': len(df),
            'range_count': range_count,
            'price_min': price_min,
            'price_max': price_max,
            'price_range': price_range,
            'valid_extended_levels': valid_extended,
            'data_quality': 'good' if range_count > 0 else 'poor'
        }
        
        logger.info(f"✓ {symbol} 分析完成: {range_count}个range")
        return result
        
    except Exception as e:
        logger.error(f"✗ {symbol} 分析失败: {e}")
        return None

def analyze_all_symbols(data_dir: str = "data") -> List[Dict]:
    """
    分析所有交易对
    
    Args:
        data_dir: 数据目录
        
    Returns:
        所有交易对的分析结果列表
    """
    logger.info("开始分析所有交易对")
    
    # 获取所有交易对文件
    symbol_files = []
    for file in os.listdir(data_dir):
        if file.endswith('_5m_processed.csv'):
            symbol = file.replace('_5m_processed.csv', '')
            symbol_files.append(symbol)
    
    logger.info(f"找到 {len(symbol_files)} 个交易对文件")
    
    results = []
    for symbol in symbol_files:
        result = analyze_single_symbol(symbol, data_dir)
        if result is not None:
            results.append(result)
    
    return results

def generate_report(results: List[Dict]) -> str:
    """
    生成分析报告
    
    Args:
        results: 分析结果列表
        
    Returns:
        报告字符串
    """
    if not results:
        return "没有可用的分析结果"
    
    # 统计信息
    total_symbols = len(results)
    total_ranges = sum(r['range_count'] for r in results)
    avg_ranges = total_ranges / total_symbols if total_symbols > 0 else 0
    
    # 按range数量排序
    sorted_results = sorted(results, key=lambda x: x['range_count'], reverse=True)
    
    # 生成报告
    report = f"""
# 区间交易系统 Range 分析报告

## 分析参数
- **观察窗口 (w)**: 120
- **带宽参数 (band_width)**: 0.1
- **阈值 (threshold)**: 0.075
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 总体统计
- **交易对数量**: {total_symbols}
- **总Range数量**: {total_ranges}
- **平均Range数量**: {avg_ranges:.2f}

## 详细结果

| 排名 | 交易对 | Range数量 | 价格范围 | 数据质量 | 扩展价位有效性 |
|------|--------|-----------|----------|----------|----------------|
"""
    
    for i, result in enumerate(sorted_results, 1):
        extended_info = ", ".join([f"{k}: {v}" for k, v in result['valid_extended_levels'].items()])
        report += f"| {i} | {result['symbol']} | {result['range_count']} | {result['price_range']:.2f} | {result['data_quality']} | {extended_info} |\n"
    
    # 添加统计摘要
    range_counts = [r['range_count'] for r in results]
    report += f"""
## 统计摘要
- **最大Range数量**: {max(range_counts)}
- **最小Range数量**: {min(range_counts)}
- **Range数量中位数**: {np.median(range_counts):.2f}
- **Range数量标准差**: {np.std(range_counts):.2f}

## 数据质量分布
- **优秀 (Range > 10)**: {len([r for r in results if r['range_count'] > 10])} 个交易对
- **良好 (Range 5-10)**: {len([r for r in results if 5 <= r['range_count'] <= 10])} 个交易对
- **一般 (Range 1-4)**: {len([r for r in results if 1 <= r['range_count'] <= 4])} 个交易对
- **较差 (Range = 0)**: {len([r for r in results if r['range_count'] == 0])} 个交易对

## 建议
1. 重点关注Range数量较多的交易对进行进一步分析
2. 对于Range数量为0的交易对，需要检查数据质量或调整参数
3. 建议对Range数量>5的交易对进行手动验证
"""
    
    return report

def main():
    """主函数"""
    logger.info("开始执行Range分析任务")
    
    # 分析所有交易对
    results = analyze_all_symbols("data")
    
    if not results:
        logger.error("没有找到任何有效的分析结果")
        return
    
    # 生成报告
    report = generate_report(results)
    
    # 保存报告
    report_file = "range_analysis_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"报告已保存到: {report_file}")
    
    # 打印摘要
    print("\n" + "="*50)
    print("Range分析摘要")
    print("="*50)
    for result in results:
        print(f"{result['symbol']}: {result['range_count']}个range")
    
    print(f"\n总计: {sum(r['range_count'] for r in results)}个range")

if __name__ == "__main__":
    main()
