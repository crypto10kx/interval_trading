"""
任务3.4: 计算扩展价位

基于key_1/key_0计算扩展价位（-2,-1,2,3）
公式：
- level_-1 = key_0 - (key_1 - key_0)
- level_-2 = key_0 - 2*(key_1 - key_0)  
- level_2 = key_1 + (key_1 - key_0)
- level_3 = key_1 + 2*(key_1 - key_0)

作者: AI Assistant
日期: 2025-09-10
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_extended_levels(df: pd.DataFrame, 
                            key_1_col: str = 'key_1', 
                            key_0_col: str = 'key_0') -> pd.DataFrame:
    """
    计算扩展价位（基于对数空间的斐波那契比例）
    
    基于key_1和key_0计算4个扩展价位（在对数空间中）：
    - level_-2: key_0 - 2*(key_1 - key_0)  # 向下扩展2倍
    - level_-1: key_0 - (key_1 - key_0)    # 向下扩展1倍  
    - level_2: key_1 + (key_1 - key_0)     # 向上扩展1倍
    - level_3: key_1 + 2*(key_1 - key_0)   # 向上扩展2倍
    
    Args:
        df: 包含key_1和key_0列的DataFrame
        key_1_col: key_1列名，默认为'key_1'
        key_0_col: key_0列名，默认为'key_0'
        
    Returns:
        包含扩展价位列的DataFrame
        
    Raises:
        ValueError: 当输入数据无效时
        KeyError: 当缺少必需列时
    """
    logger.info("开始计算扩展价位（对数空间）")
    
    # 输入验证
    if df.empty:
        logger.warning("输入数据为空")
        return df.copy()
    
    if key_1_col not in df.columns:
        raise KeyError(f"缺少必需列: {key_1_col}")
    
    if key_0_col not in df.columns:
        raise KeyError(f"缺少必需列: {key_0_col}")
    
    # 创建结果DataFrame的副本
    result_df = df.copy()
    
    # 获取key_1和key_0列（应该是对数价格）
    key_1 = df[key_1_col]
    key_0 = df[key_0_col]
    
    # 计算对数价格差 (key_1 - key_0)
    log_price_diff = key_1 - key_0
    
    # 计算扩展价位（在对数空间中，向量化操作）
    # level_-2 = key_0 - 2*(key_1 - key_0)
    result_df['level_-2'] = key_0 - 2 * log_price_diff
    
    # level_-1 = key_0 - (key_1 - key_0)  
    result_df['level_-1'] = key_0 - log_price_diff
    
    # level_2 = key_1 + (key_1 - key_0)
    result_df['level_2'] = key_1 + log_price_diff
    
    # level_3 = key_1 + 2*(key_1 - key_0)
    result_df['level_3'] = key_1 + 2 * log_price_diff
    
    # 统计有效扩展价位数量
    valid_levels = result_df[['level_-2', 'level_-1', 'level_2', 'level_3']].notna().sum()
    total_rows = len(result_df)
    
    logger.info("扩展价位计算完成:")
    logger.info(f"- 数据点总数: {total_rows}")
    logger.info(f"- level_-2有效值: {valid_levels['level_-2']}")
    logger.info(f"- level_-1有效值: {valid_levels['level_-1']}")
    logger.info(f"- level_2有效值: {valid_levels['level_2']}")
    logger.info(f"- level_3有效值: {valid_levels['level_3']}")
    
    return result_df


def validate_extended_levels(df: pd.DataFrame) -> bool:
    """
    验证扩展价位数据的合理性
    
    Args:
        df: 包含扩展价位列的DataFrame
        
    Returns:
        验证是否通过
    """
    logger.info("开始验证扩展价位数据")
    
    # 检查必需列是否存在
    required_cols = ['level_-2', 'level_-1', 'level_2', 'level_3']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"缺少扩展价位列: {missing_cols}")
        raise KeyError(f"缺少扩展价位列: {missing_cols}")
    
    # 检查价格层级关系
    # level_-2 < level_-1 < key_0 < key_1 < level_2 < level_3
    valid_mask = df[required_cols].notna().all(axis=1)
    
    if valid_mask.sum() > 0:
        valid_data = df[valid_mask]
        
        # 检查层级关系
        level_order_correct = (
            (valid_data['level_-2'] < valid_data['level_-1']) &
            (valid_data['level_-1'] < valid_data['level_2']) &
            (valid_data['level_2'] < valid_data['level_3'])
        ).all()
        
        if not level_order_correct:
            logger.error("扩展价位层级关系不正确")
            return False
        
        # 检查价格差一致性
        diff_1 = valid_data['level_-1'] - valid_data['level_-2']  # 应该等于key_1-key_0
        diff_2 = valid_data['level_2'] - valid_data['level_-1']   # 应该等于2*(key_1-key_0)
        diff_3 = valid_data['level_3'] - valid_data['level_2']    # 应该等于key_1-key_0
        
        if not np.allclose(diff_1, diff_3, rtol=1e-10):
            logger.error("扩展价位价格差不一致")
            return False
    
    logger.info("✓ 扩展价位数据验证通过")
    return True


def process_key_levels_data(input_file: str, output_file: str) -> None:
    """
    处理关键价位数据文件，计算扩展价位
    
    Args:
        input_file: 输入的关键价位数据文件路径
        output_file: 输出的扩展价位数据文件路径
    """
    logger.info(f"开始处理关键价位数据: {input_file}")
    
    try:
        # 读取关键价位数据
        df = pd.read_csv(input_file)
        logger.info(f"读取数据: {len(df)} 条记录")
        
        # 检查是否有key_level列
        if 'key_level' not in df.columns:
            logger.warning("数据中缺少key_level列，跳过扩展价位计算")
            return
        
        # 从key_level列中提取key_1和key_0的价格
        key_1_mask = df['key_level'] == 'key_1'
        key_0_mask = df['key_level'] == 'key_0'
        
        if not key_1_mask.any() or not key_0_mask.any():
            logger.warning("数据中缺少key_1或key_0标记，跳过扩展价位计算")
            return
        
        # 获取key_1和key_0的对数价格
        key_1_log_price = df.loc[key_1_mask, 'log_high'].iloc[0] if key_1_mask.any() else None
        key_0_log_price = df.loc[key_0_mask, 'log_low'].iloc[0] if key_0_mask.any() else None
        
        if key_1_log_price is None or key_0_log_price is None:
            logger.warning("无法获取key_1或key_0对数价格，跳过扩展价位计算")
            return
        
        # 创建包含key_1和key_0列的DataFrame（使用对数价格）
        df_with_keys = df.copy()
        df_with_keys['key_1'] = key_1_log_price
        df_with_keys['key_0'] = key_0_log_price
        
        # 计算扩展价位
        result_df = calculate_extended_levels(df_with_keys)
        
        # 验证结果
        if not validate_extended_levels(result_df):
            logger.error("扩展价位验证失败")
            return
        
        # 保存结果
        result_df.to_csv(output_file, index=False)
        logger.info(f"✓ 扩展价位计算完成，结果保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"处理关键价位数据时出错: {e}")
        raise


def main():
    """主函数：处理所有交易对的扩展价位计算"""
    logger.info("开始执行任务3.4：计算扩展价位")
    
    # 获取所有关键价位数据文件
    import glob
    import os
    
    data_dir = "data"
    key_levels_files = glob.glob(os.path.join(data_dir, "*_key_levels.csv"))
    
    if not key_levels_files:
        logger.warning("未找到关键价位数据文件")
        return
    
    logger.info(f"找到 {len(key_levels_files)} 个关键价位数据文件")
    
    success_count = 0
    
    for input_file in key_levels_files:
        try:
            # 生成输出文件名
            base_name = os.path.basename(input_file).replace("_key_levels.csv", "")
            output_file = os.path.join(data_dir, f"{base_name}_extended_levels.csv")
            
            # 处理单个文件
            process_key_levels_data(input_file, output_file)
            success_count += 1
            
        except Exception as e:
            logger.error(f"处理文件 {input_file} 时出错: {e}")
            continue
    
    logger.info("=" * 50)
    logger.info("任务3.4完成 - 扩展价位计算结果:")
    logger.info("=" * 50)
    logger.info(f"成功处理: {success_count}/{len(key_levels_files)} 个文件")
    logger.info("扩展价位文件已保存到 data/*_extended_levels.csv")


if __name__ == "__main__":
    main()
