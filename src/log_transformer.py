"""
全局对数转换模块 - 任务3.5: 全局对数转换
在数据加载后立即转换为对数刻度，更新所有下游计算
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def apply_log_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    全局转换为对数刻度
    
    将价格列转换为对数刻度，避免log(0)或负值
    对成交量进行Z-Score归一化，突出异常值
    
    Args:
        df: K线数据DataFrame，必须包含['open', 'high', 'low', 'close', 'volume']列
        
    Returns:
        添加了log_列和volume_zscore列的DataFrame
        
    Raises:
        ValueError: 当输入数据无效时
        KeyError: 当缺少必需列时
    """
    logger.info("开始全局对数转换")
    
    # 输入验证
    if df.empty:
        raise ValueError("输入DataFrame为空")
    
    # 检查必要列
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"缺少必要列: {missing_columns}")
    
    # 创建结果DataFrame的副本
    result_df = df.copy()
    
    # 转换价格列为对数刻度，使用clip避免log(0)或负值
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        result_df[f'log_{col}'] = np.log(df[col].clip(lower=1e-6))
        logger.debug(f"转换 {col} 为 log_{col}")
    
    # 对成交量进行Z-Score归一化，突出异常值
    volume_mean = df['volume'].mean()
    volume_std = df['volume'].std()
    
    if volume_std > 0:
        result_df['volume_zscore'] = (df['volume'] - volume_mean) / volume_std
    else:
        # 如果标准差为0，所有值设为0
        result_df['volume_zscore'] = 0.0
        logger.warning("成交量标准差为0，volume_zscore设为0")
    
    logger.info("✓ 全局对数转换完成")
    logger.info(f"- 价格列转换: {price_cols}")
    logger.info(f"- 成交量Z-Score归一化完成")
    logger.info(f"- 数据点总数: {len(result_df)}")
    
    return result_df


def validate_log_transformation(df: pd.DataFrame) -> bool:
    """
    验证对数转换结果
    
    Args:
        df: 转换后的DataFrame
        
    Returns:
        验证是否通过
    """
    try:
        # 检查log列是否存在且为正值
        log_cols = ['log_open', 'log_high', 'log_low', 'log_close']
        for col in log_cols:
            if col not in df.columns:
                logger.error(f"缺少log列: {col}")
                return False
            
            if (df[col] <= 0).any():
                logger.error(f"log列 {col} 包含非正值")
                return False
        
        # 检查volume_zscore列
        if 'volume_zscore' not in df.columns:
            logger.error("缺少volume_zscore列")
            return False
        
        # 检查是否有NaN值
        if df[log_cols + ['volume_zscore']].isnull().any().any():
            logger.error("转换结果包含NaN值")
            return False
        
        logger.info("✓ 对数转换验证通过")
        return True
        
    except Exception as e:
        logger.error(f"验证失败: {e}")
        return False


def test_log_transformer():
    """测试对数转换模块"""
    logger.info("开始测试对数转换模块")
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'open': [100, 110, 120, 105, 115],
        'high': [105, 115, 125, 110, 120],
        'low': [95, 105, 115, 100, 110],
        'close': [102, 112, 122, 107, 117],
        'volume': [1000, 2000, 1500, 3000, 2500]
    })
    
    logger.info(f"测试数据: {len(test_data)} 条记录")
    
    # 执行转换
    result = apply_log_scale(test_data)
    
    # 验证结果
    assert validate_log_transformation(result), "对数转换验证失败"
    
    # 检查log列
    log_cols = ['log_open', 'log_high', 'log_low', 'log_close']
    for col in log_cols:
        assert col in result.columns, f"缺少 {col} 列"
        assert (result[col] > 0).all(), f"{col} 列包含非正值"
    
    # 检查volume_zscore列
    assert 'volume_zscore' in result.columns, "缺少 volume_zscore 列"
    assert not result['volume_zscore'].isnull().any(), "volume_zscore 包含NaN"
    
    # 检查原始列是否保留
    original_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in original_cols:
        assert col in result.columns, f"原始列 {col} 丢失"
    
    logger.info("✓ 对数转换模块测试通过")
    return True


if __name__ == "__main__":
    # 运行测试
    test_log_transformer()
    print("所有测试通过")
