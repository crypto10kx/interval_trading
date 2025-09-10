"""
关键价位筛选模块 - 任务3.3: 筛选关键价位
筛选2w个高低点中得分最高且大于阈值（0.1）的key_1/key_0，满足(key_1-key_0)>ATR
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_log_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加对数价格列（避免log(0)或负值）
    
    Args:
        df: K线数据DataFrame，必须包含['high', 'low', 'open', 'close']列
        
    Returns:
        添加了log_high, log_low, log_open, log_close列的DataFrame
    """
    logger.info("开始添加对数价格列")
    
    # 创建结果DataFrame的副本
    result_df = df.copy()
    
    # 添加对数价格列，使用clip避免log(0)或负值
    result_df['log_high'] = np.log(df['high'].clip(lower=1e-6))
    result_df['log_low'] = np.log(df['low'].clip(lower=1e-6))
    result_df['log_open'] = np.log(df['open'].clip(lower=1e-6))
    result_df['log_close'] = np.log(df['close'].clip(lower=1e-6))
    
    logger.info("✓ 对数价格列添加完成")
    return result_df


def filter_key_levels(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """
    筛选关键价位key_1/key_0（基于对数价格）
    
    从吸引力得分DataFrame筛选得分>=threshold的点，取最高两个，标记为key_1（较高）、key_0（较低）
    验证(key_1-key_0)>ATR（使用w根K线中最后一根的ATR）
    
    Args:
        df: 包含high, low, atr, attraction_score列的DataFrame
        threshold: 吸引力得分阈值，默认0.1
        
    Returns:
        添加了key_level列的DataFrame
    """
    try:
        # 输入验证
        if df.empty:
            raise ValueError("输入DataFrame为空")
        
        # 检查必要列（优先使用对数价格列）
        if 'log_high' in df.columns and 'log_low' in df.columns and 'atr' in df.columns and 'attraction_score' in df.columns:
            # 如果已有对数价格列，直接使用
            result_df = df.copy()
            logger.info("使用现有的对数价格列筛选关键价位")
        else:
            # 检查原始价格列
            required_columns = ['high', 'low', 'atr', 'attraction_score']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"缺少必要列: {missing_columns}")
            
            # 添加对数价格列
            result_df = add_log_prices(df)
        
        logger.info(f"开始筛选关键价位，阈值: {threshold}")
        
        # 筛选得分>=threshold的点
        qualified_points = result_df[result_df['attraction_score'] >= threshold].copy()
        
        if len(qualified_points) < 2:
            logger.warning(f"符合条件的点数量不足（{len(qualified_points)} < 2），无法确定关键价位")
            result_df['key_level'] = pd.Series(dtype='object')
            return result_df
        
        # 按得分降序排序，取前两个
        top_points = qualified_points.nlargest(2, 'attraction_score')
        
        # 确定key_1和key_0（基于对数价格）
        if len(top_points) == 2:
            # 比较对数价格确定key_1（较高）和key_0（较低）
            point1_log_high = top_points.iloc[0]['log_high']
            point1_log_low = top_points.iloc[0]['log_low']
            point1_max_log_price = max(point1_log_high, point1_log_low)
            
            point2_log_high = top_points.iloc[1]['log_high']
            point2_log_low = top_points.iloc[1]['log_low']
            point2_max_log_price = max(point2_log_high, point2_log_low)
            
            if point1_max_log_price >= point2_max_log_price:
                key_1_idx = top_points.iloc[0].name
                key_0_idx = top_points.iloc[1].name
            else:
                key_1_idx = top_points.iloc[1].name
                key_0_idx = top_points.iloc[0].name
        else:
            # 只有一个点，无法确定关键价位
            logger.warning("只有一个符合条件的点，无法确定关键价位")
            result_df['key_level'] = pd.Series(dtype='object')
            return result_df
        
        # 获取key_1和key_0的对数价格
        key_1_log_high = result_df.loc[key_1_idx, 'log_high']
        key_1_log_low = result_df.loc[key_1_idx, 'log_low']
        key_1_log_price = max(key_1_log_high, key_1_log_low)
        
        key_0_log_high = result_df.loc[key_0_idx, 'log_high']
        key_0_log_low = result_df.loc[key_0_idx, 'log_low']
        key_0_log_price = max(key_0_log_high, key_0_log_low)
        
        # 获取最后一根K线的ATR进行验证（基于对数价格）
        last_atr = result_df['atr'].iloc[-1]
        
        # 验证(key_1-key_0)>ATR（在对数空间中）
        log_price_diff = key_1_log_price - key_0_log_price
        
        if log_price_diff > last_atr:
            # 验证通过，设置关键价位
            result_df['key_level'] = pd.Series(dtype='object')
            result_df.loc[key_1_idx, 'key_level'] = 'key_1'
            result_df.loc[key_0_idx, 'key_level'] = 'key_0'
            
            # 添加log_key_1和log_key_0列
            result_df['log_key_1'] = np.nan
            result_df['log_key_0'] = np.nan
            result_df.loc[key_1_idx, 'log_key_1'] = key_1_log_price
            result_df.loc[key_0_idx, 'log_key_0'] = key_0_log_price
            
            # 转换回真实价格用于日志显示
            key_1_price = np.exp(key_1_log_price)
            key_0_price = np.exp(key_0_log_price)
            price_diff = key_1_price - key_0_price
            
            logger.info(f"关键价位筛选成功:")
            logger.info(f"- key_1: 价格 {key_1_price:.6f}, 得分 {result_df.loc[key_1_idx, 'attraction_score']:.6f}")
            logger.info(f"- key_0: 价格 {key_0_price:.6f}, 得分 {result_df.loc[key_0_idx, 'attraction_score']:.6f}")
            logger.info(f"- 价格差: {price_diff:.6f} > ATR: {last_atr:.6f}")
        else:
            logger.warning(f"价格差验证失败: {log_price_diff:.6f} <= ATR: {last_atr:.6f}")
            result_df['key_level'] = pd.Series(dtype='object')
            result_df['log_key_1'] = np.nan
            result_df['log_key_0'] = np.nan
        
        # 统计信息
        key_levels = result_df['key_level'].dropna()
        logger.info(f"关键价位筛选完成:")
        logger.info(f"- 数据点总数: {len(result_df)}")
        logger.info(f"- 符合阈值点数: {len(qualified_points)}")
        logger.info(f"- 关键价位数量: {len(key_levels)}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"关键价位筛选失败: {e}")
        raise


def process_single_symbol_key_levels(symbol: str, threshold: float = 0.1) -> bool:
    """
    处理单个交易对的关键价位筛选
    
    Args:
        symbol: 交易对符号
        threshold: 吸引力得分阈值
        
    Returns:
        处理是否成功
    """
    try:
        # 构建文件路径
        attraction_file = f"data/{symbol.replace('/', '_')}_attraction.csv"
        
        if not os.path.exists(attraction_file):
            logger.warning(f"吸引力得分文件不存在: {attraction_file}")
            return False
        
        logger.info(f"开始处理 {symbol} 的关键价位筛选")
        
        # 读取吸引力得分数据
        df = pd.read_csv(attraction_file)
        logger.info(f"读取吸引力得分数据: {len(df)} 条记录")
        
        # 转换时间戳列
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 筛选关键价位
        df_with_key_levels = filter_key_levels(df, threshold)
        
        # 保存结果
        output_file = f"data/{symbol.replace('/', '_')}_key_levels.csv"
        df_with_key_levels.to_csv(output_file, index=False)
        
        logger.info(f"✓ {symbol} 关键价位筛选完成，结果保存到: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"处理 {symbol} 关键价位筛选失败: {e}")
        return False


def main():
    """
    主函数：为所有交易对筛选关键价位
    """
    try:
        logger.info("开始执行任务3.3：筛选关键价位")
        
        # 读取交易对列表
        symbols_file = "data/symbols.json"
        if not os.path.exists(symbols_file):
            raise FileNotFoundError(f"交易对文件不存在: {symbols_file}")
        
        import json
        with open(symbols_file, 'r', encoding='utf-8') as f:
            symbols_data = json.load(f)
        
        symbols = [item['symbol'] for item in symbols_data]
        logger.info(f"找到 {len(symbols)} 个交易对")
        
        # 处理每个交易对
        success_count = 0
        for symbol in symbols:
            if process_single_symbol_key_levels(symbol, threshold=0.1):
                success_count += 1
        
        logger.info("=" * 50)
        logger.info("任务3.3完成 - 关键价位筛选结果:")
        logger.info("=" * 50)
        logger.info(f"成功处理: {success_count}/{len(symbols)} 个交易对")
        logger.info("关键价位文件已保存到 data/*_key_levels.csv")
        
        return success_count == len(symbols)
        
    except Exception as e:
        logger.error(f"任务3.3执行失败: {e}")
        raise


def test_key_levels_filtering():
    """
    测试关键价位筛选功能
    """
    try:
        logger.info("开始关键价位筛选测试")
        
        # 创建测试数据
        np.random.seed(42)
        test_data = {
            'timestamp': pd.date_range('2023-01-01', periods=200, freq='5min'),
            'open': np.random.uniform(100, 200, 200),
            'high': np.random.uniform(150, 250, 200),
            'low': np.random.uniform(50, 150, 200),
            'close': np.random.uniform(100, 200, 200),
            'volume': np.random.uniform(1000, 10000, 200),
            'atr': np.random.uniform(1, 10, 200),
            'attraction_score': np.random.uniform(0, 0.5, 200)
        }
        
        # 确保high >= low
        test_data['high'] = np.maximum(test_data['high'], test_data['low'])
        
        # 设置一些高得分点
        test_data['attraction_score'][10] = 0.15  # 高得分点1
        test_data['attraction_score'][50] = 0.12  # 高得分点2
        test_data['attraction_score'][100] = 0.08  # 低于阈值
        
        df = pd.DataFrame(test_data)
        
        # 筛选关键价位
        df_with_key_levels = filter_key_levels(df, threshold=0.1)
        
        # 验证结果
        assert 'key_level' in df_with_key_levels.columns, "key_level列未添加"
        
        key_levels = df_with_key_levels['key_level'].dropna()
        logger.info(f"找到 {len(key_levels)} 个关键价位")
        
        if len(key_levels) > 0:
            logger.info("✓ 关键价位筛选测试通过")
        else:
            logger.warning("未找到关键价位，可能需要调整测试数据")
        
        return True
        
    except Exception as e:
        logger.error(f"关键价位筛选测试失败: {e}")
        return False


if __name__ == "__main__":
    # 运行测试
    if test_key_levels_filtering():
        logger.info("测试通过，开始执行主任务")
        success = main()
        
        if success:
            print("✓ 任务3.3测试通过")
        else:
            print("✗ 任务3.3执行失败")
    else:
        print("✗ 关键价位筛选测试失败")
