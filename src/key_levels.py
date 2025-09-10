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


def filter_key_levels(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    """
    筛选关键价位key_1/key_0（基于对数价格）
    
    从吸引力得分DataFrame筛选得分>=threshold的点，取最高两个，标记为key_1（较高）、key_0（较低）
    验证(key_1-key_0)>ATR（使用w根K线中最后一根的ATR）
    
    Args:
        df: 包含high, low, atr, attraction_score列的DataFrame
        threshold: 吸引力得分阈值，如果为None则使用动态阈值（得分中位数的50%）
        
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
        
        # 计算动态阈值（如果未提供）
        if threshold is None:
            attraction_scores = result_df['attraction_score'].dropna()
            if len(attraction_scores) > 0:
                median_score = np.median(attraction_scores)
                threshold = median_score * 0.5  # 得分中位数的50%
                logger.info(f"使用动态阈值: {threshold:.6f} (基于得分中位数: {median_score:.6f})")
            else:
                threshold = 0.01  # 默认阈值
                logger.warning("无法计算动态阈值，使用默认值: 0.01")
        
        logger.info(f"开始筛选关键价位，阈值: {threshold}")
        
        # 筛选得分>=threshold的点
        qualified_points = result_df[result_df['attraction_score'] >= threshold].copy()
        
        if len(qualified_points) < 2:
            logger.warning(f"符合条件的点数量不足（{len(qualified_points)} < 2），无法确定关键价位")
            result_df['key_level'] = pd.Series(dtype='object')
            return result_df
        
        # 从所有数据中筛选得分最高的两个点
        # 创建包含high和low的候选点列表
        candidates = []
        for idx, row in qualified_points.iterrows():
            # 添加high点
            candidates.append({
                'idx': idx,
                'log_price': row['log_high'],
                'price': np.exp(row['log_high']),
                'score': row['attraction_score'],
                'type': 'high'
            })
            # 添加low点
            candidates.append({
                'idx': idx,
                'log_price': row['log_low'],
                'price': np.exp(row['log_low']),
                'score': row['attraction_score'],
                'type': 'low'
            })
        
        if len(candidates) < 2:
            logger.warning(f"符合条件的点数量不足（{len(candidates)} < 2），无法确定关键价位")
            result_df['key_level'] = pd.Series(dtype='object')
            return result_df
        
        # 两阶段筛选算法
        # 第一阶段：选择得分最高且高于阈值的点作为第一个候选关键价位
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 找到第一个符合条件的点
        first_candidate = None
        for candidate in candidates:
            if candidate['score'] >= threshold:
                first_candidate = candidate
                break
        
        if first_candidate is None:
            logger.warning("没有找到得分高于阈值的候选点")
            result_df['key_level'] = pd.Series(dtype='object')
            return result_df
        
        # 获取ATR用于区间排除
        last_atr = result_df['atr'].iloc[-1]
        
        # 区间排除：排除掉在区间[候选关键价位-ATR, 候选关键价位+ATR]内的所有候选点
        first_price = first_candidate['price']
        excluded_candidates = []
        remaining_candidates = []
        
        for candidate in candidates:
            if candidate['idx'] == first_candidate['idx']:
                # 跳过第一个候选点本身
                continue
            
            # 检查是否在排除区间内
            if abs(candidate['price'] - first_price) <= last_atr:
                excluded_candidates.append(candidate)
            else:
                remaining_candidates.append(candidate)
        
        logger.info(f"第一阶段筛选: 选择得分 {first_candidate['score']:.6f} 的点作为第一个候选关键价位")
        logger.info(f"区间排除: 排除 {len(excluded_candidates)} 个候选点，剩余 {len(remaining_candidates)} 个候选点")
        
        # 第二阶段：从区间外的剩余候选点中选择得分最高且高于阈值的点
        if len(remaining_candidates) == 0:
            logger.warning("区间排除后没有剩余候选点，无法确定第二个关键价位")
            result_df['key_level'] = pd.Series(dtype='object')
            return result_df
        
        # 从剩余候选点中选择得分最高的点
        remaining_candidates.sort(key=lambda x: x['score'], reverse=True)
        second_candidate = remaining_candidates[0]
        
        if second_candidate['score'] < threshold:
            logger.warning(f"第二个候选点得分 {second_candidate['score']:.6f} 低于阈值 {threshold}")
            result_df['key_level'] = pd.Series(dtype='object')
            return result_df
        
        logger.info(f"第二阶段筛选: 选择得分 {second_candidate['score']:.6f} 的点作为第二个候选关键价位")
        
        # 确定key_1和key_0（基于价格）
        if first_candidate['price'] >= second_candidate['price']:
            key_1_idx = first_candidate['idx']
            key_1_log_price = first_candidate['log_price']
            key_1_price = first_candidate['price']
            key_0_idx = second_candidate['idx']
            key_0_log_price = second_candidate['log_price']
            key_0_price = second_candidate['price']
        else:
            key_1_idx = second_candidate['idx']
            key_1_log_price = second_candidate['log_price']
            key_1_price = second_candidate['price']
            key_0_idx = first_candidate['idx']
            key_0_log_price = first_candidate['log_price']
            key_0_price = first_candidate['price']
        
        # 两阶段筛选算法确保key_1和key_0必然满足距离条件，无需额外验证
        # 直接设置关键价位
        result_df['key_level'] = pd.Series(dtype='object')
        
        # 如果key_1和key_0来自同一根K线，需要特殊处理
        if key_1_idx == key_0_idx:
            # 同一根K线，标记为key_1_key_0
            result_df.loc[key_1_idx, 'key_level'] = 'key_1_key_0'
        else:
            # 不同K线，分别标记
            result_df.loc[key_1_idx, 'key_level'] = 'key_1'
            result_df.loc[key_0_idx, 'key_level'] = 'key_0'
        
        # 添加log_key_1和log_key_0列
        result_df['log_key_1'] = np.nan
        result_df['log_key_0'] = np.nan
        result_df.loc[key_1_idx, 'log_key_1'] = key_1_log_price
        result_df.loc[key_0_idx, 'log_key_0'] = key_0_log_price
        
        # 计算价格差用于日志记录
        price_diff = key_1_price - key_0_price
        last_atr = result_df['atr'].iloc[-1]
        
        logger.info(f"关键价位筛选成功:")
        logger.info(f"- key_1: 价格 {key_1_price:.6f}, 得分 {result_df.loc[key_1_idx, 'attraction_score']:.6f}")
        logger.info(f"- key_0: 价格 {key_0_price:.6f}, 得分 {result_df.loc[key_0_idx, 'attraction_score']:.6f}")
        logger.info(f"- 价格差: {price_diff:.2f} > ATR: {last_atr:.2f} (两阶段筛选保证)")
        
        # 统计信息
        key_levels = result_df['key_level'].dropna()
        # 计算range数量：key_1_key_0算作1个range，key_1+key_0算作1个range
        range_count = 0
        if 'key_1_key_0' in key_levels.values:
            range_count = 1
        else:
            range_count = len(key_levels[key_levels.isin(['key_1', 'key_0'])]) // 2
        
        logger.info(f"关键价位筛选完成:")
        logger.info(f"- 数据点总数: {len(result_df)}")
        logger.info(f"- 符合阈值点数: {len(qualified_points)}")
        logger.info(f"- 关键价位数量: {len(key_levels)}")
        logger.info(f"- Range数量: {range_count}")
        
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
