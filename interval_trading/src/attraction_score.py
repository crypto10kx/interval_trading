"""
吸引力得分计算模块 - 任务3.2: 计算吸引力得分
为w根K线的2w个高低点计算吸引力得分（指数衰减）
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Union, Tuple
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


def calculate_attraction_score(df: pd.DataFrame, w: int = 120, 
                             band_width: float = 0.1) -> pd.DataFrame:
    """
    计算吸引力得分（基于对数价格）
    
    对每个high_n/low_n，统计w-1根K线的高低点在[price-band_width*ATR, price+band_width*ATR]内
    得分公式：score=exp(-|price-target_price|/ATR)，总分/w，范围[0,1]
    同一K线只计最高得分（high/low择一）
    
    Args:
        df: 包含high, low, atr列的DataFrame
        w: K线窗口大小，默认120
        band_width: ATR带宽系数，默认0.1
        
    Returns:
        添加了attraction_score列的DataFrame
    """
    try:
        # 输入验证
        if df.empty:
            raise ValueError("输入DataFrame为空")
        
        if len(df) < w:
            logger.warning(f"数据长度({len(df)})小于窗口大小({w})，将使用可用数据计算")
            w = len(df)
        
        # 检查必要列（优先使用对数价格列）
        if 'log_high' in df.columns and 'log_low' in df.columns and 'atr' in df.columns:
            # 如果已有对数价格列，直接使用
            result_df = df.copy()
            logger.info("使用现有的对数价格列计算吸引力得分")
        else:
            # 检查原始价格列
            required_columns = ['high', 'low', 'atr']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"缺少必要列: {missing_columns}")
            
            # 添加对数价格列
            result_df = add_log_prices(df)
        
        logger.info(f"开始计算吸引力得分，窗口大小: {w}, 带宽系数: {band_width}")
        
        # 初始化吸引力得分列
        result_df['attraction_score'] = 0.0
        
        # 使用优化的向量化方法计算吸引力得分（基于对数价格）
        attraction_scores = _calculate_attraction_scores_optimized(
            result_df['log_high'].values, 
            result_df['log_low'].values, 
            result_df['atr'].values, 
            w, 
            band_width
        )
        
        result_df['attraction_score'] = attraction_scores
        
        # 统计信息
        valid_scores = attraction_scores[~np.isnan(attraction_scores)]
        logger.info(f"吸引力得分计算完成:")
        logger.info(f"- 数据点总数: {len(result_df)}")
        logger.info(f"- 有效得分: {len(valid_scores)}")
        logger.info(f"- 得分范围: {valid_scores.min():.6f} - {valid_scores.max():.6f}")
        logger.info(f"- 平均得分: {valid_scores.mean():.6f}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"吸引力得分计算失败: {e}")
        raise


def _calculate_attraction_scores_optimized(high: np.ndarray, low: np.ndarray, 
                                         atr: np.ndarray, w: int, 
                                         band_width: float) -> np.ndarray:
    """
    优化的吸引力得分计算算法
    
    使用滑动窗口和向量化操作，避免创建大型距离矩阵
    算法复杂度：O(n×w)，其中w是固定窗口大小
    
    算法逻辑：
    1. 对于每个K线，使用滑动窗口计算其吸引力得分
    2. 每个点的价格区间：[price - band_width*ATR, price + band_width*ATR]
    3. 统计窗口内剩余K线的高低点中，有多少个落在这个区间内
    4. 对于每个落在区间内的价格点，计算得分：exp(-|price-target_price|/ATR)
    5. 每个K线只计一次最高得分（high/low择一）
    6. 最终得分 = 所有得分总和 / w
    
    Args:
        high: 最高价数组
        low: 最低价数组
        atr: ATR数组
        w: 窗口大小
        band_width: 带宽系数
        
    Returns:
        吸引力得分数组
    """
    n = len(high)
    scores = np.zeros(n)
    
    # 处理ATR为0或NaN的情况
    atr_safe = np.where(np.isnan(atr) | (atr == 0), 1.0, atr)
    
    for i in range(n):
        # 计算窗口范围
        start_idx = max(0, i - w + 1)
        end_idx = i + 1
        window_size = end_idx - start_idx
        
        if window_size < 2:
            scores[i] = 0.0
            continue
        
        # 获取窗口数据
        window_high = high[start_idx:end_idx]
        window_low = low[start_idx:end_idx]
        window_atr = atr_safe[start_idx:end_idx]
        
        # 当前目标价格和ATR
        target_price = high[i] if i < len(high) else low[i]
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


def _calculate_attraction_scores_broadcast(target_prices: np.ndarray, 
                                          all_high: np.ndarray, 
                                          all_low: np.ndarray, 
                                          all_atr: np.ndarray, 
                                          w: int, 
                                          band_width: float) -> np.ndarray:
    """
    使用NumPy广播完全向量化计算吸引力得分
    
    使用NumPy广播加速计算：
    1. 完全向量化，无for循环
    2. 使用广播计算所有价格点之间的距离
    3. 使用向量化条件判断和计算
    
    Args:
        target_prices: 目标价格数组
        all_high: 所有最高价数组
        all_low: 所有最低价数组
        all_atr: 所有ATR数组
        w: 窗口大小
        band_width: 带宽系数
        
    Returns:
        吸引力得分数组
    """
    n = len(target_prices)
    scores = np.zeros(n)
    
    # 处理ATR为0或NaN的情况
    atr_safe = np.where(np.isnan(all_atr) | (all_atr == 0), 1.0, all_atr)
    
    # 创建所有候选价格点（high和low）
    all_candidates = np.concatenate([all_high, all_low])
    
    # 使用广播计算距离矩阵
    # shape: (n, 2*n) - 每个目标价格对所有候选价格的距离
    dist_matrix = np.abs(target_prices[:, np.newaxis] - all_candidates)
    
    # 计算ATR矩阵（对应每个候选价格）
    atr_matrix = np.concatenate([atr_safe, atr_safe])
    
    # 计算带宽矩阵
    bandwidth_matrix = band_width * atr_matrix
    
    # 创建掩码：距离在带宽内的点
    mask = dist_matrix <= bandwidth_matrix
    
    # 计算权重：exp(-距离/ATR)
    weights = np.exp(-dist_matrix / atr_matrix)
    
    # 应用掩码
    masked_weights = weights * mask
    
    # 计算得分：每行（每个目标价格）的权重和
    scores = np.sum(masked_weights, axis=1) / w
    
    # 确保得分在[0,1]范围内
    scores = np.clip(scores, 0, 1)
    
    return scores


def _calculate_attraction_scores_fully_vectorized(target_prices: np.ndarray, 
                                                all_high: np.ndarray, 
                                                all_low: np.ndarray, 
                                                all_atr: np.ndarray, 
                                                w: int, 
                                                band_width: float) -> np.ndarray:
    """
    完全向量化计算吸引力得分（仅保留外部for循环）
    
    使用NumPy广播和向量化操作实现：
    1. 外部for循环遍历每个K线
    2. 内部所有计算完全向量化，无for循环
    3. 使用广播计算所有价格点之间的距离
    4. 使用向量化条件判断和计算
    
    Args:
        target_prices: 目标价格数组
        all_high: 所有最高价数组
        all_low: 所有最低价数组
        all_atr: 所有ATR数组
        w: 窗口大小
        band_width: 带宽系数
        
    Returns:
        吸引力得分数组
    """
    n = len(target_prices)
    scores = np.zeros(n)
    
    # 外部for循环：遍历每个K线
    for i in range(n):
        # 计算窗口范围：[i-w+1, i]，确保窗口大小始终为w
        start_idx = max(0, i - w + 1)
        end_idx = i + 1
        
        # 实际窗口大小
        window_size = end_idx - start_idx
        
        # 获取当前窗口内的数据
        window_high = all_high[start_idx:end_idx]
        window_low = all_low[start_idx:end_idx]
        window_atr = all_atr[start_idx:end_idx]
        
        # 当前目标价格和ATR
        target_price = target_prices[i]
        target_atr = all_atr[i]
        
        # 计算价格区间
        bandwidth = band_width * target_atr
        lower_bound = target_price - bandwidth
        upper_bound = target_price + bandwidth
        
        # 创建掩码：排除当前K线（向量化）
        mask = np.ones(len(window_high), dtype=bool)
        # 排除当前K线（当前K线在窗口中的相对位置）
        current_relative_idx = i - start_idx
        if 0 <= current_relative_idx < len(window_high):
            mask[current_relative_idx] = False
        
        # 应用掩码（向量化）
        window_high_masked = window_high[mask]
        window_low_masked = window_low[mask]
        window_atr_masked = window_atr[mask]
        
        if len(window_high_masked) == 0:
            scores[i] = 0.0
            continue
        
        # 计算high点的得分（完全向量化）
        high_in_range = (window_high_masked >= lower_bound) & (window_high_masked <= upper_bound)
        high_distances = np.abs(window_high_masked - target_price)
        high_scores = np.where(high_in_range, 
                              np.exp(-high_distances / window_atr_masked), 
                              0.0)
        
        # 计算low点的得分（完全向量化）
        low_in_range = (window_low_masked >= lower_bound) & (window_low_masked <= upper_bound)
        low_distances = np.abs(window_low_masked - target_price)
        low_scores = np.where(low_in_range, 
                             np.exp(-low_distances / window_atr_masked), 
                             0.0)
        
        # 每个K线只计一次最高得分（high/low择一）（向量化）
        kline_scores = np.maximum(high_scores, low_scores)
        
        # 计算最终得分：总分 / w（向量化）
        total_score = np.sum(kline_scores)
        final_score = total_score / w
        
        scores[i] = min(final_score, 1.0)  # 确保不超过1
    
    return scores


def _calculate_point_attraction_score_correct(target_price: float, all_high: np.ndarray,
                                            all_low: np.ndarray, all_atr: np.ndarray,
                                            start_idx: int, end_idx: int, current_idx: int,
                                            band_width: float, w: int) -> float:
    """
    计算单个点的吸引力得分（正确算法）
    
    算法逻辑：
    1. 计算目标价格的价格区间：[target_price - band_width*ATR, target_price + band_width*ATR]
    2. 统计剩余的w-1根K线的2w-2个价格点中，有多少个落在这个区间内
    3. 对于每个落在区间内的价格点，计算得分：exp(-|price-target_price|/ATR)
    4. 每个K线只计一次最高得分（high/low择一）
    5. 最终得分 = 所有得分总和 / w
    
    Args:
        target_price: 目标价格
        all_high: 所有最高价数组
        all_low: 所有最低价数组
        all_atr: 所有ATR数组
        start_idx: 窗口开始索引
        end_idx: 窗口结束索引
        current_idx: 当前K线索引
        band_width: 带宽系数
        w: 窗口大小
        
    Returns:
        吸引力得分
    """
    if end_idx - start_idx <= 1:
        return 0.0
    
    # 计算目标价格的价格区间
    target_atr = all_atr[current_idx]
    bandwidth = band_width * target_atr
    lower_bound = target_price - bandwidth
    upper_bound = target_price + bandwidth
    
    # 统计在价格区间内的点数
    total_score = 0.0
    kline_count = 0  # 统计有多少根K线贡献了得分
    
    # 遍历窗口内的K线（排除当前K线）
    for j in range(start_idx, end_idx):
        if j == current_idx:
            continue  # 跳过当前K线
        
        kline_high = all_high[j]
        kline_low = all_low[j]
        kline_atr = all_atr[j]
        
        # 计算当前K线的high和low点的得分
        high_score = 0.0
        low_score = 0.0
        
        # 检查high点是否在区间内
        if lower_bound <= kline_high <= upper_bound:
            distance = abs(kline_high - target_price)
            if kline_atr > 0:
                high_score = np.exp(-distance / kline_atr)
        
        # 检查low点是否在区间内
        if lower_bound <= kline_low <= upper_bound:
            distance = abs(kline_low - target_price)
            if kline_atr > 0:
                low_score = np.exp(-distance / kline_atr)
        
        # 每个K线只计一次最高得分（high/low择一）
        kline_max_score = max(high_score, low_score)
        if kline_max_score > 0:
            total_score += kline_max_score
            kline_count += 1
    
    # 计算最终得分：总分 / w
    if kline_count > 0:
        final_score = total_score / w
        return min(final_score, 1.0)  # 确保不超过1
    
    return 0.0


def _calculate_point_attraction_score(target_price: float, window_high: np.ndarray,
                                    window_low: np.ndarray, window_atr: np.ndarray,
                                    band_width: float) -> float:
    """
    计算单个点的吸引力得分（旧版本，保留用于兼容性）
    
    Args:
        target_price: 目标价格
        window_high: 窗口内最高价数组
        window_low: 窗口内最低价数组
        window_atr: 窗口内ATR数组
        band_width: 带宽系数
        
    Returns:
        吸引力得分
    """
    if len(window_high) == 0:
        return 0.0
    
    # 计算带宽
    bandwidth = band_width * window_atr
    
    # 计算价格区间 [target_price - bandwidth, target_price + bandwidth]
    lower_bound = target_price - bandwidth
    upper_bound = target_price + bandwidth
    
    # 统计在价格区间内的点数
    in_range_count = 0
    total_score = 0.0
    
    for j in range(len(window_high)):
        # 检查high点是否在区间内
        if lower_bound[j] <= window_high[j] <= upper_bound[j]:
            distance = abs(window_high[j] - target_price)
            if window_atr[j] > 0:
                score = np.exp(-distance / window_atr[j])
                total_score += score
                in_range_count += 1
        
        # 检查low点是否在区间内
        if lower_bound[j] <= window_low[j] <= upper_bound[j]:
            distance = abs(window_low[j] - target_price)
            if window_atr[j] > 0:
                score = np.exp(-distance / window_atr[j])
                total_score += score
                in_range_count += 1
    
    # 计算平均得分
    if in_range_count > 0:
        avg_score = total_score / in_range_count
        # 归一化到[0,1]范围
        normalized_score = min(avg_score, 1.0)
        return normalized_score
    
    return 0.0


def process_single_symbol_attraction(symbol: str, w: int = 120, 
                                   band_width: float = 0.1) -> bool:
    """
    处理单个交易对的吸引力得分计算
    
    Args:
        symbol: 交易对符号
        w: 窗口大小
        band_width: 带宽系数
        
    Returns:
        处理是否成功
    """
    try:
        # 构建文件路径
        atr_file = f"data/{symbol.replace('/', '_')}_atr.csv"
        
        if not os.path.exists(atr_file):
            logger.warning(f"ATR文件不存在: {atr_file}")
            return False
        
        logger.info(f"开始处理 {symbol} 的吸引力得分计算")
        
        # 读取ATR数据
        df = pd.read_csv(atr_file)
        logger.info(f"读取ATR数据: {len(df)} 条记录")
        
        # 转换时间戳列
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 计算吸引力得分
        df_with_attraction = calculate_attraction_score(df, w, band_width)
        
        # 保存结果
        output_file = f"data/{symbol.replace('/', '_')}_attraction.csv"
        df_with_attraction.to_csv(output_file, index=False)
        
        logger.info(f"✓ {symbol} 吸引力得分计算完成，结果保存到: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"处理 {symbol} 吸引力得分计算失败: {e}")
        return False


def main():
    """
    主函数：为所有交易对计算吸引力得分
    """
    try:
        logger.info("开始执行任务3.2：计算吸引力得分")
        
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
            if process_single_symbol_attraction(symbol, w=120, band_width=0.1):
                success_count += 1
        
        logger.info("=" * 50)
        logger.info("任务3.2完成 - 吸引力得分计算结果:")
        logger.info("=" * 50)
        logger.info(f"成功处理: {success_count}/{len(symbols)} 个交易对")
        logger.info("吸引力得分文件已保存到 data/*_attraction.csv")
        
        return success_count == len(symbols)
        
    except Exception as e:
        logger.error(f"任务3.2执行失败: {e}")
        raise


def test_attraction_calculation():
    """
    测试吸引力得分计算功能
    """
    try:
        logger.info("开始吸引力得分计算测试")
        
        # 创建测试数据
        np.random.seed(42)
        test_data = {
            'timestamp': pd.date_range('2023-01-01', periods=200, freq='5min'),
            'open': np.random.uniform(100, 200, 200),
            'high': np.random.uniform(150, 250, 200),
            'low': np.random.uniform(50, 150, 200),
            'close': np.random.uniform(100, 200, 200),
            'volume': np.random.uniform(1000, 10000, 200),
            'atr': np.random.uniform(1, 10, 200)
        }
        
        # 确保high >= low
        test_data['high'] = np.maximum(test_data['high'], test_data['low'])
        
        df = pd.DataFrame(test_data)
        
        # 计算吸引力得分
        df_with_attraction = calculate_attraction_score(df, w=120, band_width=0.1)
        
        # 验证结果
        assert 'attraction_score' in df_with_attraction.columns, "吸引力得分列未添加"
        assert not df_with_attraction['attraction_score'].isna().all(), "吸引力得分值全为NaN"
        assert (df_with_attraction['attraction_score'] >= 0).all(), "吸引力得分不应为负数"
        assert (df_with_attraction['attraction_score'] <= 1).all(), "吸引力得分不应超过1"
        
        logger.info("✓ 吸引力得分计算测试通过")
        return True
        
    except Exception as e:
        logger.error(f"吸引力得分计算测试失败: {e}")
        return False


if __name__ == "__main__":
    # 运行测试
    if test_attraction_calculation():
        logger.info("测试通过，开始执行主任务")
        success = main()
        
        if success:
            print("✓ 任务3.2测试通过")
        else:
            print("✗ 任务3.2执行失败")
    else:
        print("✗ 吸引力得分计算测试失败")
