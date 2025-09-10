"""
峰谷算法模块 - 任务2.1: 实现HL算法
对K线数据计算摆动高点H和低点L（g=3），标记HL点
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

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


def identify_high_low_points(df: pd.DataFrame, g: int = 3) -> pd.DataFrame:
    """
    识别K线数据中的摆动高点H和低点L（基于对数价格）
    
    Args:
        df: K线数据DataFrame，必须包含['high', 'low']列
        g: 摆动参数，默认3（需要前后g根K线确认）
        
    Returns:
        添加了is_high和is_low列的DataFrame
    """
    try:
        # 验证输入数据
        if df.empty:
            raise ValueError("输入数据为空")
        
        required_columns = ['high', 'low']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"缺少必要列: {required_columns}")
        
        logger.info(f"开始识别HL点，数据量: {len(df)} 条，g={g}")
        
        # 添加对数价格列
        result_df = add_log_prices(df)
        
        # 初始化HL标记列
        result_df['is_high'] = 0
        result_df['is_low'] = 0
        
        # 计算窗口大小
        window_size = 2 * g + 1
        
        # 确保数据量足够
        if len(result_df) < window_size:
            logger.warning(f"数据量不足，需要至少{window_size}条记录")
            return result_df
        
        # 使用rolling窗口识别高点（基于对数价格）
        # 高点：当前K线的高点高于前后g根K线的高点
        high_rolling_max = result_df['log_high'].rolling(window=window_size, center=True)
        high_max_values = high_rolling_max.max()
        
        # 标记高点：当前高点等于窗口内最大值，且不是边界值
        # 确保索引在有效范围内（避免边界问题）
        valid_indices = (result_df.index >= g) & (result_df.index < len(result_df) - g)
        
        is_high_condition = (
            (result_df['log_high'] == high_max_values) &  # 当前高点等于窗口最大值
            (result_df['log_high'] > result_df['log_high'].shift(g)) &  # 高于前g根K线
            (result_df['log_high'] > result_df['log_high'].shift(-g)) &  # 高于后g根K线
            valid_indices  # 确保在有效范围内
        )
        
        # 使用rolling窗口识别低点（基于对数价格）
        # 低点：当前K线的低点低于前后g根K线的低点
        low_rolling_min = result_df['log_low'].rolling(window=window_size, center=True)
        low_min_values = low_rolling_min.min()
        
        # 标记低点：当前低点等于窗口内最小值，且不是边界值
        is_low_condition = (
            (result_df['log_low'] == low_min_values) &  # 当前低点等于窗口最小值
            (result_df['log_low'] < result_df['log_low'].shift(g)) &  # 低于前g根K线
            (result_df['log_low'] < result_df['log_low'].shift(-g)) &  # 低于后g根K线
            valid_indices  # 确保在有效范围内
        )
        
        # 应用条件标记
        result_df.loc[is_high_condition, 'is_high'] = 1
        result_df.loc[is_low_condition, 'is_low'] = 1
        
        # 确保最新K线[n-g,n]内HL不用于特征提取（避免数据泄露）
        # 将最后g根K线的HL标记设为0
        result_df.iloc[-g:, result_df.columns.get_loc('is_high')] = 0
        result_df.iloc[-g:, result_df.columns.get_loc('is_low')] = 0
        
        # 统计结果
        high_count = result_df['is_high'].sum()
        low_count = result_df['is_low'].sum()
        
        logger.info(f"HL点识别完成:")
        logger.info(f"- 高点数量: {high_count}")
        logger.info(f"- 低点数量: {low_count}")
        logger.info(f"- 最后{g}根K线已排除，避免数据泄露")
        
        return result_df
        
    except Exception as e:
        logger.error(f"HL点识别失败: {e}")
        raise


def validate_hl_points(df: pd.DataFrame, g: int = 3) -> bool:
    """
    验证HL点识别的正确性
    
    Args:
        df: 包含HL标记的DataFrame
        g: 摆动参数
        
    Returns:
        验证是否通过
    """
    try:
        # 检查列是否存在
        if 'is_high' not in df.columns or 'is_low' not in df.columns:
            logger.error("缺少is_high或is_low列")
            return False
        
        # 检查最后g根K线是否被正确排除
        last_g_highs = df['is_high'].iloc[-g:].sum()
        last_g_lows = df['is_low'].iloc[-g:].sum()
        
        if last_g_highs > 0 or last_g_lows > 0:
            logger.error(f"最后{g}根K线中仍有HL点，数据泄露风险")
            return False
        
        # 检查HL点是否满足g条件
        high_indices = df[df['is_high'] == 1].index
        low_indices = df[df['is_low'] == 1].index
        
        # 验证高点
        for idx in high_indices:
            if idx < g or idx >= len(df) - g:
                continue  # 跳过边界点
            
            current_high = df.loc[idx, 'high']
            # 检查前后g根K线的高点
            prev_highs = df.loc[idx-g:idx-1, 'high']
            next_highs = df.loc[idx+1:idx+g, 'high']
            
            if not (current_high > prev_highs.max() and current_high > next_highs.max()):
                logger.error(f"高点验证失败，索引: {idx}")
                return False
        
        # 验证低点
        for idx in low_indices:
            if idx < g or idx >= len(df) - g:
                continue  # 跳过边界点
            
            current_low = df.loc[idx, 'low']
            # 检查前后g根K线的低点
            prev_lows = df.loc[idx-g:idx-1, 'low']
            next_lows = df.loc[idx+1:idx+g, 'low']
            
            if not (current_low < prev_lows.min() and current_low < next_lows.min()):
                logger.error(f"低点验证失败，索引: {idx}")
                return False
        
        logger.info("HL点验证通过")
        return True
        
    except Exception as e:
        logger.error(f"HL点验证失败: {e}")
        return False


def process_single_symbol(symbol: str, data_dir: str = 'data') -> pd.DataFrame:
    """
    处理单个交易对的HL点识别
    
    Args:
        symbol: 交易对符号，如 'BTC/USDT'
        data_dir: 数据目录
        
    Returns:
        包含HL标记的DataFrame
    """
    try:
        # 构建文件路径
        filename = f"{symbol.replace('/', '_')}_5m.csv"
        filepath = f"{data_dir}/{filename}"
        
        logger.info(f"处理交易对: {symbol}")
        
        # 读取K线数据
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"加载数据: {len(df)} 条记录")
        logger.info(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
        
        # 识别HL点
        result_df = identify_high_low_points(df, g=3)
        
        # 验证结果
        if validate_hl_points(result_df, g=3):
            logger.info(f"✓ {symbol} HL点识别成功")
        else:
            logger.warning(f"⚠ {symbol} HL点验证失败")
        
        return result_df
        
    except Exception as e:
        logger.error(f"处理 {symbol} 失败: {e}")
        raise


def deduplicate_peaks_valleys(df: pd.DataFrame) -> pd.DataFrame:
    """
    对HL序列去重，形成H L H L交错的峰谷序列
    
    Args:
        df: 包含is_high和is_low列的DataFrame
        
    Returns:
        峰谷序列DataFrame，包含timestamp, price, is_peak列
    """
    try:
        # 获取所有HL点
        high_points = df[df['is_high'] == 1].copy()
        low_points = df[df['is_low'] == 1].copy()
        
        if high_points.empty and low_points.empty:
            logger.warning("没有找到任何HL点")
            return pd.DataFrame(columns=['timestamp', 'price', 'is_peak', 'is_high', 'is_low', 'log_high', 'log_low', 'log_open', 'log_close', 'volume_zscore'])
        
        # 为高点添加标记（使用对数价格）
        high_points['is_peak'] = 1  # 1表示峰（高点）
        high_points['price'] = high_points['log_high']  # 使用对数价格
        
        # 为低点添加标记（使用对数价格）
        low_points['is_peak'] = 0  # 0表示谷（低点）
        low_points['price'] = low_points['log_low']  # 使用对数价格
        
        # 合并所有HL点，保留原始列
        all_points = pd.concat([
            high_points[['timestamp', 'price', 'is_peak', 'is_high', 'is_low', 'log_high', 'log_low', 'log_open', 'log_close', 'volume_zscore']],
            low_points[['timestamp', 'price', 'is_peak', 'is_high', 'is_low', 'log_high', 'log_low', 'log_open', 'log_close', 'volume_zscore']]
        ], ignore_index=True)
        
        # 按时间排序
        all_points = all_points.sort_values('timestamp').reset_index(drop=True)
        
        if all_points.empty:
            logger.warning("合并后没有HL点")
            return pd.DataFrame(columns=['timestamp', 'price', 'is_peak', 'is_high', 'is_low', 'log_high', 'log_low', 'log_open', 'log_close', 'volume_zscore'])
        
        # 去重逻辑：处理连续的相同类型点
        result_points = []
        current_type = None
        current_group = []
        
        for idx, row in all_points.iterrows():
            point_type = row['is_peak']
            
            if current_type is None:
                # 第一个点
                current_type = point_type
                current_group = [row]
            elif point_type == current_type:
                # 相同类型，加入当前组
                current_group.append(row)
            else:
                # 不同类型，处理当前组并开始新组
                if current_group:
                    # 从当前组中选择最佳点
                    best_point = select_best_point_from_group(current_group, current_type)
                    result_points.append(best_point)
                
                # 开始新组
                current_type = point_type
                current_group = [row]
        
        # 处理最后一组
        if current_group:
            best_point = select_best_point_from_group(current_group, current_type)
            result_points.append(best_point)
        
        # 转换为DataFrame
        if not result_points:
            logger.warning("去重后没有峰谷点")
            return pd.DataFrame(columns=['timestamp', 'price', 'is_peak', 'is_high', 'is_low', 'log_high', 'log_low', 'log_open', 'log_close', 'volume_zscore'])
        
        result_df = pd.DataFrame(result_points)
        
        # 确保峰谷交错：H L H L...
        result_df = ensure_alternating_pattern(result_df)
        
        logger.info(f"峰谷去重完成: {len(result_df)} 个峰谷点")
        logger.info(f"- 峰点数量: {result_df['is_peak'].sum()}")
        logger.info(f"- 谷点数量: {len(result_df) - result_df['is_peak'].sum()}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"峰谷去重失败: {e}")
        raise


def select_best_point_from_group(group: list, point_type: int) -> dict:
    """
    从一组相同类型的点中选择最佳点
    
    Args:
        group: 相同类型的点列表
        point_type: 点类型（1=峰，0=谷）
        
    Returns:
        最佳点的字典
    """
    if not group:
        return None
    
    if len(group) == 1:
        return group[0].to_dict()
    
    if point_type == 1:  # 高点组，选择最高值（相等时取最早）
        best_point = max(group, key=lambda x: (x['price'], -x['timestamp'].timestamp()))
    else:  # 低点组，选择最低值（相等时取最早）
        best_point = min(group, key=lambda x: (x['price'], -x['timestamp'].timestamp()))
    
    return best_point.to_dict()


def ensure_alternating_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保峰谷序列交错：H L H L...
    
    Args:
        df: 峰谷序列DataFrame
        
    Returns:
        修正后的峰谷序列DataFrame
    """
    if df.empty or len(df) <= 1:
        return df
    
    result_points = []
    current_type = None
    
    for idx, row in df.iterrows():
        point_type = row['is_peak']
        
        if current_type is None:
            # 第一个点
            result_points.append(row)
            current_type = point_type
        elif point_type != current_type:
            # 不同类型，添加
            result_points.append(row)
            current_type = point_type
        else:
            # 相同类型，跳过（去重）
            continue
    
    # 处理最后一个峰谷的确认问题
    # 如果最后一个峰谷需要相反类型确认，可能需要调整
    if len(result_points) >= 2:
        last_point = result_points[-1]
        second_last_point = result_points[-2]
        
        # 如果最后两个点类型相同，移除最后一个
        if last_point['is_peak'] == second_last_point['is_peak']:
            result_points = result_points[:-1]
    
    return pd.DataFrame(result_points).reset_index(drop=True)


def validate_peak_valley_sequence(df: pd.DataFrame) -> bool:
    """
    验证峰谷序列的正确性
    
    Args:
        df: 峰谷序列DataFrame
        
    Returns:
        验证是否通过
    """
    try:
        if df.empty:
            logger.warning("峰谷序列为空")
            return False
        
        # 检查列是否存在
        required_columns = ['timestamp', 'price', 'is_peak']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"缺少必要列: {required_columns}")
            return False
        
        # 检查是否按时间排序
        if not df['timestamp'].is_monotonic_increasing:
            logger.error("峰谷序列未按时间排序")
            return False
        
        # 检查峰谷交错模式
        if len(df) > 1:
            for i in range(len(df) - 1):
                if df.iloc[i]['is_peak'] == df.iloc[i + 1]['is_peak']:
                    logger.error(f"峰谷序列不交错，位置 {i} 和 {i+1} 类型相同")
                    return False
        
        # 检查价格合理性（对数价格可以为负值）
        if df['price'].isna().any():
            logger.error("峰谷序列包含无效价格")
            return False
        
        logger.info("峰谷序列验证通过")
        return True
        
    except Exception as e:
        logger.error(f"峰谷序列验证失败: {e}")
        return False


def process_peak_valley_deduplication(symbol: str, data_dir: str = 'data') -> pd.DataFrame:
    """
    处理单个交易对的峰谷去重
    
    Args:
        symbol: 交易对符号
        data_dir: 数据目录
        
    Returns:
        峰谷序列DataFrame
    """
    try:
        # 构建HL文件路径
        hl_filename = f"{symbol.replace('/', '_')}_hl.csv"
        hl_filepath = f"{data_dir}/{hl_filename}"
        
        logger.info(f"处理峰谷去重: {symbol}")
        
        # 读取HL数据
        df = pd.read_csv(hl_filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"加载HL数据: {len(df)} 条记录")
        
        # 执行峰谷去重
        peak_valley_df = deduplicate_peaks_valleys(df)
        
        if peak_valley_df.empty:
            logger.warning(f"{symbol} 峰谷去重后为空")
            return peak_valley_df
        
        # 验证结果
        if validate_peak_valley_sequence(peak_valley_df):
            logger.info(f"✓ {symbol} 峰谷去重成功")
        else:
            logger.warning(f"⚠ {symbol} 峰谷去重验证失败")
        
        return peak_valley_df
        
    except Exception as e:
        logger.error(f"处理 {symbol} 峰谷去重失败: {e}")
        raise


def main():
    """
    主函数：处理所有交易对的HL点识别和峰谷去重
    """
    try:
        import json
        
        # 加载交易对列表
        with open('data/symbols.json', 'r', encoding='utf-8') as f:
            symbols = json.load(f)
        
        logger.info(f"开始处理 {len(symbols)} 个交易对的HL点识别和峰谷去重")
        
        results = {}
        
        for i, symbol_info in enumerate(symbols, 1):
            symbol = symbol_info['symbol']
            
            try:
                logger.info(f"处理第 {i}/{len(symbols)} 个交易对: {symbol}")
                
                # 步骤1：处理HL点识别
                result_df = process_single_symbol(symbol)
                
                # 保存HL结果
                hl_output_file = f"data/{symbol.replace('/', '_')}_hl.csv"
                result_df.to_csv(hl_output_file, index=False)
                
                # 步骤2：处理峰谷去重
                peak_valley_df = process_peak_valley_deduplication(symbol)
                
                # 保存峰谷去重结果
                pv_output_file = f"data/{symbol.replace('/', '_')}_peaks_valleys.csv"
                peak_valley_df.to_csv(pv_output_file, index=False)
                
                # 统计信息
                high_count = result_df['is_high'].sum()
                low_count = result_df['is_low'].sum()
                peak_count = peak_valley_df['is_peak'].sum() if not peak_valley_df.empty else 0
                valley_count = len(peak_valley_df) - peak_count if not peak_valley_df.empty else 0
                
                results[symbol] = {
                    'high_count': high_count,
                    'low_count': low_count,
                    'peak_count': peak_count,
                    'valley_count': valley_count,
                    'total_records': len(result_df),
                    'hl_output_file': hl_output_file,
                    'pv_output_file': pv_output_file
                }
                
                logger.info(f"✓ {symbol} 完成 - 高点: {high_count}, 低点: {low_count}, 峰: {peak_count}, 谷: {valley_count}")
                
            except Exception as e:
                logger.error(f"✗ {symbol} 失败: {e}")
                results[symbol] = None
        
        # 输出汇总
        logger.info("=" * 80)
        logger.info("HL点识别和峰谷去重完成 - 结果汇总:")
        logger.info("=" * 80)
        
        success_count = 0
        for symbol, result in results.items():
            if result:
                logger.info(f"{symbol:<12} | 高点: {result['high_count']:>6} | 低点: {result['low_count']:>6} | 峰: {result['peak_count']:>6} | 谷: {result['valley_count']:>6} | 记录: {result['total_records']:>8}")
                success_count += 1
            else:
                logger.info(f"{symbol:<12} | 处理失败")
        
        logger.info("=" * 80)
        logger.info(f"成功处理: {success_count}/{len(symbols)} 个交易对")
        
        return results
        
    except Exception as e:
        logger.error(f"主函数执行失败: {e}")
        raise


if __name__ == "__main__":
    # 执行主函数
    results = main()
    
    # 验证结果
    success_count = sum(1 for result in results.values() if result is not None)
    assert success_count > 0, "所有交易对HL点识别失败"
    
    print("✓ 任务2.1测试通过")
