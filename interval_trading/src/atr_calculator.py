"""
ATR计算模块 - 任务3.1: 计算ATR
为w根K线计算标准ATR，周期与w同步
使用标准ATR公式：TR=max(high-low, |high-close_prev|, |low-close_prev|)，ATR为w周期均值
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Union
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


def calculate_tr(df: pd.DataFrame) -> pd.Series:
    """
    计算True Range (TR) - 基于对数价格
    
    TR = max(log_high-log_low, |log_high-log_close_prev|, |log_low-log_close_prev|)
    
    Args:
        df: 包含log_high, log_low, log_close列的DataFrame
        
    Returns:
        包含TR值的Series
    """
    try:
        # 计算前一日收盘价（对数）
        log_close_prev = df['log_close'].shift(1)
        
        # 计算三个TR组成部分（在对数空间中）
        hl = df['log_high'] - df['log_low']  # 最高价 - 最低价
        hc = abs(df['log_high'] - log_close_prev)  # |最高价 - 前收盘价|
        lc = abs(df['log_low'] - log_close_prev)   # |最低价 - 前收盘价|
        
        # 取三者最大值作为TR
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        
        logger.debug(f"TR计算完成，数据点数量: {len(tr)}")
        return tr
        
    except Exception as e:
        logger.error(f"TR计算失败: {e}")
        raise


def calculate_atr(df: pd.DataFrame, period: int = 120) -> pd.DataFrame:
    """
    计算ATR (Average True Range) - 基于对数价格
    
    ATR = TR的period周期移动平均
    
    Args:
        df: 包含high, low, close列的DataFrame
        period: ATR计算周期，默认120
        
    Returns:
        添加了atr列的DataFrame
    """
    try:
        # 输入验证
        if df.empty:
            raise ValueError("输入DataFrame为空")
        
        if len(df) < period:
            logger.warning(f"数据长度({len(df)})小于ATR周期({period})，将使用可用数据计算")
            period = len(df)
        
        # 检查必要列
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要列: {missing_columns}")
        
        # 添加对数价格列
        result_df = add_log_prices(df)
        
        # 计算TR（基于对数价格）
        logger.info(f"开始计算TR，数据点数量: {len(result_df)}")
        tr = calculate_tr(result_df)
        
        # 计算ATR (TR的period周期移动平均)
        logger.info(f"开始计算ATR，周期: {period}")
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        # 添加ATR列
        result_df['atr'] = atr
        
        # 统计信息
        valid_atr_count = atr.notna().sum()
        logger.info(f"ATR计算完成:")
        logger.info(f"- 数据点总数: {len(result_df)}")
        logger.info(f"- 有效ATR值: {valid_atr_count}")
        logger.info(f"- ATR范围: {atr.min():.6f} - {atr.max():.6f}")
        logger.info(f"- 平均ATR: {atr.mean():.6f}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"ATR计算失败: {e}")
        raise


def process_single_symbol_atr(symbol: str, period: int = 120) -> bool:
    """
    处理单个交易对的ATR计算
    
    Args:
        symbol: 交易对符号
        period: ATR计算周期
        
    Returns:
        处理是否成功
    """
    try:
        # 构建文件路径 - 处理期货交易对格式 ETH/USDT:USDT -> ETH_USDT:USDT
        kline_file = f"data/{symbol.replace('/', '_')}_5m.csv"
        
        if not os.path.exists(kline_file):
            logger.warning(f"K线文件不存在: {kline_file}")
            return False
        
        logger.info(f"开始处理 {symbol} 的ATR计算")
        
        # 读取K线数据
        df = pd.read_csv(kline_file)
        logger.info(f"读取K线数据: {len(df)} 条记录")
        
        # 转换时间戳列
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 计算ATR
        df_with_atr = calculate_atr(df, period)
        
        # 保存结果
        output_file = f"data/{symbol.replace('/', '_')}_atr.csv"
        df_with_atr.to_csv(output_file, index=False)
        
        logger.info(f"✓ {symbol} ATR计算完成，结果保存到: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"处理 {symbol} ATR计算失败: {e}")
        return False


def main():
    """
    主函数：为所有交易对计算ATR
    """
    try:
        logger.info("开始执行任务3.1：计算ATR")
        
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
            if process_single_symbol_atr(symbol, period=120):
                success_count += 1
        
        logger.info("=" * 50)
        logger.info("任务3.1完成 - ATR计算结果:")
        logger.info("=" * 50)
        logger.info(f"成功处理: {success_count}/{len(symbols)} 个交易对")
        logger.info("ATR文件已保存到 data/*_atr.csv")
        
        return success_count == len(symbols)
        
    except Exception as e:
        logger.error(f"任务3.1执行失败: {e}")
        raise


def test_atr_calculation():
    """
    测试ATR计算功能
    """
    try:
        logger.info("开始ATR计算测试")
        
        # 创建测试数据
        test_data = {
            'timestamp': pd.date_range('2023-01-01', periods=200, freq='5min'),
            'open': np.random.uniform(100, 200, 200),
            'high': np.random.uniform(150, 250, 200),
            'low': np.random.uniform(50, 150, 200),
            'close': np.random.uniform(100, 200, 200),
            'volume': np.random.uniform(1000, 10000, 200)
        }
        
        df = pd.DataFrame(test_data)
        
        # 确保high >= low
        df['high'] = np.maximum(df['high'], df['low'])
        
        # 计算ATR
        df_with_atr = calculate_atr(df, period=120)
        
        # 验证结果
        assert 'atr' in df_with_atr.columns, "ATR列未添加"
        assert not df_with_atr['atr'].isna().all(), "ATR值全为NaN"
        assert df_with_atr['atr'].min() >= 0, "ATR值不应为负数"
        
        logger.info("✓ ATR计算测试通过")
        return True
        
    except Exception as e:
        logger.error(f"ATR计算测试失败: {e}")
        return False


if __name__ == "__main__":
    # 运行测试
    if test_atr_calculation():
        logger.info("测试通过，开始执行主任务")
        success = main()
        
        if success:
            print("✓ 任务3.1测试通过")
        else:
            print("✗ 任务3.1执行失败")
    else:
        print("✗ ATR计算测试失败")
