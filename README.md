# 区间交易系统 (Interval Trading System)

一个基于机器学习的加密货币区间交易系统，专注于识别关键价位和交易机会。

## 项目概述

本项目实现了一个完整的加密货币区间交易系统，包括数据获取、技术指标计算、关键价位识别等核心功能。

## 技术栈

- **Python 3.13** - 主要编程语言
- **Poetry** - 依赖管理
- **CCXT** - 加密货币交易所API
- **Pandas + NumPy** - 数据处理和向量化计算
- **向量化优化** - 针对大数据集性能优化

## 核心模块

### 数据获取模块
- `src/data_acquisition.py` - 获取期货交易对数据
- `src/kline_fetcher.py` - 下载K线数据

### 技术指标模块
- `src/atr_calculator.py` - ATR (平均真实波幅) 计算
- `src/attraction_score.py` - 吸引力得分计算
- `src/key_levels.py` - 关键价位筛选

### 峰谷识别模块
- `src/peak_valley.py` - HL算法峰谷识别

## 安装和使用

1. 克隆仓库
```bash
git clone https://github.com/crypto10kx/interval_trading.git
cd interval_trading
```

2. 安装依赖
```bash
cd interval_trading
poetry install
```

3. 运行模块
```bash
# 数据获取
python src/data_acquisition.py

# ATR计算
python src/atr_calculator.py

# 吸引力得分计算
python src/attraction_score.py

# 关键价位筛选
python src/key_levels.py
```

## 项目结构

```
interval_trading/
├── src/                    # 核心模块
│   ├── data_acquisition.py # 数据获取
│   ├── kline_fetcher.py    # K线数据获取
│   ├── atr_calculator.py   # ATR计算
│   ├── attraction_score.py # 吸引力得分
│   ├── key_levels.py       # 关键价位筛选
│   └── peak_valley.py      # 峰谷识别
├── pyproject.toml          # 项目配置
└── README.md              # 项目说明
```

## 功能特性

- **期货数据支持** - 支持Binance永续合约数据
- **向量化计算** - 高性能数据处理
- **关键价位识别** - 基于吸引力得分的智能筛选
- **内存优化** - 针对8G RAM优化
- **模块化设计** - 易于扩展和维护

## 开发进度

- ✅ 数据获取模块
- ✅ ATR计算模块  
- ✅ 吸引力得分计算
- ✅ 关键价位筛选
- 🔄 峰谷识别模块 (开发中)

## 注意事项

- 数据文件较大，已从仓库中排除
- 需要配置API密钥才能获取实时数据
- 建议在虚拟环境中运行

## 许可证

MIT License
