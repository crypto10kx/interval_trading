# 区间交易系统 - 项目结构

## 📁 目录结构

```
interval_trading/
├── src/                          # 核心源代码模块
│   ├── data_acquisition.py       # 任务1.1: 获取交易对列表
│   ├── kline_fetcher.py          # 任务1.2: 获取K线数据
│   ├── peak_valley.py            # 任务2.1-2.2: HL算法识别与去重
│   ├── atr_calculator.py         # 任务3.1: ATR计算
│   ├── attraction_score.py       # 任务3.2: 吸引力得分计算
│   ├── key_levels.py             # 任务3.3: 关键价位筛选
│   ├── extended_levels.py        # 任务3.4: 扩展价位计算
│   └── log_transformer.py        # 任务3.5: 对数转换工具
├── scripts/                      # 脚本文件
│   ├── download/                 # 数据下载脚本
│   │   ├── download_data.py      # 基础下载脚本
│   │   ├── optimized_download.py # 优化下载脚本
│   │   ├── quick_download.py     # 快速下载脚本
│   │   └── simple_download.py    # 简单下载脚本
│   ├── test/                     # 测试脚本
│   │   ├── test_attraction_performance.py # 性能测试
│   │   ├── test_full.py          # 完整测试
│   │   ├── test_simple.py        # 简单测试
│   │   ├── test_single_symbol.py # 单交易对测试
│   │   └── run_task_1_2.py       # 任务1-2运行脚本
│   └── debug/                    # 调试脚本
│       └── debug_futures.py      # 期货调试脚本
├── tests/                        # 单元测试
│   ├── test_data_acquisition.py  # 数据获取测试
│   ├── test_kline_fetcher.py     # K线获取测试
│   ├── test_peak_valley.py       # HL算法测试
│   ├── test_atr_calculator.py    # ATR计算测试
│   ├── test_attraction_score.py  # 吸引力得分测试
│   ├── test_key_levels.py        # 关键价位测试
│   └── test_extended_levels.py   # 扩展价位测试
├── data/                         # 数据文件
│   ├── *.csv                     # 原始K线数据
│   └── *_processed.csv           # 处理后的数据
├── models/                       # 模型文件（预留）
├── docs/                         # 文档文件（预留）
├── examples/                     # 示例文件（预留）
├── batch_process_modules.py      # 批量处理主脚本
├── pyproject.toml                # Poetry配置
├── poetry.lock                   # 依赖锁定文件
├── README.md                     # 项目说明
└── PROJECT_STRUCTURE.md          # 项目结构说明
```

## 🎯 核心模块说明

### 数据获取模块 (src/)
- **data_acquisition.py**: 获取Binance前10个USDT期货交易对
- **kline_fetcher.py**: 下载2年5分钟K线数据，自动应用对数转换

### 技术分析模块 (src/)
- **peak_valley.py**: HL算法识别高低点，支持去重
- **atr_calculator.py**: 计算ATR指标（对数空间）
- **attraction_score.py**: 计算吸引力得分（优化算法）
- **key_levels.py**: 筛选关键价位
- **extended_levels.py**: 计算扩展价位

### 工具模块 (src/)
- **log_transformer.py**: 对数价格转换工具

## 🚀 使用方法

### 1. 环境设置
```bash
poetry install
```

### 2. 数据下载
```bash
# 使用优化下载脚本
python scripts/download/optimized_download.py

# 或使用快速下载
python scripts/download/quick_download.py
```

### 3. 批量处理
```bash
# 执行所有模块（1-3）
python batch_process_modules.py
```

### 4. 测试
```bash
# 运行单元测试
poetry run pytest tests/

# 运行性能测试
python scripts/test/test_attraction_performance.py
```

## 📊 数据流程

1. **数据获取** → 下载K线数据 → 应用对数转换
2. **HL识别** → 识别高低点 → 去重处理
3. **ATR计算** → 计算真实波动范围
4. **吸引力得分** → 计算价格吸引力
5. **关键价位** → 筛选重要价位
6. **扩展价位** → 计算斐波那契扩展

## 🔧 性能优化

- **内存优化**: 针对8GB RAM优化，使用生成器模式
- **算法优化**: 吸引力得分算法从O(n²)优化到O(n×w)
- **向量化**: 所有计算使用NumPy向量化操作
- **对数空间**: 所有价格计算在对数空间进行，避免负值

## 📈 当前状态

- ✅ 模块1-3全部完成
- ✅ 批量处理脚本完成
- ✅ 性能优化完成
- ✅ 文件结构整理完成
- 🔄 准备进入机器学习验证阶段
