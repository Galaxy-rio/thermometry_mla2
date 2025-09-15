# 项目目录结构说明

## 项目文件夹用途说明

### 📁 src/ (源代码目录)
存放所有Python源代码文件，包括：
- `light_field_processor.py` - 光场图像处理核心类
- `thermometry_analyzer.py` - 温度场分析工具类  
- `visualizer.py` - 可视化工具类
- `utils.py` - 通用工具函数
- `config.py` - 配置文件和参数设置
- `__init__.py` - 包初始化文件

### 📁 data/ (数据目录)
存放所有输入数据，建议按以下结构组织：
```
data/
├── raw/              # 原始光场图像
│   ├── experiment_1/
│   ├── experiment_2/
│   └── ...
├── calibration/      # 标定数据
│   ├── intensity_temp_curve.csv
│   └── microlens_params.json
└── sample/           # 示例数据
    └── test_image.png
```

### 📁 output/ (输出目录)
存放所有处理结果和生成的文件：
```
output/
├── processed_images/ # 处理后的图像
├── temperature_maps/ # 温度分布图
├── analysis_results/ # 分析结果文件
├── visualizations/   # 可视化图表
└── reports/          # 实验报告
```

### 📁 docs/ (文档目录)
存放项目文档：
- `README.md` - 项目说明
- `API_reference.md` - API文档
- `user_guide.md` - 使用指南
- `theory.md` - 理论背景
- `examples/` - 示例代码和教程

### 📁 tests/ (测试目录)
存放单元测试和集成测试：
- `test_light_field_processor.py`
- `test_thermometry_analyzer.py`
- `test_visualizer.py`
- `test_data/` - 测试用数据

### 📁 .venv/ (虚拟环境)
Python虚拟环境目录，包含项目依赖包

### 📁 .idea/ (IDE配置)
PyCharm IDE的项目配置文件

## 推荐的文件命名规范

### 数据文件命名
- 原始图像：`YYYYMMDD_HHMMSS_experiment_name.tif`
- 温度图：`YYYYMMDD_HHMMSS_temperature_map.png`
- 分析结果：`YYYYMMDD_HHMMSS_analysis_results.json`

### 输出文件命名
- 处理后图像：`processed_[原文件名]`
- 可视化图表：`[类型]_[实验名]_[时间戳].png`
- 报告文件：`report_[实验名]_[日期].pdf`
