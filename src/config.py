"""
Author: Galaxyrio
光场图像处理工具的配置管理
集中管理所有可配置的参数
"""

import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path


class LightFieldConfig:
    """光场处理配置类，集中管理所有可配置参数"""

    def __init__(self, pmr: float = 56.0):
        """
        初始化配置

        Args:
            pmr: 子图像直径（像素），默认40
        """
        self.pmr = pmr

        # 基于PMR的比例因子配置
        self._pmr_based_ratios = {
            'view_spacing_ratio': 0.5,                    # 多视角综合图像间距 = PMR * 这个比例
            'min_distance_ratio': 0.375,                  # 局部最大值最小距离 = PMR * 这个比例 (15/40=0.375)
            'max_grid_size_ratio': 50,                    # 最大网格尺寸 = PMR * 这个比例 (200/40=5)
            'max_output_size_ratio': 100,                 # 最大输出尺寸 = PMR * 这个比例 (4000/40=100)
            'neighbor_threshold_ratio': 0.25,             # 邻居判断阈值 = PMR * 这个比例 (10/40=0.25)
            'min_distance_threshold_ratio': 0.125,        # 最小距离阈值 = PMR * 这个比例 (5/40=0.125)
            'view_extraction_scale': 0.2,                 # 每个子图像提取区域的缩放倍率 (8/40=0.2)
            'view_offset_scale': 0.2,                    # 视角偏移的缩放倍率 (相对于子图像半径的比例)
            'title_offset_ratio': 0.125,                  # 标题偏移 = PMR * 这个比例 (5/40=0.125)
        }

        # 固定参数
        self._fixed_params = {
            'edge_mask_width': 100,                       # 边缘屏蔽宽度（像素），固定值
            'subpixel_window_size': 5,                    # 亚像素精确定位窗口大小
            'gaussian_sigma': 1.0,                        # 高斯滤波标准差
            'detection_threshold_multiplier': 2.0,        # 检测阈值倍数（相对于标准差）
            'jpeg_quality': 95,                           # JPEG保存质量
            'png_compression': 9,                         # PNG压缩级别
        }

        # 检测相关参数
        self._detection_params = {
            'expected_spacing': 56.0,                     # 期望的网格间距（像素）
            'aperture_min_distance': 50,                  # 光圈检测最小距离
            'aperture_threshold': 20,                      # 光圈检测阈值
            'spectrum_min_distance': 15,                  # 光谱检测最小距离
            'spectrum_threshold': 12,                     # 光谱检测阈值
        }

        # 可视化相关参数
        self._visualization_params = {
            'marker_size_small': 2,                       # 小标记尺寸
            'marker_size_medium': 3,                      # 中等标记尺寸
            'marker_size_large': 4,                       # 大标记尺寸
            'font_scale_small': 0.4,                      # 小字体缩放
            'font_scale_medium': 0.5,                     # 中等字体缩放
            'font_scale_large': 0.6,                      # 大字体缩放
            'line_thickness': 1,                          # 线条粗细
            'text_thickness': 2,                          # 文字粗细
        }

    def set_pmr(self, pmr: float):
        """
        设置PMR值并重新计算相关参数

        Args:
            pmr: 新的PMR值
        """
        self.pmr = pmr

    def get_pmr_based_value(self, key: str) -> float:
        """
        根据PMR值获取基于比例的参数实际值

        Args:
            key: 参数键名

        Returns:
            计算后的实际值

        Raises:
            KeyError: 如果键不存在
        """
        if key not in self._pmr_based_ratios:
            raise KeyError(f"未找到PMR相关参数: {key}")
        return self._pmr_based_ratios[key] * self.pmr

    def get_fixed_value(self, key: str) -> Any:
        """
        获取固定参数值

        Args:
            key: 参数键名

        Returns:
            参数值

        Raises:
            KeyError: 如果键不存在
        """
        if key not in self._fixed_params:
            raise KeyError(f"未找到固定参数: {key}")
        return self._fixed_params[key]

    def get_detection_value(self, key: str) -> Any:
        """
        获取检测相关参数值

        Args:
            key: 参数键名

        Returns:
            参数值

        Raises:
            KeyError: 如果键不存在
        """
        if key not in self._detection_params:
            raise KeyError(f"未找到检测参数: {key}")
        return self._detection_params[key]

    def get_visualization_value(self, key: str) -> Any:
        """
        获取可视化相关参数值

        Args:
            key: 参数键名

        Returns:
            参数值

        Raises:
            KeyError: 如果键不存在
        """
        if key not in self._visualization_params:
            raise KeyError(f"未找到可视化参数: {key}")
        return self._visualization_params[key]

    def update_pmr_ratio(self, key: str, ratio: float):
        """
        更新PMR相关的比例参数

        Args:
            key: 参数键名
            ratio: 新的比例值
        """
        if key not in self._pmr_based_ratios:
            raise KeyError(f"未找到PMR相关参数: {key}")
        self._pmr_based_ratios[key] = ratio

    def get_pmr_ratio(self, key: str) -> float:
        """
        获取PMR相关的比例参数（不乘以PMR值）

        Args:
            key: 参数键名

        Returns:
            比例参数值

        Raises:
            KeyError: 如果键不存在
        """
        if key not in self._pmr_based_ratios:
            raise KeyError(f"未找到PMR相关参数: {key}")
        return self._pmr_based_ratios[key]

    def get_all_current_values(self) -> Dict[str, Any]:
        """
        获取所有当前参数值（包括计算后的PMR相关值）

        Returns:
            包含所有参数的字典
        """
        result = {
            'pmr': self.pmr,
            'pmr_based': {},
            'fixed': self._fixed_params.copy(),
            'detection': self._detection_params.copy(),
            'visualization': self._visualization_params.copy(),
        }

        # 计算PMR相关的实际值
        for key, ratio in self._pmr_based_ratios.items():
            result['pmr_based'][key] = {
                'ratio': ratio,
                'actual_value': ratio * self.pmr
            }

        return result

    def print_config_summary(self):
        """打印配置摘要"""
        print(f"\n=== 光场处理配置摘要 ===")
        print(f"PMR值: {self.pmr} 像素")

        print(f"\n基于PMR的参数（实际值）:")
        for key, ratio in self._pmr_based_ratios.items():
            actual_value = ratio * self.pmr
            print(f"  {key}: {ratio} * {self.pmr} = {actual_value:.2f}")

        print(f"\n固定参数:")
        for key, value in self._fixed_params.items():
            print(f"  {key}: {value}")

        print(f"\n检测参数:")
        for key, value in self._detection_params.items():
            print(f"  {key}: {value}")

        print(f"\n可视化参数:")
        for key, value in self._visualization_params.items():
            print(f"  {key}: {value}")

    def load_from_yaml(self, yaml_path: Union[str, Path]) -> bool:
        """
        从YAML文件加载配置参数（只覆盖文件中存在的参数）

        Args:
            yaml_path: YAML配置文件路径

        Returns:
            加载成功返回True，失败返回False
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            print(f"警告：配置文件不存在 - {yaml_path}")
            return False

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                print(f"警告：配置文件为空 - {yaml_path}")
                return False

            print(f"从配置文件加载参数: {yaml_path}")

            # 加载PMR值
            if 'pmr' in config_data:
                old_pmr = self.pmr
                self.pmr = float(config_data['pmr'])
                print(f"  PMR: {old_pmr} -> {self.pmr}")

            # 加载基于PMR的比例参数
            if 'pmr_based_ratios' in config_data:
                for key, value in config_data['pmr_based_ratios'].items():
                    if key in self._pmr_based_ratios:
                        old_value = self._pmr_based_ratios[key]
                        self._pmr_based_ratios[key] = float(value)
                        print(f"  {key}: {old_value} -> {self._pmr_based_ratios[key]}")
                    else:
                        print(f"  警告：未知的PMR相关参数 - {key}")

            # 加载固定参数
            if 'fixed_params' in config_data:
                for key, value in config_data['fixed_params'].items():
                    if key in self._fixed_params:
                        old_value = self._fixed_params[key]
                        self._fixed_params[key] = value
                        print(f"  {key}: {old_value} -> {self._fixed_params[key]}")
                    else:
                        print(f"  警告：未知的固定参数 - {key}")

            # 加载检测参数
            if 'detection_params' in config_data:
                for key, value in config_data['detection_params'].items():
                    if key in self._detection_params:
                        old_value = self._detection_params[key]
                        self._detection_params[key] = value
                        print(f"  {key}: {old_value} -> {self._detection_params[key]}")
                    else:
                        print(f"  警告：未知的检测参数 - {key}")

            # 加载可视化参数
            if 'visualization_params' in config_data:
                for key, value in config_data['visualization_params'].items():
                    if key in self._visualization_params:
                        old_value = self._visualization_params[key]
                        self._visualization_params[key] = value
                        print(f"  {key}: {old_value} -> {self._visualization_params[key]}")
                    else:
                        print(f"  警告：未知的可视化参数 - {key}")

            print("配置加载完成！")
            return True

        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return False

    def save_to_yaml(self, yaml_path: Union[str, Path], save_all: bool = False) -> bool:
        """
        将当前配置保存到YAML文件

        Args:
            yaml_path: YAML配置文件路径
            save_all: 是否保存所有参数（True）还是只保存非默认值（False）

        Returns:
            保存成功返回True，失败返回False
        """
        yaml_path = Path(yaml_path)

        try:
            # 创建输出目录
            yaml_path.parent.mkdir(parents=True, exist_ok=True)

            if save_all:
                # 保存所有参数
                config_data = {
                    'pmr': self.pmr,
                    'pmr_based_ratios': self._pmr_based_ratios.copy(),
                    'fixed_params': self._fixed_params.copy(),
                    'detection_params': self._detection_params.copy(),
                    'visualization_params': self._visualization_params.copy(),
                }
            else:
                # 只保存与默认值不同的参数
                default_config = LightFieldConfig()  # 创建默认配置用于比较
                config_data = {}

                # 检查PMR
                if self.pmr != default_config.pmr:
                    config_data['pmr'] = self.pmr

                # 检查PMR相关参数
                pmr_diffs = {}
                for key, value in self._pmr_based_ratios.items():
                    if value != default_config._pmr_based_ratios[key]:
                        pmr_diffs[key] = value
                if pmr_diffs:
                    config_data['pmr_based_ratios'] = pmr_diffs

                # 检查固定参数
                fixed_diffs = {}
                for key, value in self._fixed_params.items():
                    if value != default_config._fixed_params[key]:
                        fixed_diffs[key] = value
                if fixed_diffs:
                    config_data['fixed_params'] = fixed_diffs

                # 检查检测参数
                detection_diffs = {}
                for key, value in self._detection_params.items():
                    if value != default_config._detection_params[key]:
                        detection_diffs[key] = value
                if detection_diffs:
                    config_data['detection_params'] = detection_diffs

                # 检查可视化参数
                viz_diffs = {}
                for key, value in self._visualization_params.items():
                    if value != default_config._visualization_params[key]:
                        viz_diffs[key] = value
                if viz_diffs:
                    config_data['visualization_params'] = viz_diffs

            # 写入YAML文件
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2)

            print(f"配置已保存到: {yaml_path}")
            return True

        except Exception as e:
            print(f"保存配置文件失败: {e}")
            return False

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path], base_pmr: float = 40.0) -> 'LightFieldConfig':
        """
        从YAML文件创建配置实例

        Args:
            yaml_path: YAML配置文件路径
            base_pmr: 基础PMR值（如果YAML文件中没有指定PMR）

        Returns:
            配置实例
        """
        config = cls(base_pmr)
        config.load_from_yaml(yaml_path)
        return config


# 创建全局默认配置实例
default_config = LightFieldConfig()


def get_config(pmr: float = None) -> LightFieldConfig:
    """
    获取配置实例

    Args:
        pmr: PMR值，如果为None则使用默认配置

    Returns:
        配置实例
    """
    if pmr is None:
        return default_config
    else:
        return LightFieldConfig(pmr)


def set_global_pmr(pmr: float):
    """
    设置全局默认PMR值

    Args:
        pmr: 新的PMR值
    """
    default_config.set_pmr(pmr)


def get_config_from_experiment(exp_dir: Union[str, Path], config_filename: str = "config.yaml") -> LightFieldConfig:
    """
    从实验目录加载配置

    Args:
        exp_dir: 实验目录路径
        config_filename: 配置文件名，默认为"config.yaml"

    Returns:
        配置实例
    """
    exp_dir = Path(exp_dir)
    config_path = exp_dir / config_filename

    if config_path.exists():
        print(f"找到实验配置文件: {config_path}")
        return LightFieldConfig.from_yaml(config_path)
    else:
        print(f"未找到实验配置文件 {config_path}，使用默认配置")
        return LightFieldConfig()


def create_experiment_config_template(exp_dir: Union[str, Path],
                                    config_filename: str = "config.yaml",
                                    pmr: float = 40.0) -> bool:
    """
    为实验目录创建配置文件模板

    Args:
        exp_dir: 实验目录路径
        config_filename: 配置文件名，默认为"config.yaml"
        pmr: PMR值

    Returns:
        创建成功返回True，失败返回False
    """
    exp_dir = Path(exp_dir)
    config_path = exp_dir / config_filename

    if config_path.exists():
        print(f"配置文件已存在: {config_path}")
        return False

    # 创建一个包含常用参数的模板配置
    config = LightFieldConfig(pmr)

    # 只保存一些关键参数作为模板
    template_data = {
        'pmr': pmr,
        'pmr_based_ratios': {
            'view_extraction_scale': 0.2,
            'view_offset_scale': 0.25,
            'view_spacing_ratio': 0.5,
        },
        'detection_params': {
            'aperture_min_distance': 15,
            'aperture_threshold': 0,
            'spectrum_min_distance': 15,
            'spectrum_threshold': 12,
        }
    }

    try:
        # 创建目录
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 写入模板配置
        with open(config_path, 'w', encoding='utf-8') as f:
            # 添加注释说明
            f.write("# 光场图像处理实验配置文件\n")
            f.write("# 只需要指定与默认值不同的参数\n")
            f.write("# 未指定的参数将使用默认值\n\n")

            yaml.dump(template_data, f, default_flow_style=False, allow_unicode=True, indent=2)

        print(f"配置模板已创建: {config_path}")
        return True

    except Exception as e:
        print(f"创建配置模板失败: {e}")
        return False
