"""
简单的光场图像处理工具
提供基本的图片读取和保存功能
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Optional, Union
from scipy.ndimage import gaussian_filter, maximum_filter

class ImageIO:
    """图像输入输出工具类"""

    def __init__(self):
        """初始化图像IO工具"""
        self.current_image = None
        self.image_path = None

    def read_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        读取图像文件

        Args:
            image_path: 图像文件路径

        Returns:
            图像数组，如果读取失败则返回None
        """
        try:
            # 转换为字符串路径
            image_path = str(image_path)

            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"错误：文件不存在 - {image_path}")
                return None

            # 读取图像
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if image is None:
                print(f"错误：无法读取图像文件 - {image_path}")
                return None

            # 保存当前图像信息
            self.current_image = image
            self.image_path = image_path

            print(f"成功读取图像：{image_path}")
            print(f"图像尺寸：{image.shape}")
            print(f"图像类型：{image.dtype}")

            return image

        except Exception as e:
            print(f"读取图像时发生错误：{e}")
            return None

    def save_image(self, image: np.ndarray, output_path: Union[str, Path],
                   quality: int = 95) -> bool:
        """
        保存图像文件

        Args:
            image: 要保存的图像数组
            output_path: 输出文件路径
            quality: JPEG质量(0-100)，仅对JPEG格式有效

        Returns:
            保存成功返回True，失败返回False
        """
        try:
            # 转换为字符串路径
            output_path = str(output_path)

            # 创建输出目录（如果不存在）
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"创建目录：{output_dir}")

            # 根据文件扩展名设置保存参数
            file_ext = os.path.splitext(output_path)[1].lower()

            if file_ext in ['.jpg', '.jpeg']:
                # JPEG格式，设置质量
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                success = cv2.imwrite(output_path, image, encode_params)
            elif file_ext == '.png':
                # PNG格式，设置压缩级别
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
                success = cv2.imwrite(output_path, image, encode_params)
            else:
                # 其他格式，使用默认参数
                success = cv2.imwrite(output_path, image)

            if success:
                print(f"图像已成功保存到：{output_path}")
                return True
            else:
                print(f"保存图像失败：{output_path}")
                return False

        except Exception as e:
            print(f"保存图像时发生错误：{e}")
            return False

    def get_image_info(self, image: Optional[np.ndarray] = None) -> dict:
        """
        获取图像信息

        Args:
            image: 图像数组，如果为None则使用当前图像

        Returns:
            包含图像信息的字典
        """
        if image is None:
            image = self.current_image

        if image is None:
            return {"error": "没有可用的图像"}

        info = {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "size": image.size,
            "channels": len(image.shape),
            "height": image.shape[0],
            "width": image.shape[1]
        }

        if len(image.shape) == 3:
            info["channels"] = image.shape[2]
        else:
            info["channels"] = 1

        return info

    def copy_image(self, image: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        复制图像

        Args:
            image: 要复制的图，如果为None则复制当前图像

        Returns:
            图像副本
        """
        if image is None:
            image = self.current_image

        if image is None:
            print("错误：没有可复制的图像")
            return None

        return image.copy()


class DotArrayDetector:
    """点阵检测器 - 简化版本，只进行基本的点检测"""

    def __init__(self, expected_spacing: float = 16.0):
        """
        初始化点阵检测器

        Args:
            expected_spacing: 期望的网格间距（像素）
        """
        self.expected_spacing = expected_spacing
        self.detected_peaks = []
        self.refined_centers = []

    def detect_local_maxima(self, image: np.ndarray, min_distance: int = 8,
                            threshold_abs: Optional[float] = None,
                            edge_mask_width: int = 300) -> list:
        """
        检测局部最亮点

        Args:
            image: 输入图像（灰度图）
            min_distance: 局部最大值之间的最小距离
            threshold_abs: 绝对阈值，低于此值的点不被考虑
            edge_mask_width: 边缘屏蔽宽度（像素），默认300像素

        Returns:
            局部最大值点的坐标列表 [(x, y), ...]
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 创建边缘屏蔽掩码
        h, w = gray.shape
        mask = np.ones((h, w), dtype=bool)

        # 屏蔽边缘区域（上下左右各edge_mask_width像素）
        mask[:edge_mask_width, :] = False  # 上边缘
        mask[-edge_mask_width:, :] = False  # 下边缘
        mask[:, :edge_mask_width] = False  # 左边缘
        mask[:, -edge_mask_width:] = False  # 右边缘

        print(f"屏蔽边缘区域：{edge_mask_width}像素宽度")
        print(f"有效检测区域：{np.sum(mask)} / {h * w} 像素")

        # 高斯滤波平滑图像
        smoothed = gaussian_filter(gray.astype(np.float32), sigma=1.0)

        # 使用最大值滤波器找局部最大值
        local_maxima = maximum_filter(smoothed, size=min_distance) == smoothed

        # 应用边缘屏蔽
        local_maxima = local_maxima & mask

        # 设置阈值
        if threshold_abs is None:
            threshold_abs = np.mean(smoothed[mask]) + 2 * np.std(smoothed[mask])  # 只在有效区域计算阈值

        # 应用阈值
        local_maxima = local_maxima & (smoothed > threshold_abs)

        # 获取局部最大值的坐标
        y_coords, x_coords = np.where(local_maxima)
        peaks = list(zip(x_coords, y_coords))

        self.detected_peaks = peaks
        print(f"检测到 {len(peaks)} 个局部最亮点（已排除边缘区域）")

        return peaks

    def refine_centers_subpixel(self, image: np.ndarray, peaks: list,
                                window_size: int = 5) -> list:
        """
        使用二次函数插值精确定位亚像素中心

        Args:
            image: 输入图像
            peaks: 粗略的峰值位置列表
            window_size: 拟合窗口的半径

        Returns:
            精确的中心位置列表 [(x, y), ...]
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        refined_centers = []

        for x, y in peaks:
            try:
                # 定义拟合窗口
                x_min = max(0, x - window_size)
                x_max = min(gray.shape[1], x + window_size + 1)
                y_min = max(0, y - window_size)
                y_max = min(gray.shape[0], y + window_size + 1)

                # 提取窗口区域
                window = gray[y_min:y_max, x_min:x_max].astype(np.float32)

                if window.size < 9:  # 窗口太小，跳过
                    refined_centers.append((float(x), float(y)))
                    continue

                # 创建坐标网格
                h, w = window.shape
                xx, yy = np.meshgrid(np.arange(w), np.arange(h))

                # 将坐标和强度值展平
                coords = np.column_stack([xx.ravel(), yy.ravel()])
                intensities = window.ravel()

                # 构建二次函数的系数矩阵 [1, x, y, x^2, xy, y^2]
                intensity_quadratic_func = np.column_stack([
                    np.ones(len(coords)),
                    coords[:, 0],  # x
                    coords[:, 1],  # y
                    coords[:, 0] ** 2,  # x^2
                    coords[:, 0] * coords[:, 1],  # xy
                    coords[:, 1] ** 2  # y^2
                ])

                # 最小二乘拟合
                coeffs, residuals, rank, s = np.linalg.lstsq(intensity_quadratic_func, intensities, rcond=None)

                # 计算二次函数的最大值点（亚像素中心）
                # f(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*x*y + a5*y^2
                # 求导并令其为0: df/dx = a1 + 2*a3*x + a4*y = 0
                #                df/dy = a2 + a4*x + 2*a5*y = 0

                a1, a2, a3, a4, a5 = coeffs[1:6]

                # 解线性方程组
                det = 4 * a3 * a5 - a4 * a4
                if abs(det) > 1e-10:
                    center_x = -(2 * a5 * a1 - a4 * a2) / det
                    center_y = -(2 * a3 * a2 - a4 * a1) / det

                    # 转换回全局坐标
                    global_x = center_x + x_min
                    global_y = center_y + y_min

                    # 检查结果是否合理（应该在原始点附近）
                    if (abs(global_x - x) < window_size and
                            abs(global_y - y) < window_size):
                        refined_centers.append((global_x, global_y))
                    else:
                        refined_centers.append((float(x), float(y)))
                else:
                    # 二次函数退化，使用原始坐标
                    refined_centers.append((float(x), float(y)))

            except Exception as e:
                # 出错时使用原始坐标
                refined_centers.append((float(x), float(y)))

        self.refined_centers = refined_centers
        print(f"精确定位了 {len(refined_centers)} 个中心点")

        return refined_centers

    def detect_bright_spots(self, image: np.ndarray,
                         min_distance: Optional[int] = None,
                         threshold_abs: Optional[float] = None) -> dict:
        """
        完整的亮点检测流程

        Args:
            image: 输入图像
            min_distance: 局部最大最小距离
            threshold_abs: 检测阈值

        Returns:
            检测结果字典
        """
        if min_distance is None:
            min_distance = max(4, int(self.expected_spacing / 3))

        print("开始亮点检测...")

        # 步骤1: 检测局部最亮点
        peaks = self.detect_local_maxima(image, min_distance, threshold_abs)

        if len(peaks) < 1:
            print("警告：未检测到任何亮点")
            return {
                'detected_peaks': [],
                'refined_centers': [],
                'num_detected': 0
            }

        # 步骤2: 亚像素精确定位
        refined_centers = self.refine_centers_subpixel(image, peaks)

        result = {
            'detected_peaks': peaks,
            'refined_centers': refined_centers,
            'num_detected': len(refined_centers)
        }

        print(f"亮点检测完成! 检测到 {len(refined_centers)} 个亮点")

        return result


class DotArrayVisualizer:
    """点检测结果可视化器"""

    def __init__(self):
        """初始化可视化器"""
        pass

    def visualize_detection_results(self, image: np.ndarray, detection_result: dict,
                                    save_path: Optional[str] = None) -> np.ndarray:
        """
        可视化点检测结果（原有的OpenCV方法，保持向后兼容）

        Args:
            image: 原始图像
            detection_result: 检测结果字典
            save_path: 保存路径（可选）

        Returns:
            可视化结果图像
        """
        # 创建结果图像副本
        result_img = image.copy()
        if len(result_img.shape) == 2:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)

        detected_peaks = detection_result.get('detected_peaks', [])
        refined_centers = detection_result.get('refined_centers', [])

        # 绘制检测到的粗略峰值点（红色圆圈）
        for x, y in detected_peaks:
            cv2.circle(result_img, (int(x), int(y)), 2, (0, 0, 255), 1)

        # 绘制精确定位的中心点（绿色十字）
        for x, y in refined_centers:
            x, y = int(x), int(y)
            cv2.line(result_img, (x - 3, y), (x + 3, y), (0, 255, 0), 1)
            cv2.line(result_img, (x, y - 3), (x, y + 3), (0, 255, 0), 1)

        # 添加图例
        legend_y = 30
        cv2.putText(result_img, "Red: Detected peaks", (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(result_img, "Green: Refined centers", (10, legend_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # 添加检测到的点数信息
        num_detected = detection_result.get('num_detected', 0)
        info_text = f"Detected points: {num_detected}"
        cv2.putText(result_img, info_text, (10, result_img.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if save_path:
            cv2.imwrite(save_path, result_img)
            print(f"可视化结果已保存到: {save_path}")

        return result_img


if __name__ == "__main__":
    # 示例用法
    print("=== 点检测工具测试 ===")

    # 这里可以添加测试代码
    print("请使用 detect_dot_array() 函数或创建 DotArrayDetector 实例进行检测")
