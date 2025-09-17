"""
简单的光场图像处理工具
提供基本的图片读取和保存功能
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
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


class ImageProcessor:
    """图像处理器类，包含所有图像处理和分析功能"""

    def __init__(self):
        """初始化图像处理器"""
        self.io_tool = ImageIO()
        self.detector = None
        self.visualizer = DotArrayVisualizer()

    @staticmethod
    def get_rainbow_color(value: float, min_val: float, max_val: float) -> Tuple[int, int, int]:
        """
        根据数值生成彩虹色彩映射
        紫色(最小值) -> 蓝色 -> 青色 -> 绿色 -> 黄色 -> 红色(最大值)

        Args:
            value: 当前数值
            min_val: 最小数值
            max_val: 最大数值

        Returns:
            BGR颜色元组
        """
        if max_val == min_val:
            return (128, 0, 128)  # 紫色

        # 将数值归一化到0-1
        normalized = (value - min_val) / (max_val - min_val)

        # 彩虹色谱：紫色->蓝色->青色->绿色->黄色->红色
        if normalized <= 0.2:  # 紫色到蓝色
            ratio = normalized / 0.2
            b = int(255 * (1 - ratio * 0.5))  # 255->128
            g = 0
            r = int(128 * (1 - ratio))  # 128->0
        elif normalized <= 0.4:  # 蓝色到青色
            ratio = (normalized - 0.2) / 0.2
            b = 255
            g = int(255 * ratio)  # 0->255
            r = 0
        elif normalized <= 0.6:  # 青色到绿色
            ratio = (normalized - 0.4) / 0.2
            b = int(255 * (1 - ratio))  # 255->0
            g = 255
            r = 0
        elif normalized <= 0.8:  # 绿色到黄色
            ratio = (normalized - 0.6) / 0.2
            b = 0
            g = 255
            r = int(255 * ratio)  # 0->255
        else:  # 黄色到红色
            ratio = (normalized - 0.8) / 0.2
            b = 0
            g = int(255 * (1 - ratio))  # 255->0
            r = 255

        return (b, g, r)  # OpenCV使用BGR格式

    @staticmethod
    def draw_x_marker(image: np.ndarray, x: float, y: float, color: Tuple[int, int, int], size: int = 3):
        """绘制X型标记"""
        x, y = int(x), int(y)
        # 绘制X的两条对角线
        cv2.line(image, (x - size, y - size), (x + size, y + size), color, 1)
        cv2.line(image, (x - size, y + size), (x + size, y - size), color, 1)

    @staticmethod
    def draw_cross_marker(image: np.ndarray, x: float, y: float, color: Tuple[int, int, int], size: int = 3):
        """绘制十字标记"""
        x, y = int(x), int(y)
        # 绘制十字的两条线
        cv2.line(image, (x - size, y), (x + size, y), color, 1)
        cv2.line(image, (x, y - size), (x, y + size), color, 1)

    def process_aperture_images(self, image_files: List[Path], output_dir: Path, exp_name: str) -> List[Tuple[float, float]]:
        """处理光圈图像（中心点检测）"""
        print(f"\n=== 处理实验 {exp_name} 的光圈图像 ===")

        if not image_files:
            print("未找到光圈图像文件")
            return []

        # 创建实验专用输出目录
        exp_output_dir = output_dir / exp_name / "aperture_detection"
        exp_output_dir.mkdir(parents=True, exist_ok=True)

        all_centers = []  # 存储所有检测到的中心点

        # 处理每个图像
        for img_file in image_files:
            print(f"\n处理图像: {img_file.name}")

            try:
                # 读取图像
                image = self.io_tool.read_image(img_file)

                if image is None:
                    print(f"图像读取失败: {img_file.name}")
                    continue

                # 创建检测器
                self.detector = DotArrayDetector(expected_spacing=16.0)

                # 执行中心点检测
                result = self.detector.detect_bright_spots(
                    image=image,
                    min_distance=15,  # 局部最大值最小距离
                    threshold_abs=0   # 亮度阈值
                )

                # 保存检测到的中心点
                centers = result.get('refined_centers', [])
                all_centers.extend(centers)

                # 可视化结果
                output_path = exp_output_dir / f"detection_{img_file.stem}.png"
                vis_result = self.visualizer.visualize_detection_results(
                    image=image,
                    detection_result=result,
                    save_path=str(output_path)
                )

                print(f"检测完成，结果保存至: {output_path}")

            except Exception as e:
                print(f"处理图像 {img_file.name} 失败: {e}")

        return all_centers

    def process_spectrum_images(self, image_files: List[Path], output_dir: Path, exp_name: str) -> Dict:
        """处理光谱校准图像（中心点检测）"""
        print(f"\n=== 处理实验 {exp_name} 的光谱校准图像 ===")

        if not image_files:
            print("未找到光谱校准图像文件")
            return {}

        # 创建实验专用输出目录
        exp_output_dir = output_dir / exp_name / "spectrum_detection"
        exp_output_dir.mkdir(parents=True, exist_ok=True)

        spectrum_data = {}  # 存储按文件名数字分组的中心点数据

        # 提取文件名中的数字并排序
        file_numbers = []
        file_mapping = {}

        for img_file in image_files:
            try:
                # 提取文件名中的数字
                number = int(''.join(filter(str.isdigit, img_file.stem)))
                file_numbers.append(number)
                file_mapping[number] = img_file
            except ValueError:
                print(f"警告：无法从文件名 {img_file.stem} 中提取数字，跳过")
                continue

        if not file_numbers:
            print("未找到包含数字的光谱文件")
            return {}

        file_numbers.sort()
        min_num, max_num = min(file_numbers), max(file_numbers)
        print(f"光谱文件数字范围: {min_num} - {max_num}")

        # 处理每个图像
        for number in file_numbers:
            img_file = file_mapping[number]
            print(f"\n处理图像: {img_file.name} (数值: {number})")

            try:
                # 读取图像
                image = self.io_tool.read_image(img_file)

                if image is None:
                    print(f"图像读取失败: {img_file.name}")
                    continue

                # 创建检测器
                self.detector = DotArrayDetector(expected_spacing=16.0)

                # 执行中心点检测
                result = self.detector.detect_bright_spots(
                    image=image,
                    min_distance=15,  # 局部最大值最小距离
                    threshold_abs=12  # 亮度阈值
                )

                # 保存检测到的中心点和对应的颜色
                centers = result.get('refined_centers', [])
                color = self.get_rainbow_color(number, min_num, max_num)

                spectrum_data[number] = {
                    'centers': centers,
                    'color': color,
                    'filename': img_file.stem
                }

                print(f"检测到 {len(centers)} 个中心点，颜色: {color}")

                # 可视化结果
                output_path = exp_output_dir / f"spectrum_{img_file.stem}.png"
                vis_result = self.visualizer.visualize_detection_results(
                    image=image,
                    detection_result=result,
                    save_path=str(output_path)
                )

                print(f"检测完成，结果保存至: {output_path}")

            except Exception as e:
                print(f"处理图像 {img_file.name} 失败: {e}")

        return spectrum_data

    def create_combined_visualization(self, aperture_centers: List[Tuple[float, float]],
                                    spectrum_data: Dict, dispersion_image_path: Path,
                                    output_dir: Path, exp_name: str):
        """在dispersion图像上叠加显示所有检测到的中心点"""
        print(f"\n=== 创建综合可视化图像 ===")

        try:
            # 读取dispersion图像
            dispersion_image = self.io_tool.read_image(dispersion_image_path)

            if dispersion_image is None:
                print(f"无法读取dispersion图像: {dispersion_image_path}")
                return

            print(f"使用背景图像: {dispersion_image_path}")
            print(f"光圈中心点数量: {len(aperture_centers)}")

            # 统计光谱中心点总数
            total_spectrum_spots = sum(len(data['centers']) for data in spectrum_data.values())
            print(f"光谱中心点数量: {total_spectrum_spots}")

            # 创建结果图像副本
            result_img = dispersion_image.copy()
            if len(result_img.shape) == 2:
                result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)

            # 绘制光圈检测的中心点（绿色十字）
            for x, y in aperture_centers:
                self.draw_cross_marker(result_img, x, y, (0, 255, 0), size=4)  # 绿色十字，稍大一些

            # 绘制光谱校准检测的中心点（彩虹色X标记）
            legend_items = []
            for number in sorted(spectrum_data.keys()):
                data = spectrum_data[number]
                centers = data['centers']
                color = data['color']
                filename = data['filename']

                # 绘制X标记
                for x, y in centers:
                    self.draw_x_marker(result_img, x, y, color, size=2)  # 较小的X标记

                legend_items.append((filename, color, len(centers)))

            # 添加图例
            legend_y = 30
            cv2.putText(result_img, "Green crosses: Aperture (aperture_min)", (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            legend_y += 20
            cv2.putText(result_img, "Rainbow X marks: Spectrum (spectrum_cali)", (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 添加光谱文件的详细图例
            legend_y += 25
            for filename, color, count in legend_items:
                legend_text = f"{filename}: {count} spots"
                cv2.putText(result_img, legend_text, (20, legend_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                # 在文字前绘制一个小X作为示例
                self.draw_x_marker(result_img, 12, legend_y - 3, color, size=2)
                legend_y += 15

            # 添加统计信息
            info_text_1 = f"Aperture spots: {len(aperture_centers)}"
            info_text_2 = f"Spectrum spots: {total_spectrum_spots}"
            info_text_3 = f"Total spots: {len(aperture_centers) + total_spectrum_spots}"

            cv2.putText(result_img, info_text_1, (10, result_img.shape[0] - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_img, info_text_2, (10, result_img.shape[0] - 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_img, info_text_3, (10, result_img.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 保存综合可视化结果
            combined_output_dir = output_dir / exp_name / "combined_visualization"
            combined_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = combined_output_dir / "combined_bright_spots.png"

            cv2.imwrite(str(output_path), result_img)
            print(f"综合可视化图像已保存到: {output_path}")

        except Exception as e:
            print(f"创建综合可视化失败: {e}")

    def process_experiment_data(self, exp_name: str, exp_data: Dict, output_dir: Path):
        """处理单个实验的所有数据"""
        print(f"\n=== 处理实验 {exp_name} ===")

        aperture_centers = []
        spectrum_data = {}

        # 处理光圈图像 - 只选择aperture_min，避免Windows大小写不敏感导致重复
        if "aperture_min" in exp_data:
            aperture_centers = self.process_aperture_images(exp_data["aperture_min"], output_dir, exp_name)
        elif "aperture" in exp_data:
            aperture_centers = self.process_aperture_images(exp_data["aperture"], output_dir, exp_name)

        # 处理光谱校准图像
        if "spectrum_cali" in exp_data:
            spectrum_data = self.process_spectrum_images(exp_data["spectrum_cali"], output_dir, exp_name)
        else:
            print(f"实验 {exp_name} 中未找到光谱校准图像")

        # 如果有dispersion图像，创建综合可视化
        if "dispersion" in exp_data and (aperture_centers or spectrum_data):
            # 默认选择dispersion0.bmp，如果不存在则选择第一个
            dispersion_files = exp_data["dispersion"]
            dispersion_image = None

            # 寻找dispersion0.bmp
            for img_file in dispersion_files:
                if img_file.stem == "dispersion0":
                    dispersion_image = img_file
                    break

            # 如果没找到dispersion0，使用第一个文件
            if dispersion_image is None and dispersion_files:
                dispersion_image = dispersion_files[0]
                print(f"未找到dispersion0.bmp，使用 {dispersion_image.name}")

            if dispersion_image:
                self.create_combined_visualization(aperture_centers, spectrum_data,
                                                dispersion_image, output_dir, exp_name)
