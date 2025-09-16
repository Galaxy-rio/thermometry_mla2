"""
光场图像处理主程序
用于每次试验后处理数据
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.light_field_2_tool import ImageIO, DotArrayDetector, DotArrayVisualizer

def get_rainbow_color(value, min_val, max_val):
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

def draw_x_marker(image, x, y, color, size=3):
    """绘制X型标记"""
    x, y = int(x), int(y)
    # 绘制X的两条对角线
    cv2.line(image, (x - size, y - size), (x + size, y + size), color, 1)
    cv2.line(image, (x - size, y + size), (x + size, y - size), color, 1)

def draw_cross_marker(image, x, y, color, size=3):
    """绘制十字标记"""
    x, y = int(x), int(y)
    # 绘制十字的两条线
    cv2.line(image, (x - size, y), (x + size, y), color, 1)
    cv2.line(image, (x, y - size), (x, y + size), color, 1)

def scan_experiment_data(data_dir):
    """扫描实验数据目录结构"""
    print(f"\n扫描数据目录: {data_dir.absolute()}")

    experiments = {}

    # 扫描实验目录
    for exp_dir in data_dir.iterdir():
        if exp_dir.is_dir():
            exp_name = exp_dir.name
            experiments[exp_name] = {}
            print(f"\n实验: {exp_name}")

            # 扫描图片类型目录
            for type_dir in exp_dir.iterdir():
                if type_dir.is_dir():
                    type_name = type_dir.name

                    # 查找图片文件（使用set去重，避免Windows系统大小写不敏感导致重复）
                    image_files = set()
                    for ext in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                        # 只搜索小写扩展名，Windows会自动匹配大小写变体
                        image_files.update(type_dir.glob(f"*{ext}"))

                    # 转换为列表并排序
                    image_files = sorted(list(image_files))

                    if image_files:
                        experiments[exp_name][type_name] = image_files
                        print(f"  {type_name}: {len(image_files)} 个图片文件")
                        for img in image_files:
                            print(f"    - {img.name}")

    return experiments

def process_aperture_images(image_files, output_dir, exp_name):
    """处理光圈图像（亮点检测）"""
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
            io_tool = ImageIO()
            image = io_tool.read_image(img_file)

            if image is None:
                print(f"图像读取失败: {img_file.name}")
                continue

            # 创建检测器
            detector = DotArrayDetector(expected_spacing=16.0)

            # 执行亮点检测
            result = detector.detect_bright_spots(
                image=image,
                min_distance=15,  # 局部最大值最小距离
                threshold_abs=0   # 亮度阈值
            )

            # 保存检测到的中心点
            centers = result.get('refined_centers', [])
            all_centers.extend(centers)

            # 可视化结果
            visualizer = DotArrayVisualizer()
            output_path = exp_output_dir / f"detection_{img_file.stem}.png"
            vis_result = visualizer.visualize_detection_results(
                image=image,
                detection_result=result,
                save_path=str(output_path)
            )

            print(f"检测完成，结果保存至: {output_path}")

        except Exception as e:
            print(f"处理图像 {img_file.name} 失败: {e}")

    return all_centers

def process_spectrum_images(image_files, output_dir, exp_name):
    """处理光谱校准图像（亮点检测）"""
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
            io_tool = ImageIO()
            image = io_tool.read_image(img_file)

            if image is None:
                print(f"图像读取失败: {img_file.name}")
                continue

            # 创建检测器
            detector = DotArrayDetector(expected_spacing=16.0)

            # 执行亮点检测
            result = detector.detect_bright_spots(
                image=image,
                min_distance=15,  # 局部最大值最小距离
                threshold_abs=12  # 亮度阈值
            )

            # 保存检测到的中心点和对应的颜色
            centers = result.get('refined_centers', [])
            color = get_rainbow_color(number, min_num, max_num)

            spectrum_data[number] = {
                'centers': centers,
                'color': color,
                'filename': img_file.stem
            }

            print(f"检测到 {len(centers)} 个亮点，颜色: {color}")

            # 可视化结果
            visualizer = DotArrayVisualizer()
            output_path = exp_output_dir / f"spectrum_{img_file.stem}.png"
            vis_result = visualizer.visualize_detection_results(
                image=image,
                detection_result=result,
                save_path=str(output_path)
            )

            print(f"检测完成，结果保存至: {output_path}")

        except Exception as e:
            print(f"处理图像 {img_file.name} 失败: {e}")

    return spectrum_data

def create_combined_visualization(aperture_centers, spectrum_data, dispersion_image_path, output_dir, exp_name):
    """在dispersion图像上叠加显示所有检测到的亮点"""
    print(f"\n=== 创建综合可视化图像 ===")

    try:
        # 读取dispersion图像
        io_tool = ImageIO()
        dispersion_image = io_tool.read_image(dispersion_image_path)

        if dispersion_image is None:
            print(f"无法读取dispersion图像: {dispersion_image_path}")
            return

        print(f"使用背景图像: {dispersion_image_path}")
        print(f"光圈亮点数量: {len(aperture_centers)}")

        # 统计光谱亮点总数
        total_spectrum_spots = sum(len(data['centers']) for data in spectrum_data.values())
        print(f"光谱亮点数量: {total_spectrum_spots}")

        # 创建结果图像副本
        result_img = dispersion_image.copy()
        if len(result_img.shape) == 2:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)

        # 绘制光圈检测的亮点（绿色十字）
        for x, y in aperture_centers:
            draw_cross_marker(result_img, x, y, (0, 255, 0), size=4)  # 绿色十字，稍大一些

        # 绘制光谱校准检测的亮点（彩虹色X标记）
        legend_items = []
        for number in sorted(spectrum_data.keys()):
            data = spectrum_data[number]
            centers = data['centers']
            color = data['color']
            filename = data['filename']

            # 绘制X标记
            for x, y in centers:
                draw_x_marker(result_img, x, y, color, size=2)  # 较小的X标记

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
            draw_x_marker(result_img, 12, legend_y - 3, color, size=2)
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

def process_experiment_data(exp_name, exp_data, output_dir):
    """处理单个实验的所有数据"""
    print(f"\n=== 处理实验 {exp_name} ===")

    aperture_centers = []
    spectrum_data = {}

    # 处理光圈图像
    for type_name in ["aperture_min", "aperture"]:
        if type_name in exp_data:
            aperture_centers = process_aperture_images(exp_data[type_name], output_dir, exp_name)
            break

    # 处理光谱校准图像
    if "spectrum_cali" in exp_data:
        spectrum_data = process_spectrum_images(exp_data["spectrum_cali"], output_dir, exp_name)
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
            create_combined_visualization(aperture_centers, spectrum_data,
                                        dispersion_image, output_dir, exp_name)

def select_experiment_and_process(experiments):
    """选择实验并处理"""
    if not experiments:
        print("未找到任何实验数据")
        return

    print("\n=== 可用的实验 ===")
    exp_list = list(experiments.keys())
    for i, exp_name in enumerate(exp_list, 1):
        print(f"{i}. {exp_name}")
        for type_name, files in experiments[exp_name].items():
            print(f"   - {type_name}: {len(files)} 个文件")

    # 询问用户选择
    print("\n选择处理方式:")
    print("1. 处理所有实验（光圈图像 + 光谱校准图像）")
    print("2. 选择特定实验（光圈图像 + 光谱校准图像）")
    print("3. 仅处理光圈图像（aperture_min）")
    print("4. 仅处理光谱校准图像（spectrum_cali）")

    try:
        choice = input("请输入选择 (1-4): ").strip()

        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        if choice == "1":
            # 处理所有实验的光圈图像和光谱校准图像
            for exp_name, exp_data in experiments.items():
                process_experiment_data(exp_name, exp_data, output_dir)

        elif choice == "2":
            # 选择特定实验
            exp_choice = input(f"请输入实验编号 (1-{len(exp_list)}): ").strip()
            try:
                exp_idx = int(exp_choice) - 1
                if 0 <= exp_idx < len(exp_list):
                    exp_name = exp_list[exp_idx]
                    exp_data = experiments[exp_name]
                    process_experiment_data(exp_name, exp_data, output_dir)
                else:
                    print("无效的实验编号")
            except ValueError:
                print("请输入有效的数字")

        elif choice == "3":
            # 仅处理光圈图像
            for exp_name, exp_data in experiments.items():
                for type_name in ["aperture_min", "aperture"]:
                    if type_name in exp_data:
                        process_aperture_images(exp_data[type_name], output_dir, exp_name)
                        break

        elif choice == "4":
            # 仅处理光谱校准图像
            for exp_name, exp_data in experiments.items():
                if "spectrum_cali" in exp_data:
                    process_spectrum_images(exp_data["spectrum_cali"], output_dir, exp_name)
                else:
                    print(f"实验 {exp_name} 中未找到光谱校准图像")

        else:
            print("无效选择")

    except KeyboardInterrupt:
        print("\n用户取消操作")

def main():
    """主函数"""
    print("=== 光场图像点阵检测系统 ===")

    # 检查数据目录
    data_dir = Path("data")
    if not data_dir.exists():
        print("数据目录不存在，正在创建...")
        data_dir.mkdir()
        print("请将实验数据放入data目录中")
        print("目录结构应为: data/实验名/图片类型/图片文件")
        return

    # 扫描实验数据
    experiments = scan_experiment_data(data_dir)

    if not experiments:
        print("\n未找到实验数据")
        print("请确保目录结构为: data/实验名/图片类型/图片文件")
        print("例如: data/mla2_test_250909/aperture_min/min.bmp")
        return

    # 选择并处理实验
    select_experiment_and_process(experiments)

    print("\n=== 程序结束 ===")
    print("检查output目录查看处理结果")


if __name__ == "__main__":
    main()
