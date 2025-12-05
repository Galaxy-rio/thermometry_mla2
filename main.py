"""
Author: Galaxyrio
光场图像处理主程序
用于每次试验后处理数据
"""

import os
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.light_field_2_tool import ImageProcessor

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

def select_experiment_and_process(experiments):
    """选择实验并处理 - 两步选择流程"""
    if not experiments:
        print("未找到任何实验数据")
        return

    # 第一步：选择实验
    print("\n=== 第一步：选择实验 ===")
    exp_list = list(experiments.keys())
    print("0. 全部实验")
    for i, exp_name in enumerate(exp_list, 1):
        print(f"{i}. {exp_name}")
        for type_name, files in experiments[exp_name].items():
            print(f"   - {type_name}: {len(files)} 个文件")

    try:
        exp_choice = input(f"\n请选择实验 (0-{len(exp_list)}): ").strip()
        exp_idx = int(exp_choice)

        # 确定要处理的实验列表
        selected_experiments = {}
        if exp_idx == 0:
            # 选择全部实验
            selected_experiments = experiments
            print("\n已选择：全部实验")
        elif 1 <= exp_idx <= len(exp_list):
            # 选择特定实验
            exp_name = exp_list[exp_idx - 1]
            selected_experiments[exp_name] = experiments[exp_name]
            print(f"\n已选择实验：{exp_name}")
        else:
            print("无效的实验编号")
            return

        # 第二步：选择操作类型
        print("\n=== 第二步：选择操作类型 ===")
        print("0. 全部操作")
        print("1. 寻找最小光圈中心")
        print("2. 寻找最小光圈中心 + 各个波段图像的中心")
        print("3. 生成第一类多视角图像")
        print("4. 按温度生成第一类多视角图像")
        print("5. 未来功能（暂未实现）")

        operation_choice = input("\n请选择操作类型 (0-4): ").strip()
        operation_idx = int(operation_choice)

        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        data_dir = Path("data")  # 数据根目录

        # 根据操作类型执行相应处理
        if operation_idx == 0:
            # 全部操作
            print("\n开始执行：全部操作")
            for exp_name, exp_data in selected_experiments.items():
                print(f"\n处理实验：{exp_name}")
                
                # 为每个实验创建独立的处理器，自动加载对应配置
                exp_dir = data_dir / exp_name
                processor = ImageProcessor(exp_dir=exp_dir)
                
                processor.process_experiment_data(exp_name, exp_data, output_dir)
                print(f"实验 {exp_name} 处理完成，可视化结果已保存到 output/{exp_name}/")

        elif operation_idx == 1:
            # 仅寻找最小光圈中心
            print("\n开始执行：仅寻找最小光圈中心")
            for exp_name, exp_data in selected_experiments.items():
                print(f"\n处理实验：{exp_name}")
                
                # 为每个实验创建独立的处理器，自动加载对应配置
                exp_dir = data_dir / exp_name
                processor = ImageProcessor(exp_dir=exp_dir)
                
                if "aperture_min" in exp_data:
                    processor.process_aperture_images(exp_data["aperture_min"], output_dir, exp_name)
                    print(f"实验 {exp_name} 光圈中心检测完成，可视化结果已保存到 output/{exp_name}/aperture_detection/")
                elif "aperture" in exp_data:
                    processor.process_aperture_images(exp_data["aperture"], output_dir, exp_name)
                    print(f"实验 {exp_name} 光圈中心检测完成，可视化结果已保存到 output/{exp_name}/aperture_detection/")
                else:
                    print(f"实验 {exp_name} 中未找到光圈图像")

        elif operation_idx == 2:
            # 寻找最小光圈中心 + 各个波段图像的中心
            print("\n开始执行：光圈中心检测 + 光谱中心检测")
            for exp_name, exp_data in selected_experiments.items():
                print(f"\n处理实验：{exp_name}")
                
                # 为每个实验创建独立的处理器，自动加载对应配置
                exp_dir = data_dir / exp_name
                processor = ImageProcessor(exp_dir=exp_dir)

                aperture_centers = []
                spectrum_data = {}

                # 处理光圈图像
                if "aperture_min" in exp_data:
                    aperture_centers = processor.process_aperture_images(exp_data["aperture_min"], output_dir, exp_name)
                    print(f"  - 光圈中心检测完成")
                elif "aperture" in exp_data:
                    aperture_centers = processor.process_aperture_images(exp_data["aperture"], output_dir, exp_name)
                    print(f"  - 光圈中心检测完成")
                else:
                    print(f"  - 实验 {exp_name} 中未找到光圈图像")

                # 处理光谱校准图像
                if "spectrum_cali" in exp_data:
                    spectrum_data = processor.process_spectrum_images(exp_data["spectrum_cali"], output_dir, exp_name)
                    print(f"  - 光谱中心检测完成")
                else:
                    print(f"  - 实验 {exp_name} 中未找到光谱校准图像")

                # 如果有lf_raw图像，创建综合可视化（将所有点画在光场原图上）
                if "lf_raw" in exp_data and (aperture_centers or spectrum_data):
                    # 默认选择lf_raw0.bmp，如果不存在则选择第一个
                    lf_raw_files = exp_data["lf_raw"]
                    lf_raw_image = None

                    # 寻找lf_raw0.bmp
                    for img_file in lf_raw_files:
                        if img_file.stem == "lf_raw0":
                            lf_raw_image = img_file
                            break

                    # 如果没找到lf_raw0，使用第一个文件
                    if lf_raw_image is None and lf_raw_files:
                        lf_raw_image = lf_raw_files[0]
                        print(f"    未找到lf_raw0.bmp，使用 {lf_raw_image.name}")

                    if lf_raw_image:
                        processor.create_combined_visualization(aperture_centers, spectrum_data,
                                                            lf_raw_image, output_dir, exp_name)
                        print(f"  - 综合可视化完成，所有检测点已画在{lf_raw_image.name}上")
                elif "lf_raw" not in exp_data:
                    print(f"  - 实验 {exp_name} 中未找到光场原图，跳过综合可视化")

                print(f"实验 {exp_name} 处理完成，可视化结果已保存到 output\\{exp_name}\\")

        elif operation_idx == 3:
            # 生成第一类多视角图像
            print("\n开始执行：生成第一类多视角图像")

            # 首先需要检测光圈中心
            for exp_name, exp_data in selected_experiments.items():
                print(f"\n处理实验：{exp_name}")

                # 为每个实验创建独立的处理器，自动加载对应配置
                exp_dir = data_dir / exp_name
                processor = ImageProcessor(exp_dir=exp_dir)

                # 检测光圈中心点
                aperture_centers = []
                if "aperture_min" in exp_data:
                    aperture_centers = processor.process_aperture_images(exp_data["aperture_min"], output_dir, exp_name)
                    print(f"  - 光圈中心检测完成，共检测到 {len(aperture_centers)} 个中心点")
                elif "aperture" in exp_data:
                    aperture_centers = processor.process_aperture_images(exp_data["aperture"], output_dir, exp_name)
                    print(f"  - 光圈中心检测完成，共检测到 {len(aperture_centers)} 个中心点")
                else:
                    print(f"  - 实验 {exp_name} 中未找到光圈图像，无法生成多视角图像")
                    continue

                if not aperture_centers:
                    print(f"  - 未检测到光圈中心，跳过多视角图像生成")
                    continue

                # 检查是否有光场原图
                if "lf_raw" not in exp_data:
                    print(f"  - 实验 {exp_name} 中未找到光场原图，无法生成多视角图像")
                    continue

                # 选择光场原图
                lf_raw_files = exp_data["lf_raw"]
                lf_raw_image = None

                # 寻找lf_raw0.bmp
                for img_file in lf_raw_files:
                    if img_file.stem == "lf_raw0":
                        lf_raw_image = img_file
                        break

                # 如果没找到lf_raw0，使用第一个文件
                if lf_raw_image is None and lf_raw_files:
                    lf_raw_image = lf_raw_files[0]
                    print(f"    未找到lf_raw0.bmp，使用 {lf_raw_image.name}")

                if lf_raw_image:
                    # 从配置中获取PMR值
                    pmr = processor.config.pmr
                    print(f"  - 使用PMR值: {pmr}")

                    # 生成多视角图像
                    processor.generate_multi_view_images(
                        lf_raw_image, aperture_centers, output_dir, exp_name, pmr=pmr
                    )
                    print(f"  - 多视角图像生成完成")

                print(f"实验 {exp_name} 多视角图像处理完成，结果已保存到 output\\{exp_name}\\multi_view\\")


        elif operation_idx == 4:
            # 批量生成多视角图像（处理所有lf_raw图像）
            print("\n开始执行：批量生成多视角图像（处理所有lf_raw图像）")

            for exp_name, exp_data in selected_experiments.items():
                print(f"\n处理实验：{exp_name}")

                # 为每个实验创建独立的处理器，自动加载对应配置
                exp_dir = data_dir / exp_name
                processor = ImageProcessor(exp_dir=exp_dir)

                # 检测光圈中心点
                aperture_centers = []
                if "aperture_min" in exp_data:
                    aperture_centers = processor.process_aperture_images(exp_data["aperture_min"], output_dir, exp_name)
                    print(f"  - 光圈中心检测完成，共检测到 {len(aperture_centers)} 个中心点")
                elif "aperture" in exp_data:
                    aperture_centers = processor.process_aperture_images(exp_data["aperture"], output_dir, exp_name)
                    print(f"  - 光圈中心检测完成，共检测到 {len(aperture_centers)} 个中心点")
                else:
                    print(f"  - 实验 {exp_name} 中未找到光圈图像，无法生成多视角图像")
                    continue

                if not aperture_centers:
                    print(f"  - 未检测到光圈中心，跳过多视角图像生成")
                    continue

                # 检查是否有光场原图
                if "lf_raw" not in exp_data:
                    print(f"  - 实验 {exp_name} 中未找到光场原图，无法生成多视角图像")
                    continue

                # 获取所有lf_raw图像
                lf_raw_files = exp_data["lf_raw"]
                print(f"  - 找到 {len(lf_raw_files)} 个光场原图，准备批量处理")

                # 从配置中获取PMR值
                pmr = processor.config.pmr
                print(f"  - 使用PMR值: {pmr}")

                # 逐个处理每张光场原图
                for idx, lf_raw_image in enumerate(lf_raw_files, 1):
                    print(f"  - 处理第 {idx}/{len(lf_raw_files)} 张图像: {lf_raw_image.name}")

                    # 获取图像文件名（不含扩展名）作为子文件夹名称
                    image_name = lf_raw_image.stem

                    # 生成多视角图像，指定子文件夹名称
                    processor.generate_multi_view_images(
                        lf_raw_image, aperture_centers, output_dir, exp_name,
                        pmr=pmr, subfolder=image_name
                    )
                    print(f"    √ {image_name} 的多视角图像生成完成")

                print(f"\n实验 {exp_name} 批量多视角图像处理完成")
                print(f"结果已保存到 output\\{exp_name}\\multi_view\\<图像名称>\\")



        elif operation_idx >= 5:
            # 未来功能
            print(f"\n操作类型 {operation_idx} 为未来功能，暂未实现")
            print("请选择其他操作类型")

        else:
            print("无效的操作类型")

    except ValueError:
        print("请输入有效的数字")
    except KeyboardInterrupt:
        print("\n用户取消操作")

def main():
    """主函数"""
    print("=== 光场图像中心点检测系统 ===")

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

if __name__ == "__main__":
    main()
