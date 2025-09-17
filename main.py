"""
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

        # 创建图像处理器
        processor = ImageProcessor()

        if choice == "1":
            # 处理所有实验的光圈图像和光谱校准图像
            for exp_name, exp_data in experiments.items():
                processor.process_experiment_data(exp_name, exp_data, output_dir)

        elif choice == "2":
            # 选择特定实验
            exp_choice = input(f"请输入实验编号 (1-{len(exp_list)}): ").strip()
            try:
                exp_idx = int(exp_choice) - 1
                if 0 <= exp_idx < len(exp_list):
                    exp_name = exp_list[exp_idx]
                    exp_data = experiments[exp_name]
                    processor.process_experiment_data(exp_name, exp_data, output_dir)
                else:
                    print("无效的实验编号")
            except ValueError:
                print("请输入有效的数字")

        elif choice == "3":
            # 仅处理光圈图像
            for exp_name, exp_data in experiments.items():
                if "aperture_min" in exp_data:
                    processor.process_aperture_images(exp_data["aperture_min"], output_dir, exp_name)
                elif "aperture" in exp_data:
                    processor.process_aperture_images(exp_data["aperture"], output_dir, exp_name)

        elif choice == "4":
            # 仅处理光谱校准图像
            for exp_name, exp_data in experiments.items():
                if "spectrum_cali" in exp_data:
                    processor.process_spectrum_images(exp_data["spectrum_cali"], output_dir, exp_name)
                else:
                    print(f"实验 {exp_name} 中未找到光谱校准图像")

        else:
            print("无效选择")

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
