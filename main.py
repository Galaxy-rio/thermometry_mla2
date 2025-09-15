"""
光场图像处理主程序
用于每次试验后处理数据
"""

import os
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.light_field_2_tool import ImageIO, DotArrayDetector, DotArrayVisualizer

def test_custom_detection():
    """自定义参数的点阵检测测试"""
    print("\n=== 自定义参数检测测试 ===")

    data_dir = Path("data")
    output_dir = Path("output")

    # 查找图片
    image_files = list(data_dir.glob("*.bmp")) + list(data_dir.glob("*.BMP"))

    if not image_files:
        print("未找到BMP格式的测试图片")
        return

    test_image = image_files[0]
    print(f"测试图片: {test_image}")

    try:
        # 读取图像
        io_tool = ImageIO()
        image = io_tool.read_image(test_image)

        if image is None:
            print("图像读取失败")
            return

        # 创建检测器 - 你可以调整这些参数
        detector = DotArrayDetector(expected_spacing=16.0)  # 根据实际情况调整

        # 执行检测 - 你可以调整这些参数
        result = detector.detect_dot_array(
            image=image,
            min_distance=15,  # 局部最大值最小距离
            threshold_abs=0   # 全检测      None  # 自动阈值
        )

        # 可视化结果
        visualizer = DotArrayVisualizer()
        output_path = output_dir / f"custom_detection_{test_image.stem}.png"
        vis_result = visualizer.visualize_detection_results(
            image=image,
            detection_result=result,
            save_path=str(output_path)
        )

    except Exception as e:
        print(f"自定义检测失败: {e}")


def main():
    """主函数"""
    print("=== 光场图像点阵检测系统 ===")

    # 检查数据目录
    data_dir = Path("data")
    if not data_dir.exists():
        print("数据目录不存在，正在创建...")
        data_dir.mkdir()

    # 列出可用的图片文件
    print(f"\n数据目录: {data_dir.absolute()}")
    image_files = []
    for ext in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
        image_files.extend(list(data_dir.glob(f"*{ext}")))
        image_files.extend(list(data_dir.glob(f"*{ext.upper()}")))

    if image_files:
        print(f"找到 {len(image_files)} 个图片文件:")
        for i, img_file in enumerate(image_files[:5]):  # 最多显示5个
            print(f"  {i+1}. {img_file.name}")
        if len(image_files) > 5:
            print(f"  ... 还有 {len(image_files) - 5} 个文件")
    else:
        print("未找到图片文件，请将测试图片放入data目录")
        print("支持的格式: .bmp, .jpg, .jpeg, .png, .tiff, .tif")
        return

    # 运行测试
    test_custom_detection()

    print("\n=== 程序结束 ===")
    print("检查output目录查看可视化结果")


if __name__ == "__main__":
    main()
