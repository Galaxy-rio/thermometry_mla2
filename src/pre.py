import os
import cv2
import numpy as np

# ====== 参数设置 ======
exp_name = 'mla2_temp_251218_var'

src_root = os.path.join(
    r'C:\Users\galaxyrio\Desktop',
    exp_name
)

dst_root = os.path.join(
    r'C:\MS1\code\thermometry_mla2\data',
    exp_name,
    'lf_raw'
)

os.makedirs(dst_root, exist_ok=True)

folders = range(1400, 1601, 10)   # 1400, 1410, ..., 1600

# ====== 主循环 ======
for f in folders:
    src_dir = os.path.join(src_root, str(f))

    if not os.path.isdir(src_dir):
        print(f'[WARN] 文件夹不存在: {src_dir}')
        continue

    img_files = sorted([
        fn for fn in os.listdir(src_dir)
        if fn.lower().endswith('.bmp')
    ])

    if len(img_files) == 0:
        print(f'[WARN] {src_dir} 中没有 bmp 文件')
        continue

    # 读取第一张图确定尺寸
    first_img = cv2.imread(
        os.path.join(src_dir, img_files[0]),
        cv2.IMREAD_UNCHANGED
    )

    img_sum = np.zeros_like(first_img, dtype=np.float64)

    # 累加
    for fn in img_files:
        img = cv2.imread(
            os.path.join(src_dir, fn),
            cv2.IMREAD_UNCHANGED
        ).astype(np.float64)

        img_sum += img

    # 平均
    img_mean = img_sum / len(img_files)

    # 保持原始位深保存
    if first_img.dtype == np.uint16:
        img_mean = np.clip(img_mean, 0, 65535).astype(np.uint16)
    else:
        img_mean = np.clip(img_mean, 0, 255).astype(np.uint8)

    # 输出
    out_name = f'{f}.bmp'
    out_path = os.path.join(dst_root, out_name)
    cv2.imwrite(out_path, img_mean)

    print(f'[OK] 已保存: {out_name}')

print('=== 所有文件夹处理完成 ===')
