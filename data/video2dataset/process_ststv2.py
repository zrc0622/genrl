import os
import cv2
import numpy as np
import concurrent.futures
from tqdm import tqdm

def crop_and_resize(image, target_size=128):
    """
    中心裁剪并缩放图像。
    """
    h, w, c = image.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    
    # 中心裁剪
    cropped_image = image[start_y : start_y + min_dim, start_x : start_x + min_dim]
    # 缩放
    resized_image = cv2.resize(cropped_image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized_image

def process_single_video(args):
    """
    处理单个视频文件，不包含 action, reward, discount。
    """
    video_path, output_dir, target_size = args
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"Error: 无法打开视频 {video_path}"

    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 裁剪并缩放
        processed_frame = crop_and_resize(frame_rgb, target_size=target_size)
        frames.append(processed_frame)
    
    cap.release()
    
    if len(frames) == 0:
        return f"Warning: 视频 {video_name} 为空，跳过。"

    # --- 组装数据 ---
    
    # 1. 堆叠: (T, H, W, C) -> 转换维度: (T, C, H, W)
    obs_array = np.stack(frames, axis=0)
    observation = obs_array.transpose(0, 3, 1, 2)
    T = observation.shape[0]
    
    # 2. 生成必要的时序标记 (用于 DataLoader 区分视频边界)
    is_first = np.zeros((T,), dtype=bool)
    is_first[0] = True
    
    is_last = np.zeros((T,), dtype=bool)
    is_last[-1] = True
    
    is_terminal = np.copy(is_last) 
    
    # 3. 构造保存字典 (仅包含图像和时序标记)
    npz_data = {
        'observation': observation,   # uint8 or float, shape (T, C, H, W)
        'is_first': is_first,         # bool, shape (T,)
        'is_last': is_last,           # bool, shape (T,)
        'is_terminal': is_terminal,   # bool, shape (T,)
    }
    
    # 文件名格式：{VideoID}_{Length}.npz
    output_filename = f'{video_name}_{T}.npz'
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        # 使用压缩保存以节省空间
        np.savez_compressed(output_path, **npz_data)
        return None
    except Exception as e:
        return f"Error: 保存 {output_path} 失败: {e}"

def main():
    # --- 配置路径 ---
    input_root = '/mnt/mnt/public/fangzhirui/zrc/UniSkill/diffusion/workspace/datasets/ststv2/20bn-something-something-v2'
    output_root = '/mnt/mnt/public/fangzhirui/zrc/genrl/data/datasets/ststv2'
    
    TARGET_SIZE = 128
    MAX_WORKERS = 16  # 根据服务器负载调整
    
    # 扫描文件
    print(f"正在扫描目录: {input_root} ...")
    tasks = []
    
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith('.webm'):
                input_path = os.path.join(root, file)
                
                # 保持相对目录结构
                rel_path = os.path.relpath(root, input_root)
                target_dir = os.path.join(output_root, rel_path)
                os.makedirs(target_dir, exist_ok=True)
                
                tasks.append((input_path, target_dir, TARGET_SIZE))
    
    print(f"共找到 {len(tasks)} 个视频文件。")
    print(f"开始转换... (No Action, No Reward, No Discount)")
    
    # 并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(process_single_video, tasks), total=len(tasks), desc="Processing Videos"))
    
    # 错误统计
    errors = [res for res in results if res is not None]
    if errors:
        print(f"\n出现了 {len(errors)} 个错误。前 5 个错误:")
        for err in errors[:5]:
            print(err)
    else:
        print("\n所有视频处理成功！")

if __name__ == '__main__':
    main()