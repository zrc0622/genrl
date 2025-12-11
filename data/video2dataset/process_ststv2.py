import os
import cv2
import numpy as np
import concurrent.futures
from tqdm import tqdm
import random

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
    处理单个视频文件。
    """
    video_path, output_dir, target_size = args
    
    # 确保目标文件夹存在 (多进程中虽然可能有竞态条件，但 makedirs(exist_ok=True) 是安全的)
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # 2. 生成必要的时序标记
    is_first = np.zeros((T,), dtype=bool)
    is_first[0] = True
    
    is_last = np.zeros((T,), dtype=bool)
    is_last[-1] = True
    
    is_terminal = np.copy(is_last) 
    
    # 3. 构造保存字典
    npz_data = {
        'observation': observation,
        'is_first': is_first,
        'is_last': is_last,
        'is_terminal': is_terminal,
    }
    
    # 文件名格式：{VideoID}_{Length}.npz
    output_filename = f'{video_name}_{T}.npz'
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        # 使用压缩保存
        np.savez_compressed(output_path, **npz_data)
        return None
    except Exception as e:
        return f"Error: 保存 {output_path} 失败: {e}"

def run_batch_processing(task_list, desc_text, max_workers):
    """
    辅助函数：执行一批任务并打印结果
    """
    if not task_list:
        print(f"任务列表 {desc_text} 为空，跳过。")
        return []

    print(f"\n>>> 开始处理: {desc_text} (共 {len(task_list)} 个任务)")
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_video, task_list, chunksize=10), 
            total=len(task_list), 
            desc=desc_text,
            unit="vid"
        ))
        
    # 错误统计
    errors = [res for res in results if res is not None]
    if errors:
        print(f"--- {desc_text} 完成，但出现了 {len(errors)} 个错误。前 5 个错误:")
        for err in errors[:5]:
            print(err)
    else:
        print(f"--- {desc_text} 全部处理成功！")
        
    return results

def main():
    # --- 配置路径 ---
    input_root = '/mnt/mnt/public/fangzhirui/zrc/UniSkill/diffusion/workspace/datasets/ststv2/20bn-something-something-v2'
    
    output_root_1 = '/mnt/mnt/public/fangzhirui/zrc/genrl/data/datasets/ststv2/1'
    output_root_2 = '/mnt/mnt/public/fangzhirui/zrc/genrl/data/ststv2temp/2'
    
    TARGET_SIZE = 128
    
    MAX_WORKERS = os.cpu_count()
    if MAX_WORKERS is None:
        MAX_WORKERS = 16
    
    random.seed(42)

    print(f"使用的 CPU 核心数: {MAX_WORKERS}")
    print(f"输出路径 1 (约10%): {output_root_1}")
    print(f"输出路径 2 (约90%): {output_root_2}")
    
    # 将任务列表拆分为两个
    tasks_1 = []
    tasks_2 = []
    
    # --- 第一处 tqdm: 扫描文件 ---
    print(f"正在扫描目录: {input_root} ...")
    
    walker = list(os.walk(input_root)) 
    
    for root, dirs, files in tqdm(walker, desc="Scanning Directories"):
        for file in sorted(files):
            if file.endswith('.webm'):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_root)
                
                # 分流逻辑
                if random.random() < 0.1:
                    target_root = output_root_1
                    # 提前创建目录，避免多进程冲突（虽然process里面也有）
                    target_dir = os.path.join(target_root, rel_path)
                    tasks_1.append((input_path, target_dir, TARGET_SIZE))
                else:
                    target_root = output_root_2
                    target_dir = os.path.join(target_root, rel_path)
                    tasks_2.append((input_path, target_dir, TARGET_SIZE))
    
    print(f"\n扫描完成。")
    print(f"Set 1 (优先处理): {len(tasks_1)} 个视频")
    print(f"Set 2 (稍后处理): {len(tasks_2)} 个视频")
    
    # --- 第一阶段: 处理 1/10 ---
    run_batch_processing(tasks_1, "Phase 1/2 (Output 1)", MAX_WORKERS)
    
    # --- 第二阶段: 处理 9/10 ---
    run_batch_processing(tasks_2, "Phase 2/2 (Output 2)", MAX_WORKERS)
    
    print("\n所有阶段任务执行完毕。")

if __name__ == '__main__':
    main()