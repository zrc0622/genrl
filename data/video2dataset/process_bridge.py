import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import rlds
from tqdm import tqdm
import cv2
import concurrent.futures

def crop_and_resize(image, target_size=128):
    """辅助函数：裁剪并缩放图像。"""
    h, w, c = image.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped_image = image[start_y : start_y + min_dim, start_x : start_x + min_dim]
    resized_image = cv2.resize(cropped_image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized_image

def process_and_save_episode(args):
    """
    这是一个“工人”函数，现在它接收的是纯 NumPy 和 Python 对象。
    """
    # 解包预处理好的数据
    processed_steps, episode_index, target_dir, task_name, image_size = args

    # 为当前 episode 初始化数据收集列表
    episode_observations = []
    episode_actions = []
    episode_rewards = []
    episode_is_first = []
    episode_is_last = []
    episode_is_terminal = []

    # 直接遍历预处理好的 steps 列表
    for step_data in processed_steps:
        episode_observations.append(step_data['observation_image'])
        episode_actions.append(step_data['action'])
        episode_rewards.append(step_data['reward'])
        episode_is_first.append(step_data['is_first'])
        episode_is_last.append(step_data['is_last'])
        episode_is_terminal.append(step_data['is_terminal'])
    
    # --- 将收集的数据转换为 NumPy 数组 ---
    obs_array = np.stack(episode_observations, axis=0)
    obs_array_transposed = obs_array.transpose(0, 3, 1, 2)
    
    action_array = np.stack(episode_actions, axis=0)
    reward_array = np.stack(episode_rewards, axis=0).reshape(-1, 1)
    
    is_first_array = np.stack(episode_is_first, axis=0)
    is_last_array = np.stack(episode_is_last, axis=0)
    is_terminal_array = np.stack(episode_is_terminal, axis=0)
    
    discount_array = (1.0 - is_terminal_array.astype(np.float32)).reshape(-1, 1)
    episode_length = len(obs_array)

    # --- 准备并保存 .npz 文件 ---
    npz_payload = {
        'observation': obs_array_transposed,
        'action': action_array,
        'reward': reward_array,
        'is_first': is_first_array,
        'is_last': is_last_array,
        'is_terminal': is_terminal_array,
        'discount': discount_array,
    }
    
    output_filename = f'{episode_index}_{task_name}_{episode_length}.npz'
    output_path = os.path.join(target_dir, output_filename)
    
    np.savez_compressed(output_path, **npz_payload)
    
    return output_path

def convert_bridgedata_to_npz(builder, target_dir, split_name, image_size=128, max_workers=None):
    """
    这是“管理者”函数，它现在负责在主进程中完成所有 TF 到 NumPy 的转换。
    """
    os.makedirs(target_dir, exist_ok=True)
    task_name = os.path.basename(target_dir)

    print("\n" + "=" * 70)
    print(f"开始并行处理 Split: '{split_name}'")
    print(f"目标目录: {target_dir}")
    print(f"使用的最大工人进程数: {max_workers or '默认 (CPU核心数)'}")
    print("=" * 70)

    dataset = builder.as_dataset(split=split_name)

    # --- 核心修改点：在主进程中预处理所有数据 ---
    print("正在预处理数据集 (将 TF Tensors 转换为 NumPy)...")
    preprocessed_episodes = []
    # 使用 tqdm 包装 dataset 的迭代
    for episode_data in tqdm(dataset, desc="预处理 Episodes"):
        processed_steps = []
        for step in episode_data[rlds.STEPS]:
            # a) 处理 Action
            action_dict = step['action']
            world_vector = action_dict['world_vector'].numpy()
            rotation_delta = action_dict['rotation_delta'].numpy()
            gripper_action = 1.0 if action_dict['open_gripper'].numpy() else -1.0
            action_vector = np.concatenate([
                world_vector, rotation_delta, np.array([gripper_action], dtype=np.float32)
            ], axis=0)

            # b) 处理 Observation (只处理图像)
            image_tensor = step['observation']['image']
            image_numpy_original = image_tensor.numpy()
            resized_image = crop_and_resize(image_numpy_original, target_size=image_size)
            
            # c) 将所有数据打包成一个纯 Python/NumPy 字典
            processed_steps.append({
                'observation_image': resized_image,
                'action': action_vector,
                'reward': step['reward'].numpy(),
                'is_first': step['is_first'].numpy(),
                'is_last': step['is_last'].numpy(),
                'is_terminal': step['is_terminal'].numpy(),
            })
        preprocessed_episodes.append(processed_steps)
    
    num_total_episodes = len(preprocessed_episodes)
    print(f"预处理完成，共 {num_total_episodes} 个 episodes。")
    # -----------------------------------------------

    # 创建任务参数列表，现在传递的是预处理好的数据
    tasks = []
    for i, processed_steps_data in enumerate(preprocessed_episodes):
        tasks.append((processed_steps_data, i, target_dir, task_name, image_size))

    # 使用 ProcessPoolExecutor 进行并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_and_save_episode, tasks), 
                            total=num_total_episodes, 
                            desc=f"并行转换 {split_name} Episodes"))
    
    print(f"\n转换完成！'{split_name}' split 的所有 {len(results)} 个 episode 已成功保存。")


# --- 主执行程序 (保持不变) ---
if __name__ == "__main__":
    # ... (与之前版本相同) ...
    source_dataset_path = '/mnt/mnt/public/fangzhirui/zrc/datasets/BridgeDataV2/OpenDataLab___BridgeData_V2/raw/bridge/0.1.0'
    target_base_dir = '/mnt/mnt/public/fangzhirui/zrc/genrl/data/datasets/BridgeDataV2'
    WORKER_COUNT = None
    
    print("=" * 70)
    print("正在初始化 TFDS Builder...")
    try:
        builder = tfds.builder_from_directory(builder_dir=source_dataset_path)
        print("Builder 初始化成功。")
    except Exception as e:
        print(f"错误：无法从 '{source_dataset_path}' 创建 builder。请检查路径。")
        print(f"详细错误: {e}")
        exit()

    splits_to_process = {'test': 'test' }

    for split_name, sub_dir_name in splits_to_process.items():
        target_directory = os.path.join(target_base_dir, sub_dir_name)
        convert_bridgedata_to_npz(builder,
                                  target_directory,
                                  split_name=split_name,
                                  image_size=128,
                                  max_workers=WORKER_COUNT)

    print("\n" + "=" * 70)
    print("所有数据转换任务已全部完成！")
    print("=" * 70)