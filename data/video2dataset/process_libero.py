import h5py
import numpy as np
import os

def convert_hdf5_to_npz_with_custom_name(hdf5_path, output_base_dir):
    """
    将 HDF5 文件转换为每个 demo/episode 一个的 NPZ 文件。

    Args:
        hdf5_path (str): 输入的 HDF5 文件路径。
        output_base_dir (str): 输出 NPZ 文件的基础目录。
    """
    # 基于 HDF5 文件名创建输出 NPZ 文件的目录
    base_name = os.path.splitext(os.path.basename(hdf5_path))[0]
    output_dir = os.path.join(output_base_dir, base_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"输入文件: {hdf5_path}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)

    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'data' not in f:
                print(f"错误: 在 HDF5 文件中未找到 'data' 组。")
                return

            data_group = f['data']
            # 使用 sorted() 确保 demo_0, demo_1, ..., demo_10 顺序正确
            demo_keys = sorted(data_group.keys(), key=lambda x: int(x.split('_')[-1]))

            for demo_key in demo_keys:
                episode_group = data_group[demo_key]

                episode_id = int(demo_key.split('_')[-1])
                print(f"正在处理 {demo_key} (Episode {episode_id})...")

                try:
                    actions_data = episode_group['actions'][:]
                    T = actions_data.shape[0] # T 就是我们需要的 episode 长度
                except KeyError as e:
                    print(f"  警告: 在 {demo_key} 中跳过，因为缺少关键数据集: {e}")
                    continue

                action = actions_data.astype(np.float32)
                reward = episode_group['rewards'][:].astype(np.float32).reshape(T, 1)

                # ##################################################
                #  ↓↓↓ 这里是修改的部分 ↓↓↓
                # ##################################################

                # 1. 先将数据完全读入内存，成为一个 NumPy 数组
                obs_from_hdf5 = episode_group['obs']['agentview_rgb'][:]
                
                # 2. 对内存中的 NumPy 数组进行反转操作
                source_obs = obs_from_hdf5[:, ::-1, ::-1, :]
                
                # ##################################################
                #  ↑↑↑ 修改结束 ↑↑↑
                # ##################################################
                
                observation = source_obs.transpose(0, 3, 1, 2)

                dones = episode_group['dones'][:].astype(bool)
                is_last = np.zeros_like(dones)
                is_last[-1] = True

                is_terminal = np.copy(is_last)

                is_first = np.zeros_like(dones)
                is_first[0] = True

                discount = np.ones((T, 1), dtype=np.float32)

                npz_data = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'is_first': is_first,
                    'is_last': is_last,
                    'is_terminal': is_terminal,
                    'discount': discount,
                }

                output_filename = f'{episode_id}_{T}.npz'
                output_path = os.path.join(output_dir, output_filename)

                np.savez(output_path, **npz_data)
                print(f"  成功保存到: {output_path}")
                print(f"  Observation 的形状为: {observation.shape}")

        print("-" * 50)
        print("该文件的转换完成！")

    except FileNotFoundError:
        print(f"错误: 文件未找到 at '{hdf5_path}'")
    except Exception as e:
        print(f"处理文件时发生未知错误: {e}")

def main():
    """
    主函数，用于查找并处理所有的 HDF5 文件。
    """
    input_base_dir = '/mnt/mnt/public/fangzhirui/zrc/UniSkill/diffusion/workspace/datasets/LIBER0'
    output_base_dir_root = '/mnt/mnt/public/fangzhirui/zrc/genrl/data/datasets/LIBER0'

    for root, dirs, files in os.walk(input_base_dir):
        # 排除根目录本身，以便正确计算相对路径
        if root == input_base_dir:
            relative_dir = ""
        else:
            relative_dir = os.path.relpath(root, input_base_dir)

        output_dir = os.path.join(output_base_dir_root, relative_dir)

        for file in files:
            if file.endswith('.hdf5'):
                hdf5_file_path = os.path.join(root, file)
                convert_hdf5_to_npz_with_custom_name(hdf5_file_path, output_dir)

if __name__ == '__main__':
    main()