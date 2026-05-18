"""
3D 汽车任务：SDF 数据批量采样与预处理脚本
说明：
1. 从给定的根目录自动遍历所有子文件夹中的 STL 格式模型。
2. 保持原有的目录结构，将采样后的 SDF 数据 (.npz) 保存到统一的输出目录。
3. 采用混合采样策略 (50% 表面 + 50% 顶点)，并加入多级高斯噪声。
"""

import trimesh
import numpy as np
import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mmnpy
import time
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc=None):
        return iterable

# ==============================================================================
# ======= [全局参数配置字典] ================
# ==============================================================================
CONFIG = {
    # --- 路径配置 ---
    # 汽车原始数据根目录
    "DATA_ROOT": "/mnt/sunguoze/3DMeshesSTL",
    # 统一的输出目录（会在此目录下自动重建子文件夹结构）
    "SAVE_DIR": "/mnt/sunguoze/processed_data/car_sdf_hybrid",
    
    # --- 采样超参数 ---
    "TOTAL_SAMPLES": 100000,
    
    # --- 并发配置 ---
    "NUM_WORKERS": 32,
    "FILE_EXTENSION": "*.stl"
}

# ==============================================================================
# 单个网格处理核心逻辑 (算法逻辑保持原样不变)
# ==============================================================================
def process_single_mesh(args):
    input_path, output_path, total_samples = args
    
    if os.path.exists(output_path):
        return True, f"Skipped (Exists): {os.path.basename(input_path)}"

    try:
        # ---------------------------------------------------------
        # A. 几何修复 (Trimesh)
        # ---------------------------------------------------------
        t_mesh = trimesh.load(input_path, force='mesh')
        components = t_mesh.split(only_watertight=False)
        fixed_components = []
        
        for comp in components:
            if len(comp.faces) < 50: continue
            try:
                if comp.volume < 0:
                    comp.invert()
            except Exception:
                trimesh.repair.fix_normals(comp)
            fixed_components.append(comp)
            
        if not fixed_components:
            return False, "Mesh empty after filtering."

        t_mesh_fixed = trimesh.util.concatenate(fixed_components)
        faces_fixed = t_mesh_fixed.faces.astype(np.int32)
        
        # ---------------------------------------------------------
        # B. 转换到 MeshLib 环境
        # ---------------------------------------------------------
        vertices_initial = t_mesh_fixed.vertices.astype(np.float32)
        mesh = mmnpy.meshFromFacesVerts(faces_fixed, vertices_initial)
        if mesh is None:
            return False, "Conversion failed."

        # ---------------------------------------------------------
        # C. 归一化 (Normalization)
        # ---------------------------------------------------------
        bbox = mesh.computeBoundingBox()
        diagonal = bbox.diagonal()
        if diagonal < 1e-6: diagonal = 1.0
        
        s = 1.9 / diagonal
        scale_mtx = mm.Matrix3f()
        scale_mtx.x = mm.Vector3f(s, 0, 0)
        scale_mtx.y = mm.Vector3f(0, s, 0)
        scale_mtx.z = mm.Vector3f(0, 0, s)
        
        xf = mm.AffineXf3f()
        xf.A = scale_mtx
        xf.b = mm.Vector3f(-bbox.center().x * s, -bbox.center().y * s, -bbox.center().z * s)
        mesh.transform(xf)

        # ---------------------------------------------------------
        # D. 混合采样策略 (Hybrid Sampling)
        # ---------------------------------------------------------
        n_global = int(total_samples * 0.10)
        n_near   = total_samples - n_global 
        n_source_vertex = n_near // 2
        n_source_surface = n_near - n_source_vertex
        
        verts_normalized = mmnpy.getNumpyVerts(mesh).astype(np.float32)
        n_verts = len(verts_normalized)
        
        idx_v = np.random.choice(n_verts, n_source_vertex, replace=(n_source_vertex > n_verts))
        source_points_vertex = verts_normalized[idx_v]
        
        temp_mesh = trimesh.Trimesh(vertices=verts_normalized, faces=faces_fixed)
        source_points_surface, _ = trimesh.sample.sample_surface(temp_mesh, n_source_surface)
        source_points_surface = source_points_surface.astype(np.float32)
        
        source_points_combined = np.vstack([source_points_vertex, source_points_surface])
        np.random.shuffle(source_points_combined)
        
        n_small = int(total_samples * 0.45)
        n_medium = n_near - n_small 
        
        points_base_small = source_points_combined[:n_small]
        points_base_medium = source_points_combined[n_small:]
        
        noise_small = np.random.normal(0, 0.005, points_base_small.shape).astype(np.float32)
        points_query_small = points_base_small + noise_small
        
        noise_medium = np.random.normal(0, 0.025, points_base_medium.shape).astype(np.float32)
        points_query_medium = points_base_medium + noise_medium
        
        points_query_global = np.random.uniform(-1.0, 1.0, (n_global, 3)).astype(np.float32)
        query_points_np = np.vstack([points_query_small, points_query_medium, points_query_global])

        # ---------------------------------------------------------
        # E. 计算真实 SDF 值
        # ---------------------------------------------------------
        mesh_part = mm.MeshPart(mesh)
        signed_distances = []
        
        for i in range(len(query_points_np)):
            p = query_points_np[i]
            pt = mm.Vector3f(float(p[0]), float(p[1]), float(p[2]))
            res = mm.findSignedDistance(pt, mesh_part)
            
            if hasattr(res, 'dist'): val = res.dist
            elif hasattr(res, 'signedDist'): val = res.signedDist
            else: val = float(res)
            signed_distances.append(val)

        sdf_values = np.array(signed_distances, dtype=np.float32).reshape(-1, 1)

        # ---------------------------------------------------------
        # F. 保存
        # ---------------------------------------------------------
        perm = np.random.permutation(len(query_points_np))
        final_coords = query_points_np[perm]
        final_sdf = sdf_values[perm]

        np.savez_compressed(output_path, coords=final_coords, sdf=final_sdf)
        return True, f"Success: {os.path.basename(input_path)}"

    except Exception as e:
        return False, f"Error processing {os.path.basename(input_path)}: {str(e)}"

# ==============================================================================
# 批量目录遍历与执行逻辑
# ==============================================================================
def main():
    data_root = CONFIG["DATA_ROOT"]
    save_dir = CONFIG["SAVE_DIR"]
    
    if not os.path.exists(data_root):
        print(f"错误: 找不到数据根目录 {data_root}")
        return

    # 1. 递归搜索所有模型文件
    search_pattern = os.path.join(data_root, "**", CONFIG["FILE_EXTENSION"])
    all_files = glob.glob(search_pattern, recursive=True)
    
    if not all_files:
        print(f"未在 {data_root} 中找到任何 {CONFIG['FILE_EXTENSION']} 文件。")
        return
        
    print(f"找到总计 {len(all_files)} 个 3D 模型文件。开始构建任务列表...")

    # 2. 构建任务列表，自动重建子目录结构
    tasks = []
    for input_file in all_files:
        # 获取文件相对于 DATA_ROOT 的路径 (例如: 子文件夹A/model_1.stl)
        rel_path = os.path.relpath(input_file, data_root)
        
        # 替换后缀并拼接到 SAVE_DIR
        out_rel_path = os.path.splitext(rel_path)[0] + ".npz"
        output_file = os.path.join(save_dir, out_rel_path)
        
        # 确保输出的子文件夹存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        tasks.append((input_file, output_file, CONFIG["TOTAL_SAMPLES"]))

    # 3. 多进程处理
    print(f"启动多进程处理，分配了 {CONFIG['NUM_WORKERS']} 个 Worker...")
    success_count = 0
    fail_count = 0
    
    with ProcessPoolExecutor(max_workers=CONFIG["NUM_WORKERS"]) as executor:
        futures = [executor.submit(process_single_mesh, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(tasks), desc="Processing Meshes"):
            success, msg = f.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                # 失败时打印警告，方便追溯
                tqdm.write(f"WARNING: {msg}")

    print("-" * 50)
    print(f"处理完成！成功: {success_count}，失败: {fail_count}")
    print(f"所有数据已保存至: {save_dir}")

if __name__ == "__main__":
    main()