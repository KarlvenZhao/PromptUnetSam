# -*- coding: utf-8 -*-
# %% import packages
# pip install connected-components-3d
import numpy as np
import SimpleITK as sitk
import os
import random

join = os.path.join
from skimage import transform
from tqdm import tqdm
import cc3d

# convert nii image to npz files, including original image and corresponding masks
modality = "MR"
anatomy = "Brain"  # anatomy + dataset name
img_name_suffix = ".nii"
gt_name_suffix = ".nii"
prefix = modality + "_" + anatomy + "_"

nii_path = "data/mri_h_p/pflair"  # path to the nii images (原始pflair图像)
gt_path = "data/mri_h_p/infarct_pflair"  # path to the ground truth (病灶标注)
# 输出目录改为./data
npy_path = os.path.join("data", "sam_mri_processed_relaxed", "npy", prefix[:-1])
os.makedirs(join(npy_path, "gts"), exist_ok=True)
os.makedirs(join(npy_path, "imgs"), exist_ok=True)

# 目标配额与划分
TARGET_POS = 120
TARGET_NEG = 60
TEST_POS = 10
TEST_NEG = 10
TRAIN_POS = TARGET_POS - TEST_POS  # 110
TRAIN_NEG = TARGET_NEG - TEST_NEG  # 50

image_size = 1024

# 更宽松的清理阈值
voxel_num_thre2d = 25    # 2D: 从100减少到25 (约5x5像素区域)
voxel_num_thre3d = 100   # 3D: 从1000减少到100

# 获取所有GT文件名
names = sorted(os.listdir(gt_path))
print(f"ori # files {len(names)=}")

# 检查对应的图像文件是否存在
names = [
    name
    for name in names
    if os.path.exists(join(nii_path, name))
]
print(f"after sanity check # files {len(names)=}")

# set label ids that are excluded (对于我们的数据，不需要排除特定标签)
remove_label_ids = []  # 保留所有病灶标签

tumor_id = None  # 我们的数据是单一病灶标注，不需要实例分割

# 统计变量
total_cases = 0
valid_cases = 0
total_slices = 0

# ---------- 辅助函数 ----------

def clear_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    for fn in os.listdir(dir_path):
        if fn.endswith('.npy'):
            try:
                os.remove(join(dir_path, fn))
            except Exception as e:
                print(f"清理{dir_path}文件{fn}失败: {e}")

def preprocess_volume(image_data):
    if modality == "CT":
        WINDOW_LEVEL = 40
        WINDOW_WIDTH = 400
        lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
        upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / max(np.max(image_data_pre) - np.min(image_data_pre), 1e-8)
            * 255.0
        )
    else:
        nonzero = image_data[image_data > 0]
        if nonzero.size == 0:
            lower_bound, upper_bound = 0, 1
        else:
            lower_bound, upper_bound = np.percentile(nonzero, 0.5), np.percentile(nonzero, 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / max(np.max(image_data_pre) - np.min(image_data_pre), 1e-8)
            * 255.0
        )
        image_data_pre[image_data == 0] = 0
    return np.uint8(image_data_pre)

def clean_gt(gt_vol):
    # 3D 清理
    gt_vol = cc3d.dust(gt_vol, threshold=voxel_num_thre3d, connectivity=26, in_place=True)
    # 2D 清理
    for slice_i in range(gt_vol.shape[0]):
        gt_i = gt_vol[slice_i, :, :]
        gt_vol[slice_i, :, :] = cc3d.dust(gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True)
    return gt_vol

def resize_and_save(img_slice, gt_slice, out_img_path, out_gt_path):
    img_3c = np.repeat(img_slice[:, :, None], 3, axis=-1)
    resize_img = transform.resize(
        img_3c,
        (image_size, image_size),
        order=3,
        preserve_range=True,
        mode="constant",
        anti_aliasing=True,
    )
    resize_img_01 = (resize_img - resize_img.min()) / np.clip(resize_img.max() - resize_img.min(), a_min=1e-8, a_max=None)
    resize_img_01 = resize_img_01.astype(np.float32)
    resize_gt = transform.resize(
        gt_slice,
        (image_size, image_size),
        order=0,
        preserve_range=True,
        mode="constant",
        anti_aliasing=False,
    ).astype(np.uint8)
    assert resize_img_01.shape[:2] == resize_gt.shape
    np.save(out_img_path, resize_img_01)
    np.save(out_gt_path, resize_gt)

def load_cached_volume(cache, img_path, gt_path):
    key = (img_path, gt_path)
    if key in cache:
        return cache[key]
    img_sitk = sitk.ReadImage(img_path)
    image_data = sitk.GetArrayFromImage(img_sitk)
    gt_sitk = sitk.ReadImage(gt_path)
    gt_data = np.uint8(sitk.GetArrayFromImage(gt_sitk))
    # 移除不需要的标签
    for remove_label_id in remove_label_ids:
        gt_data[gt_data == remove_label_id] = 0
    gt_clean = clean_gt(gt_data.copy())
    img_pre = preprocess_volume(image_data)
    cache[key] = (img_pre, gt_clean)
    return cache[key]

# ---------- 辅助函数 ----------

def get_case_id_from_filename(fname: str) -> str:
    # 支持 .nii 和 .nii.gz
    if fname.endswith('.nii.gz'):
        return fname[:-7]
    if fname.endswith('.nii'):
        return fname[:-4]
    # 回退：去掉一次扩展名
    return os.path.splitext(fname)[0]

# ---------- 主流程：收集候选meta ----------
pos_meta = []  # (case_id, img_path, gt_path, z)
neg_meta = []  # (case_id, img_path, gt_path, z)

# %% save preprocessed images and masks as npz files
print("使用更宽松的清理阈值:")
print(f"- 3D连通组件阈值: {voxel_num_thre3d} (原来1000)")
print(f"- 2D连通组件阈值: {voxel_num_thre2d} (原来100)")
print()

for name in tqdm(names):  # 处理所有病例
    total_cases += 1
    image_name = name  # 图像文件名与GT文件名相同
    gt_name = name
    img_path_i = join(nii_path, image_name)
    gt_path_i = join(gt_path, gt_name)
    case_id = get_case_id_from_filename(gt_name)
    
    # 读取GT
    gt_sitk = sitk.ReadImage(gt_path_i)
    gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
    
    original_lesion_pixels = np.sum(gt_data_ori > 0)
    
    # remove label ids (我们的数据不需要移除特定标签)
    for remove_label_id in remove_label_ids:
        gt_data_ori[gt_data_ori == remove_label_id] = 0
    
    # label tumor masks as instances and remove from gt_data_ori
    if tumor_id is not None:
        tumor_bw = np.uint8(gt_data_ori == tumor_id)
        gt_data_ori[tumor_bw > 0] = 0
        # label tumor masks as instances
        tumor_inst, tumor_n = cc3d.connected_components(
            tumor_bw, connectivity=26, return_N=True
        )
        # put the tumor instances back to gt_data_ori
        gt_data_ori[tumor_inst > 0] = (
            tumor_inst[tumor_inst > 0] + np.max(gt_data_ori) + 1
        )

    # exclude the objects with less pixels in 3D (更宽松的阈值)
    gt_data_ori = cc3d.dust(
        gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
    )
    
    after_3d_pixels = np.sum(gt_data_ori > 0)
    
    # remove small objects with less pixels in 2D slices (更宽松的阈值)
    for slice_i in range(gt_data_ori.shape[0]):
        gt_i = gt_data_ori[slice_i, :, :]
        # remove small objects with less pixels
        gt_data_ori[slice_i, :, :] = cc3d.dust(
            gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
        )
    
    after_2d_pixels = np.sum(gt_data_ori > 0)
    
    # find non-zero slices
    z_index, _, _ = np.where(gt_data_ori > 0)
    z_index = np.unique(z_index)

    if len(z_index) > 0:
        valid_cases += 1
        total_slices += len(z_index)
        
        # load image and preprocess (仅用于保存npz/nii sanity，不用于切片持久化)
        img_sitk = sitk.ReadImage(img_path_i)
        image_data = sitk.GetArrayFromImage(img_sitk)
        depth_img = image_data.shape[0]
        depth_gt = gt_data_ori.shape[0]
        depth_min = min(depth_img, depth_gt)
        depth_min = min(depth_img, depth_gt)
        
        # 依据共同深度过滤z_index，避免越界
        z_index = z_index[z_index < depth_min]
        if len(z_index) == 0:
            # 若过滤后无正切片，则按无病灶处理
            for z in range(depth_min):
                neg_meta.append((case_id, img_path_i, gt_path_i, int(z)))
            print(f"□ {name}: 过滤后无有效正切片，按无病灶加入{depth_min}个负样本切片")
            continue
        
        # crop the ground truth with non-zero slices
        gt_roi = gt_data_ori[z_index, :, :]
        
        # nii preprocess start (使用MR预处理方法)
        if modality == "CT":
            WINDOW_LEVEL = 40
            WINDOW_WIDTH = 400
            lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
            upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
        else:
            # MR图像预处理：使用0.5%和99.5%分位数
            lower_bound, upper_bound = np.percentile(
                image_data[image_data > 0], 0.5
            ), np.percentile(image_data[image_data > 0], 99.5)
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
            image_data_pre[image_data == 0] = 0

        image_data_pre = np.uint8(image_data_pre)
        img_roi = image_data_pre[z_index, :, :]
        
        # 保存npz文件（保留原有行为）
        case_id = get_case_id_from_filename(gt_name)
        np.savez_compressed(
            join(npy_path, prefix + case_id + '.npz'), 
            imgs=img_roi, 
            gts=gt_roi, 
            spacing=img_sitk.GetSpacing()
        )
        
        # save the image and ground truth as nii files for sanity check;
        img_roi_sitk = sitk.GetImageFromArray(img_roi)
        gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
        sitk.WriteImage(
            img_roi_sitk,
            join(npy_path, prefix + case_id + "_img.nii.gz"),
        )
        sitk.WriteImage(
            gt_roi_sitk,
            join(npy_path, prefix + case_id + "_gt.nii.gz"),
        )
        
        # 收集正样本meta（使用原体素切片索引z）
        for z in z_index:
            pos_meta.append((case_id, img_path_i, gt_path_i, int(z)))
        
        # 收集负样本meta（与z_index互补的切片，使用共同深度）
        all_z = np.arange(depth_min)
        neg_z = np.setdiff1d(all_z, z_index)
        for z in neg_z:
            neg_meta.append((case_id, img_path_i, gt_path_i, int(z)))
        
        print(f"✓ {name}: 原始{original_lesion_pixels}→3D清理{after_3d_pixels}→2D清理{after_2d_pixels}像素, {len(z_index)}切片")
    else:
        if original_lesion_pixels > 0:
            print(f"✗ {name}: 原始{original_lesion_pixels}→3D清理{after_3d_pixels}→2D清理{after_2d_pixels}像素, 被过滤")
        else:
            # 无病灶病例：将该病例的全部切片加入负样本候选（以共同深度）
            img_sitk = sitk.ReadImage(img_path_i)
            image_data = sitk.GetArrayFromImage(img_sitk)
            depth_min = min(image_data.shape[0], gt_data_ori.shape[0])
            for z in range(depth_min):
                neg_meta.append((case_id, img_path_i, gt_path_i, int(z)))
            print(f"□ {name}: 原始无病灶标注，加入{depth_min}个负样本切片")
        print(f"✗ {name}: 原始无病灶标注")

# ---------- 统一抽样与写入 ----------
random.seed(42)
np.random.seed(42)

print("\n开始抽样与划分...")
print(f"候选正样本: {len(pos_meta)} slices, 候选负样本: {len(neg_meta)} slices")

if len(pos_meta) < TARGET_POS:
    raise RuntimeError(f"正样本不足: 需要{TARGET_POS}, 实际{len(pos_meta)}")
if len(neg_meta) < TARGET_NEG:
    raise RuntimeError(f"负样本不足: 需要{TARGET_NEG}, 实际{len(neg_meta)}")

# 打乱
random.shuffle(pos_meta)
random.shuffle(neg_meta)

pos_sel = pos_meta[:TARGET_POS]
neg_sel = neg_meta[:TARGET_NEG]

# 划分测试与训练
test_pos_list = pos_sel[:TEST_POS]
train_pos_list = pos_sel[TEST_POS:TARGET_POS]

test_neg_list = neg_sel[:TEST_NEG]
train_neg_list = neg_sel[TEST_NEG:TARGET_NEG]

assert len(train_pos_list) == TRAIN_POS
assert len(train_neg_list) == TRAIN_NEG

# 目标目录
train_imgs_dir = join(npy_path, "imgs")
train_gts_dir = join(npy_path, "gts")

test_pos_imgs_dir = join(npy_path, "test_pos", "imgs")
test_pos_gts_dir = join(npy_path, "test_pos", "gts")

test_neg_imgs_dir = join(npy_path, "test_false", "imgs")
test_neg_gts_dir = join(npy_path, "test_false", "gts")

# 清理并创建目录
clear_dir(train_imgs_dir)
clear_dir(train_gts_dir)
clear_dir(test_pos_imgs_dir)
clear_dir(test_pos_gts_dir)
clear_dir(test_neg_imgs_dir)
clear_dir(test_neg_gts_dir)

# 写入函数（基于缓存避免重复读取体数据）
cache = {}

def write_entries(entries, out_imgs, out_gts):
    for (case_id, img_p, gt_p, z) in entries:
        img_pre, gt_clean = load_cached_volume(cache, img_p, gt_p)
        depth_min = min(img_pre.shape[0], gt_clean.shape[0])
        if not (0 <= z < depth_min):
            # 跳过不合法索引（稳健处理）
            continue
        img_slice = img_pre[z, :, :]
        gt_slice = gt_clean[z, :, :]
        fn = prefix + case_id + "-" + str(z).zfill(3) + ".npy"
        resize_and_save(img_slice, gt_slice, join(out_imgs, fn), join(out_gts, fn))

print("写入训练集(不区分正负)...")
write_entries(train_pos_list + train_neg_list, train_imgs_dir, train_gts_dir)

print("写入测试集(正样本)...")
write_entries(test_pos_list, test_pos_imgs_dir, test_pos_gts_dir)

print("写入测试集(负样本)...")
write_entries(test_neg_list, test_neg_imgs_dir, test_neg_gts_dir)

print("\n=== 预处理与划分完成 ===")
print(f"总病例数: {total_cases}")
print(f"有效病例: {valid_cases} ({valid_cases/max(total_cases,1)*100:.1f}%)")
print(f"正样本(切片) 训练: {len(train_pos_list)} 测试: {len(test_pos_list)} 共: {len(pos_sel)}")
print(f"负样本(切片) 训练: {len(train_neg_list)} 测试: {len(test_neg_list)} 共: {len(neg_sel)}")
print(f"训练集共: {len(train_pos_list)+len(train_neg_list)} (imgs/gts)")
print(f"测试集: test_pos={len(test_pos_list)}, test_false={len(test_neg_list)}")
print(f"训练集路径: {train_imgs_dir} / {train_gts_dir}")
print(f"测试集路径: {join(npy_path, 'test_pos')} / {join(npy_path, 'test_false')}")
