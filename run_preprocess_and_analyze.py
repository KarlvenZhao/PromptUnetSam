# -*- coding: utf-8 -*-
"""
一键运行预处理并生成数据分析报告（Markdown）。

- 预期输入：
  - 原始图像：data/mri_h_p/pflair/*.nii[.gz]
  - 原始标注：data/mri_h_p/infarct_pflair/*.nii[.gz]

- 预期输出：
  - imgs/, gts/, test_pos/{imgs,gts}, test_false/{imgs,gts}
  - img与gt的尺寸（采样检查）
  - gt像素占比统计（均值、中位、分位数、min/max），以及正负样本条数
  - 报告输出：data/sam_mri_processed_relaxed/DATASET_ANALYSIS.md
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 仓库根目录
INPUT_IMG_DIR = PROJECT_ROOT / "data" / "mri_h_p" / "pflair"
INPUT_GT_DIR = PROJECT_ROOT / "data" / "mri_h_p" / "infarct_pflair"
OUTPUT_BASE = PROJECT_ROOT / "data" / "sam_mri_processed_relaxed" / "npy" / "MR_Brain"
REPORT_PATH = PROJECT_ROOT / "data" / "sam_mri_processed_relaxed" / "DATASET_ANALYSIS.md"

PREFIX = "MR_Brain_"  # 与预处理脚本一致

def ensure_dirs():
    (OUTPUT_BASE / "imgs").mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / "gts").mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / "test_pos" / "imgs").mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / "test_pos" / "gts").mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / "test_false" / "imgs").mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / "test_false" / "gts").mkdir(parents=True, exist_ok=True)


def has_inputs() -> bool:
    if not INPUT_IMG_DIR.exists() or not INPUT_GT_DIR.exists():
        return False
    img_files = list(INPUT_IMG_DIR.glob("*.nii")) + list(INPUT_IMG_DIR.glob("*.nii.gz"))
    gt_files = list(INPUT_GT_DIR.glob("*.nii")) + list(INPUT_GT_DIR.glob("*.nii.gz"))
    return len(img_files) > 0 and len(gt_files) > 0


def run_preprocess_if_possible() -> None:
    if not has_inputs():
        print("[WARN] 未检测到原始输入数据，跳过预处理，仅进行已存在输出的分析。\n"
              f"期望输入:\n- {INPUT_IMG_DIR}\n- {INPUT_GT_DIR}")
        return
    # 通过runpy执行预处理脚本（其为顶层可执行脚本）
    import runpy
    script_path = PROJECT_ROOT / "pre_sam_mri_relaxed.py"
    if not script_path.exists():
        print(f"[ERROR] 预处理脚本不存在: {script_path}")
        return
    print(f"[INFO] 开始执行预处理脚本: {script_path}")
    t0 = time.time()
    try:
        runpy.run_path(str(script_path), run_name="__main__")
        print(f"[INFO] 预处理完成，用时 {time.time()-t0:.1f}s")
    except SystemExit as e:
        # 预处理脚本可能抛出断言或RuntimeError导致退出
        print(f"[ERROR] 预处理脚本提前退出: {e}")


# 新增：自动从 ISLES-2022 生成输入
ISLES_BASE = PROJECT_ROOT / "ISLES-2022" / "ISLES-2022"

def _extract_subj_ses_from_path(p: Path) -> Tuple[str, str]:
    """给定文件路径，返回(sub-xxxx, ses-xxxx)"""
    parts = list(p.parts)
    # 查找类似 sub-* 和 ses-* 的片段
    subj = None
    ses = None
    for i, seg in enumerate(parts):
        if subj is None and seg.startswith("sub-"):
            subj = seg
        if ses is None and seg.startswith("ses-"):
            ses = seg
        if subj and ses:
            break
    return subj or "", ses or ""


def stage_isles_inputs(max_cases: int = 300) -> int:
    """在data/mri_h_p/{pflair,infarct_pflair}下整理ISLES-2022的FLAIR与msk，返回成功配对数量。"""
    if not ISLES_BASE.exists():
        return 0
    anat_dir = ISLES_BASE
    flairs = sorted((anat_dir.glob("sub-*/ses-0001/anat/*_FLAIR.nii")))
    if not flairs:
        return 0
    INPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_GT_DIR.mkdir(parents=True, exist_ok=True)

    import shutil
    pairs = 0
    for f in flairs:
        subj, ses = _extract_subj_ses_from_path(f)
        if not subj or not ses:
            continue
        key = f"{subj}_{ses}"
        # 期待的mask路径
        m = ISLES_BASE / "derivatives" / subj / ses / f"{subj}_{ses}_msk.nii"
        if not m.exists():
            continue
        # 目标文件
        dst_img = INPUT_IMG_DIR / f"{key}.nii"
        dst_gt = INPUT_GT_DIR / f"{key}.nii"
        try:
            if not dst_img.exists():
                shutil.copy2(f, dst_img)
            if not dst_gt.exists():
                shutil.copy2(m, dst_gt)
            pairs += 1
        except Exception as e:
            print(f"[WARN] 复制失败 {key}: {e}")
        if pairs >= max_cases:
            break
    if pairs > 0:
        print(f"[INFO] 已从ISLES-2022整理 {pairs} 对(FLAIR, msk) 到: \n- {INPUT_IMG_DIR}\n- {INPUT_GT_DIR}")
    else:
        print("[WARN] 未能在ISLES-2022中配对到有效的(FLAIR, msk)文件。")
    return pairs


def list_subset_files() -> Dict[str, List[Path]]:
    subsets = {
        "train_imgs": OUTPUT_BASE / "imgs",
        "train_gts": OUTPUT_BASE / "gts",
        "test_pos_imgs": OUTPUT_BASE / "test_pos" / "imgs",
        "test_pos_gts": OUTPUT_BASE / "test_pos" / "gts",
        "test_false_imgs": OUTPUT_BASE / "test_false" / "imgs",
        "test_false_gts": OUTPUT_BASE / "test_false" / "gts",
    }
    files = {k: sorted(p.glob("*.npy")) if p.exists() else [] for k, p in subsets.items()}
    return files


def infer_case_and_slice(fname: str) -> Tuple[str, int]:
    # 形如: MR_Brain_{case}-{zzz}.npy
    n = fname
    if n.endswith('.npy'):
        n = n[:-4]
    if n.startswith(PREFIX):
        n = n[len(PREFIX):]
    # 拆分最后一个'-'
    head, sep, tail = n.rpartition('-')
    case = head if head else n
    try:
        z = int(tail)
    except Exception:
        z = -1
    return case, z


def sample_shape(files: List[Path]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """返回(img_shape, gt_shape)，若不存在返回((), ())"""
    if not files:
        return (), ()
    # 尝试找对应的gt文件进行匹配
    f = files[0]
    # 判断是imgs还是gts路径
    if f.parent.name == 'imgs':
        # 找对应的gts
        g = f.parent.parent / 'gts' / f.name
    else:
        g = f
        f = f.parent.parent / 'imgs' / f.name
    if not f.exists() or not g.exists():
        return (), ()
    try:
        img = np.load(f)
        gt = np.load(g)
        return tuple(img.shape), tuple(gt.shape)
    except Exception:
        return (), ()


def gt_ratio_stats(files: List[Path]) -> Dict[str, float]:
    if not files:
        return {"count": 0}
    ratios = []
    pos_count = 0
    neg_count = 0
    for g in files:
        try:
            gt = np.load(g)
            total = gt.size
            pos = int((gt > 0).sum())
            r = pos / total if total > 0 else 0.0
            ratios.append(r)
            if pos > 0:
                pos_count += 1
            else:
                neg_count += 1
        except Exception:
            continue
    if not ratios:
        return {"count": 0}
    a = np.array(ratios, dtype=float)
    stats = {
        "count": int(a.size),
        "positive_slices": int(pos_count),
        "negative_slices": int(neg_count),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "min": float(a.min()),
        "max": float(a.max()),
    }
    return stats


def build_markdown(files: Dict[str, List[Path]]) -> str:
    lines = []
    lines.append(f"# 数据集分析报告\n")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("")

    lines.append("## 路径")
    lines.append(f"- 项目根目录: `{PROJECT_ROOT}`")
    lines.append(f"- 输入图像目录: `{INPUT_IMG_DIR}`")
    lines.append(f"- 输入标注目录: `{INPUT_GT_DIR}`")
    lines.append(f"- 输出基础目录: `{OUTPUT_BASE}`")
    lines.append("")

    lines.append("## 文件数量")
    for k in [
        "train_imgs","train_gts","test_pos_imgs","test_pos_gts","test_false_imgs","test_false_gts"
    ]:
        lines.append(f"- {k}: {len(files.get(k, []))}")
    lines.append("")

    # 形状采样
    tr_img_shape, tr_gt_shape = sample_shape(files.get("train_imgs", []))
    tp_img_shape, tp_gt_shape = sample_shape(files.get("test_pos_imgs", []))
    tn_img_shape, tn_gt_shape = sample_shape(files.get("test_false_imgs", []))

    lines.append("## 尺寸信息（采样）")
    lines.append(f"- 训练 imgs 形状: {tr_img_shape if tr_img_shape else 'N/A'}; gts: {tr_gt_shape if tr_gt_shape else 'N/A'}")
    lines.append(f"- 测试(正) imgs 形状: {tp_img_shape if tp_img_shape else 'N/A'}; gts: {tp_gt_shape if tp_gt_shape else 'N/A'}")
    lines.append(f"- 测试(负) imgs 形状: {tn_img_shape if tn_img_shape else 'N/A'}; gts: {tn_gt_shape if tn_gt_shape else 'N/A'}")
    lines.append("")

    # 像素占比统计
    tr_stats = gt_ratio_stats(files.get("train_gts", []))
    tp_stats = gt_ratio_stats(files.get("test_pos_gts", []))
    tn_stats = gt_ratio_stats(files.get("test_false_gts", []))

    def fmt_stats(title: str, s: Dict[str, float]):
        lines.append(f"### {title}")
        if not s or s.get("count", 0) == 0:
            lines.append("- 无数据")
            lines.append("")
            return
        lines.append(f"- 切片数: {s['count']} (阳性: {s.get('positive_slices',0)}, 阴性: {s.get('negative_slices',0)})")
        lines.append(f"- 占比均值: {s['mean']:.6f}")
        lines.append(f"- 中位数: {s['median']:.6f}")
        lines.append(f"- 分位数[p25, p75]: [{s['p25']:.6f}, {s['p75']:.6f}]")
        lines.append(f"- 极值[min, max]: [{s['min']:.6f}, {s['max']:.6f}]")
        lines.append("")

    lines.append("## GT 像素占比统计")
    fmt_stats("训练集", tr_stats)
    fmt_stats("测试集(正)", tp_stats)
    fmt_stats("测试集(负)", tn_stats)

    # 估计训练集中正负条目（通过gt>0判断）
    tr_pos = tr_stats.get("positive_slices", 0)
    tr_neg = tr_stats.get("negative_slices", 0)
    lines.append("## 切片计数总结")
    lines.append(f"- 训练集: 总计={tr_pos+tr_neg}, 正样本={tr_pos}, 负样本={tr_neg}")
    lines.append(f"- 测试集: 正={tp_stats.get('count',0)}, 负={tn_stats.get('count',0)}")
    lines.append("")

    # 病例覆盖情况（按文件名前缀解析）
    def unique_cases(files_list: List[Path]) -> int:
        cases = set()
        for f in files_list:
            c, _ = infer_case_and_slice(f.name)
            cases.add(c)
        return len(cases)

    lines.append("## 病例覆盖（根据文件名推断）")
    lines.append(f"- 训练 imgs 涉及病例数: {unique_cases(files.get('train_imgs', []))}")
    lines.append(f"- 测试(正) imgs 涉及病例数: {unique_cases(files.get('test_pos_imgs', []))}")
    lines.append(f"- 测试(负) imgs 涉及病例数: {unique_cases(files.get('test_false_imgs', []))}")
    lines.append("")

    return "\n".join(lines) + "\n"


# ===========================
# 转换为 nnUNet 数据集格式
# ===========================
def convert_relaxed_to_nnunet(
    dataset_id: int = 999,
    dataset_name: str = "MedSAM_MR_Brain",
    nnunet_raw_base: Path = None,
) -> Path:
    """将当前输出(data/sam_mri_processed_relaxed/npy/MR_Brain)转换为nnUNet_raw结构。
    训练 -> imagesTr/labelsTr；测试 -> imagesTs（不冗labelTs）。
    返回数据集根路径。
    """
    tr_imgs = OUTPUT_BASE / "imgs"
    tr_gts = OUTPUT_BASE / "gts"
    ts_pos_imgs = OUTPUT_BASE / "test_pos" / "imgs"
    ts_false_imgs = OUTPUT_BASE / "test_false" / "imgs"

    if nnunet_raw_base is None:
        nnunet_raw_base = PROJECT_ROOT / "data" / "nnUNet_raw"

    ds_name = f"Dataset{dataset_id:03d}_{dataset_name}"
    ds_root = nnunet_raw_base / ds_name
    imagesTr = ds_root / "imagesTr"
    labelsTr = ds_root / "labelsTr"
    imagesTs = ds_root / "imagesTs"

    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)
    imagesTs.mkdir(parents=True, exist_ok=True)

    # 收集训练对
    tr_pairs: List[Tuple[Path, Path]] = []
    for img_npy in sorted(tr_imgs.glob("*.npy")):
        gt_npy = tr_gts / img_npy.name
        if gt_npy.exists():
            tr_pairs.append((img_npy, gt_npy))

    # 收集测试图像（正+负）
    ts_imgs = sorted(ts_pos_imgs.glob("*.npy")) + sorted(ts_false_imgs.glob("*.npy"))

    # 转存训练集
    tr_count = 0
    for idx, (img_npy, gt_npy) in enumerate(tr_pairs):
        try:
            img = np.load(img_npy)
            gt = np.load(gt_npy)
            # 保证维度：图像(1,H,W)、标签(1,H,W) - nnUNet要求单通道
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            elif img.ndim == 3:
                if img.shape[-1] == 3:  # (H,W,3) RGB图像
                    # 转为灰度图像
                    img = np.mean(img, axis=-1, keepdims=False)
                    img = img[np.newaxis, ...]  # 添加通道维度
                elif img.shape[0] == 3:  # (3,H,W) RGB图像
                    # 转为灰度图像
                    img = np.mean(img, axis=0, keepdims=False)
                    img = img[np.newaxis, ...]  # 添加通道维度
                elif img.shape[0] not in (1,3):
                    # 若为(H,W,C)，转为(C,H,W)
                    img = np.moveaxis(img, -1, 0)
            img = img.astype(np.float32)
            
            # 标签也需要添加一个维度以匹配图像
            if gt.ndim == 2:
                gt = gt[np.newaxis, ...]
            gt = gt.astype(np.uint8)

            case = f"{dataset_name}_{idx:05d}"
            img_out = imagesTr / f"{case}_0000.nii.gz"
            gt_out = labelsTr / f"{case}.nii.gz"
            nib.save(nib.Nifti1Image(img, np.eye(4)), str(img_out))
            nib.save(nib.Nifti1Image(gt, np.eye(4)), str(gt_out))
            tr_count += 1
        except Exception as e:
            print(f"[WARN] 训练样本转换失败 {img_npy.name}: {e}")

    # 转存测试集（仅图像）
    ts_count = 0
    for idx, img_npy in enumerate(ts_imgs):
        try:
            img = np.load(img_npy)
            # 保证维度：图像(1,H,W) - nnUNet要求单通道
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            elif img.ndim == 3:
                if img.shape[-1] == 3:  # (H,W,3) RGB图像
                    # 转为灰度图像
                    img = np.mean(img, axis=-1, keepdims=False)
                    img = img[np.newaxis, ...]  # 添加通道维度
                elif img.shape[0] == 3:  # (3,H,W) RGB图像
                    # 转为灰度图像
                    img = np.mean(img, axis=0, keepdims=False)
                    img = img[np.newaxis, ...]  # 添加通道维度
                elif img.shape[0] not in (1,3):
                    # 若为(H,W,C)，转为(C,H,W)
                    img = np.moveaxis(img, -1, 0)
            img = img.astype(np.float32)

            case = f"{dataset_name}Ts_{idx:05d}"
            img_out = imagesTs / f"{case}_0000.nii.gz"
            nib.save(nib.Nifti1Image(img, np.eye(4)), str(img_out))
            ts_count += 1
        except Exception as e:
            print(f"[WARN] 测试样本转换失败 {img_npy.name}: {e}")

    # 写dataset.json
    ds_root.mkdir(parents=True, exist_ok=True)
    ds_json = {
        "channel_names": {"0": "grayscale"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": tr_count,
        "file_ending": ".nii.gz",
        "dataset_name": dataset_name,
        "description": "Relaxed preprocessed MR Brain dataset converted to nnUNet format.",
        "reference": "",
        "licence": "",
        "release": "1.0",
    }
    (ds_root / "dataset.json").write_text(json.dumps(ds_json, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[NNUNet] 转换完成:")
    print(f"- 数据集根: {ds_root}")
    print(f"- imagesTr: {tr_count}")
    print(f"- labelsTr: {tr_count}")
    print(f"- imagesTs: {ts_count}")
    return ds_root


if __name__ == "__main__":
    ensure_dirs()
    # 若无输入，尝试从ISLES-2022整理
    if not has_inputs():
        pairs = stage_isles_inputs(max_cases=300)
        if pairs == 0:
            print("[WARN] 未能准备原始输入，后续仅进行已有输出的分析。")
    # 运行预处理
    run_preprocess_if_possible()
    ensure_dirs()
    # 转换为nnUNet格式（区分训练/测试）
    convert_relaxed_to_nnunet(dataset_id=999, dataset_name="MedSAM_MR_Brain", nnunet_raw_base=PROJECT_ROOT / "data" / "nnUNet_raw")
    # 生成分析报告
    files = list_subset_files()
    md = build_markdown(files)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(md, encoding="utf-8")
    print(f"[OK] 数据分析报告已生成: {REPORT_PATH}")
