# 项目状态快照（2026-08-03 22:40）

> 本文件由 Codex 自动维护，用于会话中断后快速恢复。数据/代码/日志都在磁盘上，
> 关掉终端不影响项目文件；后台任务可能随会话结束，重启后按本文件恢复即可。

## 项目

- 目标：甲状腺结节超声良恶性分类 Q2 论文（复试用），图像 + TI-RADS 融合
- 根目录：`C:\Users\甄朝晖\Desktop\thyroid_ai`
- 环境：`C:\miniconda3\envs\lymph_yolo\python.exe`
  - torch 2.5.1+cu121（CUDA 可用，RTX 3060 6GB）
  - torchvision 0.20.1+cu121，Pillow 12.3.0

## 数据状态

| 数据集 | 状态 | 位置 |
|---|---|---|
| TN5000（5000张，良恶性） | ✅ 已下载 + manifest 就绪 | `data/thyroid/tn5000/` + `data/thyroid/manifest.csv` |
| ThyroidXL（11545张 + TI-RADS + 年龄性别） | ⏳ 等作者审核（已提交申请，token 有效） | 审核通过后下载到 `data/thyroid/thyroidxl/` |
| figshare 8508张（842例病理金标准） | ⏳ 后台龟速下载中（约20-30小时） | `data/thyroid/raw/` |
| TN-SCUI 2020（4554张） | 未下载，需 grand-challenge 登录 | `data/thyroid/tnscui/` |
| DDTI（637张） | 未下载，需 Kaggle | `data/thyroid/ddti/` |
| TN3K（3493张） | 未下载，百度网盘提取码 trfe | `data/thyroid/tn3k/` |

## 后台任务

1. **GPU 全量训练 image-only**：`logs/train_image_gpu.log`，30 epochs，
   启动时 epoch1 AUC 0.7238（CPU 版），GPU 重启后重新训练中，预计 30-60 分钟
2. **figshare 下载**：`logs/` 下 batch*_dl2.err，curl 后台
3. （可选）HuggingFace token 在 `C:\Users\甄朝晖\.huggingface\token`，
   **下载完 ThyroidXL 后应删除/吊销**（token 曾在聊天中展示）

## 关键命令

```powershell
# 训练（GPU）：image / clinical / fusion 三组消融
& C:\miniconda3\envs\lymph_yolo\python.exe -u train_thyroid.py --data_root data/thyroid --ablation image --epochs 30 --batch_size 64 --workers 4
# 评估
& C:\miniconda3\envs\lymph_yolo\python.exe -u eval_thyroid.py --weights checkpoints/thyroid/image/best.pt --split test
# 下载 ThyroidXL（等审核通过后）
& C:\miniconda3\envs\lymph_yolo\python.exe -u download_hf_dataset.py --repo hunglc007/ThyroidXL --dest data/thyroid/thyroidxl --subdirs train/images train/labels test/images test/labels
# 从 figshare RAR 生成 manifest（下载完成后）
& C:\miniconda3\envs\lymph_yolo\python.exe -u prepare_thyroid_data.py --raw_dir data/thyroid/raw --data_root data/thyroid
```

## 论文要点（融合设计）

- ThyroidXL labels（JSON）含 TI-RADS 评分 + 年龄/性别 → 临床支路特征来源（无需人工标注）
- 消融：image-only vs clinical-only vs fusion，AUC + bootstrap 95% CI + ECE + DCA
- figshare 842 例（病理金标准）作外部验证
- 报告规范：STARD / TRIPOD

## 待办

- [ ] GPU 训练完成 → 运行 eval 出测试集指标
- [ ] ThyroidXL 审核通过 → 下载 → 解析 labels JSON（TI-RADS）→ 构建融合 manifest
- [ ] figshare 下载完成 → 解压（tar.exe 支持 RAR）→ manifest → 外部验证
- [ ] 若 ThyroidXL 长时间不批 → 联系作者（越南河内内分泌医院）或先用 TN5000 + figshare

## ���£�2026-08-04 ���磩

- TN3K �����أ�376MB��HF �ٷ����� haifan-gong/TN3K��+ manifest ���ɣ�3493 ��ȫ��=test��
- TN3K �ⲿ��֤�������ݼ�ģ�ͣ���AUC 0.7131��95%CI 0.696-0.731���� ��Ư������
- ���ݼ�����������֧�� manifest �� split �У��ٷ��������ȣ�
- �����ݼ�����ѵ����TN5000+TN3K trainval��7879 �ţ���GPU ѵ���У��Ż� workers=10 + persistent_workers��
- ���Ĺ�������paper/����project_truth / decision_log / result_summary / paper_handoff / literature_matrix + manuscript.md��Methods ���壩
- figshare ���������٣���TN-SCUI �����������ֹ������
- ThyroidXL���� awaiting review
