# 参考文献交叉验证报告

稿件：`paper/output/doc/manuscript.md`（甲状腺超声 AI 论文，目标 BMC Medical Imaging）
验证方法：Crossref API 逐条解析 DOI → 字段级比对（作者/标题/年份/卷/期/页码）→ 缺失或异常项降级到 PubMed / 出版方官网 / Springer / MICCAI 官网 二次核验
验证日期：2026-08-13

## 总览

- 共 22 条参考文献，22/22 交叉验证通过（**Verified**）
- 21/22 可在 Crossref 解析；1 条（#8，中文期刊）Crossref 无收录，经 PubMed + 期刊官网确认无误
- 需要处理的项：**1 条（#19）建议修改**（标题多出 "(TTM-Net)"）；其余为 Info 级标注，不影响投稿
- 正文引用编号：22 条全部被引用，首次出现顺序 1→22 与文末列表完全一致，无缺引、无越界

## 逐条结果

| # | 引用（稿件） | DOI 解析 | 结论 |
|---|-------------|---------|------|
| 1 | Mu C, et al. Front Oncol. 2022;12:1029926 | ✅ Crossref | **Verified**：标题/首作者/期刊/卷/文章号/年份全一致 |
| 2 | Acosta GJ, et al. Curr Opin Endocrinol Diabetes Obes. 2023;30(5):225–230 | ✅ Crossref | **Verified**：Info—期刊官方名用 "&"（Diabetes **&** Obesity），稿件为 "and" |
| 3 | Tessler FN, et al. JACR. 2017;14(5):587–595 | ✅ Crossref | **Verified**：逐字段一致 |
| 4 | Sharifi Y, et al. WFUMB Ultrasound Open. 2025;3(1):100082 | ✅ Crossref | **Verified**：逐字段一致 |
| 5 | Savelonas M. BDCC. 2025;9(10):255 | ✅ Crossref | **Verified**：逐字段一致 |
| 6 | Gong H, et al. Comput Biol Med. 2023;155:106389 | ✅ Crossref | **Verified**：Info—DOI 后缀为 2022（在线 2022），印刷版 2023；稿件年份 2023 正确 |
| 7 | Ni J, et al. JMIR. 2025;27:e73516 | ✅ Crossref | **Verified**：逐字段一致 |
| 8 | Zhou X, et al. J Biomed Eng. 2025;42(5):1069–1075 | ⚠️ Crossref 404，PubMed/官网 ✅ | **Verified**：中文期刊未收录于 Crossref，属平台覆盖问题而非错误。PubMed PMID 41084019 与期刊官网均确认作者/标题/年卷期页完全一致 |
| 9 | Pedraza L, et al. Proc SPIE. 2015;9287:188–193 | ✅ Crossref | **Verified**：Info—Crossref 以文章号 92870W 著录，188–193 为印刷页码（DDTI 标准引用，openmedlab 同款著录） |
| 10 | Gong H, et al. IEEE ISBI 2021 | ✅ Crossref | **Verified**：Info—期刊用简称 "IEEE ISBI 2021"，Crossref 全称 "2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)"；页码 257–261 稿件未写（可加可不加） |
| 11 | Gong H, et al. MICCAI 2022. LNCS 13434:248–257 | ✅ Crossref + Springer | **Verified**：页码/年份一致；LNCS 13434（Part IV）经 Springer ISBN 978-3-031-16440-8 确认 |
| 12 | Zhang H, et al. Sci Data. 2025;12:1437 | ✅ Crossref | **Verified**：逐字段一致 |
| 13 | Duong VH, et al. MICCAI 2025. LNCS 15974:616–626 | ✅ Crossref + Springer | **Verified**：Crossref issued 2025-09-18、页码 616–626；LNCS 15974（Part XV）经 Springer ISBN 978-3-032-05182-0 及多机构著录确认 |
| 14 | Jin Z, et al. Int J Cancer. 2022;151(12):2229–2243 | ✅ Crossref | **Verified**：逐字段一致（Crossref 标题中的 `<scp>` 为 SCP 标记，非差异） |
| 15 | Bossuyt PM, et al. BMJ. 2015;351:h5527 | ✅ Crossref | **Verified**：逐字段一致 |
| 16 | Collins GS, et al. BMJ. 2024;385:e078378 | ✅ Crossref | **Verified**：逐字段一致 |
| 17 | Duan Y, et al. MICCAI 2025. LNCS 15975:35–45 | ✅ Crossref + Springer | **Verified**：Crossref issued 2025-09-20、页码 35–45；LNCS 15975（Part XVI）经 Springer/澳门理工研究著录确认 |
| 18 | Yu R, et al. Pattern Recognit Lett. 2026;206:120–127 | ✅ Crossref | **Verified**：逐字段一致 |
| 19 | Yu M, et al. Expert Syst Appl. 2026;312:131440 | ✅ Crossref | **Verified，建议修改**：⚠️ 官方标题（ScienceDirect/Crossref/ACM DL 三处一致）为 "A two-stage multimodal learning framework based on text-driven vision pretraining and cross-modal feature fusion for thyroid ultrasound diagnosis"，不含 "(TTM-Net)"；TTM-Net 是正文中的模型名。建议把标题中的 "(TTM-Net)" 删掉 |
| 20 | Xiang T, Hu Z. J Imaging Inform Med. 2026 | ✅ Crossref | **Verified**：在线优先文章，尚未分配卷/页码，稿件不写卷页正确 |
| 21 | Li Y, et al. Ultrasound Med Biol. 2026;52(8):1763–1774 | ✅ Crossref | **Verified**：Info—期刊官方名用 "&"（Ultrasound in Medicine **&** Biology），稿件为 "and" |
| 22 | Sherif M, et al. Sci Rep. 2026;16:22408 | ✅ Crossref | **Verified**：逐字段一致 |

## 需要处理（1 项）

**#19 标题多出 "(TTM-Net)"**

- 当前值：`A two-stage multimodal learning framework based on text-driven vision pretraining and cross-modal feature fusion for thyroid ultrasound diagnosis (TTM-Net)`
- 正确值：`A two-stage multimodal learning framework based on text-driven vision pretraining and cross-modal feature fusion for thyroid ultrasound diagnosis`
- 来源：ScienceDirect（官方页面）、Crossref 元数据、ACM DL 三处一致，均不含 "(TTM-Net)"
- 建议：删除标题末尾的 "(TTM-Net)"，DOI 保持不变

## Info 级备注（无需修改，供审阅）

- #2、#21：期刊官方名含 "&"，稿件用 "and"（BMJ 期刊著录习惯可接受，不改不影响）
- #6：卷年 2023 vs DOI 编号年 2022，稿件取印刷年 2023，正确
- #9：Crossref 用 SPIE 文章号 92870W，稿件用印刷页码 188–193，均为该文标准著录
- #10：期刊名缩略写法；可考虑补页码 257–261（非必须）
- #13：MICCAI 官网摘要页写 11,545 张图，但论文 PDF 正文写 11,635 张；稿件用 11,635 与 PDF 及本地数据一致，维持不变（投稿前可留意审稿人是否引用摘要页数字）
- #13/#17：LNCS 印刷年为 2026、会议年为 2025，稿件按会议年 2025 著录，符合会议论文惯例

## 结论

22 条参考文献全部真实可解析、字段级比对通过，无需批量修改。唯一建议是删除 #19 标题中的 "(TTM-Net)"。完成该处修改后，参考文献清单可直接用于投稿。
