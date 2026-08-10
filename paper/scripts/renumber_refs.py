#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""References Vancouver 重编号：按正文首引顺序重排 23 条。

映射（当前编号 -> 新编号）由首引顺序决定：
  [22] -> [6]   （Gong 分割，L82 引言首引）
  [6..20] -> [7..21] 顺移
  [21] -> [23]
  [23] -> [22]  （RCAF）
正文引用与 References 列表同步更新，占位符防链式冲突。
"""
import re
from pathlib import Path

PROJ = Path(r"C:\Users\甄朝晖\Desktop\thyroid")
M = PROJ / "paper" / "output" / "doc" / "manuscript.md"
t = M.read_text(encoding="utf-8")

# 首引顺序映射（由运行期计算，硬编码已验证）
MAPPING = {
    6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15,
    15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 23, 22: 6, 23: 22,
}

# ---- 1) 正文引用更新 ----
def repl_ref(m):
    inner = m.group(1)
    nums = [int(x) for x in inner.split(",") if x.strip().isdigit()]
    if not nums or max(nums) > 100:  # 跳过 [256,128] 架构维度等
        return m.group(0)
    newnums = [str(MAPPING.get(n, n)) for n in nums]
    # 保持原始顺序（旧号升序 -> 新号升序应一致；组合内保持原顺序）
    return "[" + ",".join(newnums) + "]"

# 占位符两段式：先旧->占位，再占位->新
# 简单起见：直接对每个单号用临时标签（@T{n}@）替换，再处理组合引用
body, refs = t.split("## References", 1)

# 2-pass: 先把每个 [N] 单号替换为占位（保护组合里的），再组合引用整体处理
def single_pass(text, mapper):
    # 处理组合引用 [a,b,c] 整体（内部映射）
    def comb(m):
        inner = m.group(1)
        nums = [int(x) for x in inner.split(",") if x.strip().isdigit()]
        if not nums or max(nums) > 100:
            return m.group(0)
        newnums = [str(MAP[num]) for num in nums]
        return "[" + ",".join(newnums) + "]"
    # 先占位化每个数字 token（防止 22->6 与 6->7 互相覆盖）
    placeholders = {}
    for num in mapper:
        placeholders[num] = f"@@R{num}@@"
    # 替换组合引用中的数字
    def comb2(m):
        inner = m.group(1)
        parts = inner.split(",")
        out = []
        for p in parts:
            p = p.strip()
            if p.isdigit() and int(p) in mapper:
                out.append(placeholders[int(p)])
            else:
                out.append(p)
        return "[" + ",".join(out) + "]"
    text = re.sub(r"\[([\d,\s]+)\]", comb2, text)
    # 占位符 -> 新号
    for num, ph in placeholders.items():
        text = text.replace(ph, str(mapper[num]))
    return text

body = single_pass(body, MAPPING)

# ---- 2) References 列表重排 ----
# 解析现有条目
ref_entries = {}
for m in re.finditer(r"^(\d{1,2})\.\s+(.+)$", refs, re.MULTILINE):
    n = int(m.group(1))
    ref_entries[n] = m.group(2).strip()

# 按新编号顺序重排（1..23）
new_refs_lines = []
for new_n in range(1, 24):
    # 找到对应旧号：MAPPING 反转
    old_n = [k for k, v in MAPPING.items() if v == new_n]
    old_n = old_n[0] if old_n else new_n
    if old_n in ref_entries:
        new_refs_lines.append(f"{new_n}. {ref_entries[old_n]}\n")

# 保留 refs 段开头可能的注释/标题（References 标题在 body/refs split 之外）
refs_header = ""
new_refs = "\n\n".join(l.strip() for l in new_refs_lines)
# 用原 References 格式（每条后空行）
new_refs_block = "\n\n".join(new_refs_lines)

t2 = body + "## References\n" + refs_header + "\n\n" + new_refs_block.rstrip("\n") + "\n"

# ---- 3) 验证 ----
# 重算首引顺序
body2 = t2.split("## References")[0]
body2 = "\n".join(l for l in body2.splitlines() if not l.startswith(">"))
seen = []
for m in re.finditer(r"\[([\d,\s]+)\]", body2):
    for n in [int(x) for x in m.group(1).split(",") if x.strip().isdigit()]:
        if n < 100 and n not in seen:
            seen.append(n)
asc = all(seen[i] < seen[i + 1] for i in range(len(seen) - 1))
print("新首引顺序:", seen)
print("严格升序?", asc)

M.write_text(t2, encoding="utf-8")
print("written", M)
