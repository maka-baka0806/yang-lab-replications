"""绘图统一配置：解决 matplotlib 中文显示为方块的问题。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# macOS 自带的中文字体，按优先级排列
CJK_FONTS = [
    "Arial Unicode MS",   # macOS 自带，覆盖最全
    "PingFang SC",        # 苹方
    "Hiragino Sans GB",   # 冬青黑体
    "Heiti SC",           # 黑体
    "Songti SC",          # 宋体
    "STHeiti",
    "DejaVu Sans",        # 兜底（无中文）
]


def setup_cjk_font() -> str:
    """设置 matplotlib 中文字体，返回实际使用的字体名。"""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = "DejaVu Sans"
    for name in CJK_FONTS:
        if name in available:
            chosen = name
            break

    plt.rcParams["font.sans-serif"] = [chosen] + CJK_FONTS
    plt.rcParams["axes.unicode_minus"] = False   # 负号正常显示
    plt.rcParams["figure.dpi"] = 110
    return chosen


def new_fig(width: float = 5.0, height: float = 5.0):
    """创建一个已配置好字体的图。"""
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax
