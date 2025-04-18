import math
import csv

# --- 基板1枚分のLED配置（CPL-triangle_led5.csv）を基準に、
# 設計図画像通りの三角形中心座標・回転角リストを物理mm座標で生成 ---

TRI_SIDE = 17.425
TRI_HEIGHT = TRI_SIDE * math.sqrt(3) / 2

# U1~U36のLED配置から三角形基板の中心・回転0度方向を算出
# U1~U36の重心を三角形中心、U1→U10の方向を0度とする

# --- 三角形基板の物理配置リスト ---
# triangle_layout: 各三角形の中心座標(x, y)と回転角(angle[deg])を個別指定
# 必要な数だけリストに追加してください
triangle_layout = [
    # 例: (中心x座標, 中心y座標, 回転角度[deg])
    (0, 0, -120.0),
    (0, -10.0603, 180.0),
    (8.7125, -15.0905, -120.0),
    (8.7125, -25.1508, 180.0),
    (17.425, -30.1810, -120.0),
    (17.425, -40.2413, 180.0),
    (26.1375, -45.2715, -120.0),
    (26.1375, -55.3318, 180.0),
    (17.425, -70.4223, 60.0),
    (17.425, -60.3620, 0.0),
    (8.7125, -55.3318, 60.0),
    (8.7125, -45.2715, 0.0),
    (0, -40.2413, 60.0),
    (0, -30.1810, 0.0),
    (-8.7125, -25.1508, 60.0),
    (-8.7125, -15.0905, 0.0),
]

def to_absolute(xrel, yrel):
    cx, cy = TRI_CENTER
    return cx + xrel, cy + yrel

# --- メイン処理 ---
# triangle_layoutリストの内容をCSVに出力
if __name__ == "__main__":
    with open("tools/triangle_piece_layout_mm.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["triangle_id", "x", "y", "angle"])
        for tid, (x, y, ang) in enumerate(triangle_layout, 1):
            writer.writerow([tid, f"{x:.3f}", f"{y:.3f}", f"{ang:.1f}"])
    print("triangle_piece_layout_mm.csv written.")
