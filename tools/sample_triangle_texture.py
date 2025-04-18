# sample_triangle_texture.py
"""
180x32画像に、各三角形ごとに異なる色＋番号を描画したサンプル画像を生成します。
UVマッピングのように、三角形ごとに色分け＋中央に番号を描画。
"""
# --- 必要なライブラリのインポート ---
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# --- 定数定義 ---
# 画像サイズ
IMAGE_W = 180
IMAGE_H = 32
# 三角形グリッドサイズ
TRIANGLE_GRID_W = 5
TRIANGLE_GRID_H = 16

# --- カラーシステムのインポート ---
import colorsys

# --- 画像生成 ---
# 画像を作成
img = Image.new("RGB", (IMAGE_W, IMAGE_H), (255,255,255))
draw = ImageDraw.Draw(img)

# --- フォントの設定 ---
# フォント（環境依存。なければデフォルト）
try:
    font = ImageFont.truetype("arial.ttf", 10)
except:
    font = ImageFont.load_default()

# --- 三角形の描画 ---
for gx in range(TRIANGLE_GRID_W):
    for gy in range(TRIANGLE_GRID_H):
        idx = gx * TRIANGLE_GRID_H + gy
        # HSV色相でカラフルに
        h = idx / (TRIANGLE_GRID_W * TRIANGLE_GRID_H)
        r, g, b = [int(255*x) for x in colorsys.hsv_to_rgb(h, 0.8, 1.0)]
        color = (r, g, b)
        # 三角形領域の座標
        x0 = gx * (IMAGE_W // TRIANGLE_GRID_W)
        y0 = gy * (IMAGE_H // TRIANGLE_GRID_H)
        x1 = (gx+1) * (IMAGE_W // TRIANGLE_GRID_W)
        y1 = (gy+1) * (IMAGE_H // TRIANGLE_GRID_H)
        # 正三角形の頂点座標（正確な正三角形）
        tri_w = x1-x0
        tri_h = int((tri_w/2) * 1.73205)  # 正三角形の高さ
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        # 頂点（上、左下、右下）
        tri = [
            (cx, cy - tri_h//2),
            (cx - tri_w//2, cy + tri_h//2),
            (cx + tri_w//2, cy + tri_h//2),
        ]
        # 三角形の描画
        draw.polygon(tri, fill=color)
        # 番号を白で縁取り黒文字（大きめ）
        num = f"{gx},{gy}"
        bbox = font.getbbox(num)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        for dx in [-2,-1,0,1,2]:
            for dy in [-2,-1,0,1,2]:
                if dx or dy:
                    draw.text((cx-tw//2+dx, cy-th//2+dy), num, fill=(255,255,255), font=font)
        draw.text((cx-tw//2, cy-th//2), num, fill=(0,0,0), font=font)

img.save("triangle_sample.png")
print("triangle_sample.png を生成しました（カラフルUVグリッド風）")
