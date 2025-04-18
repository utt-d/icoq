# led_emulator.py
"""
180x32 RGB画像データ（C配列やバイナリ）を、物理LED配置に従ってLEDパネルとしてエミュレーション表示するツール。

使い方:
  python led_emulator.py image.h
または
  python led_emulator.py image.rgb

- image.h: 変換済みCヘッダファイル（const uint8_t my_image[] = {...};）
- image.rgb: 180x32x3の生バイナリファイル

表示にはmatplotlibを利用します。
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import re

IMAGE_W = 180
IMAGE_H = 32

# --- 画像データの読み込み関数群 ---
# C配列形式ヘッダファイルから画像データを読み込む

def load_c_array_image(header_path):
    with open(header_path, 'r') as f:
        text = f.read()
    # 配列部分を抽出
    arr = re.findall(r'{([^}]*)}', text, re.DOTALL)
    if not arr:
        raise ValueError('No array found in header file')
    nums = [int(x) for x in arr[0].replace('\n', '').split(',') if x.strip().isdigit()]
    if len(nums) != IMAGE_W * IMAGE_H * 3:
        raise ValueError(f'Array size mismatch: {len(nums)} != {IMAGE_W*IMAGE_H*3}')
    img = np.array(nums, dtype=np.uint8).reshape((IMAGE_H, IMAGE_W, 3))
    return img

# 生のRGBバイナリファイルから画像データを読み込む

def load_raw_rgb_image(rgb_path):
    arr = np.fromfile(rgb_path, dtype=np.uint8)
    if arr.size != IMAGE_W*IMAGE_H*3:
        raise ValueError(f'Raw size mismatch: {arr.size} != {IMAGE_W*IMAGE_H*3}')
    img = arr.reshape((IMAGE_H, IMAGE_W, 3))
    return img

# --- 物理LED配置・三角形パネルのパラメータ定義 ---
NUM_PORTS = 5  # LEDポート数
TRIANGLES_PER_PORT = 16  # 1ポートあたり三角形数
LEDS_PER_TRIANGLE = 36  # 1三角形あたりLED数
TRIANGLE_GRID_W = 5  # サンプル画像内の三角形グリッド横数
TRIANGLE_GRID_H = 16 # サンプル画像内の三角形グリッド縦数

# --- CPL-triangle_led5.csvのtopレイヤー(U1〜U36)物理座標を正規化してローカル座標配列を生成 ---
def get_triangle_local_coords_from_cpl(csv_path):
    import numpy as np
    import csv
    # U1〜U36のtopレイヤー座標を抽出
    led_xyz = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Layer'].strip() == 'top' and row['Designator'].startswith('U'):
                x = float(row['Mid X'])
                y = float(row['Mid Y'])
                led_xyz.append((x, y))
    if len(led_xyz) != 36:
        raise ValueError('topレイヤーU1〜U36が36個ありません')
    # 三角形の重心・一辺長・基準角度を取得
    pts = np.array(led_xyz)
    cx = np.mean(pts[:,0])
    cy = np.mean(pts[:,1])
    # 一番遠い3点で一辺長を推定
    from itertools import combinations
    maxlen = 0
    for a, b in combinations(pts, 2):
        d = np.linalg.norm(np.array(a)-np.array(b))
        if d > maxlen:
            maxlen = d
    side = maxlen
    # ローカル正規化座標(u,v)生成
    local_coords = []
    for x, y in led_xyz:
        # 三角形を重心(cx,cy)中心、1辺=1、上向きに正規化
        dx = x - cx
        dy = y - cy
        # x軸を水平方向、y軸を上向きと仮定
        u = 0.5 + dx / side
        v = 0.5 + dy / (side * np.sqrt(3)/2)
        local_coords.append([u, v])
    return np.array(local_coords)

triangle_local_coords = get_triangle_local_coords_from_cpl('tools/CPL-triangle_led5.csv')

# 画像内で三角形ごとのエリアからLEDに対応するピクセルを取得する関数

def get_image_pixel_for_led(segment, tri, led):
    # サンプル画像の三角形エリア中央寄せ・ローカル座標(u,v)を三角形エリア内に正規化
    tri_img_w = IMAGE_W // TRIANGLE_GRID_W
    tri_img_h = IMAGE_H // TRIANGLE_GRID_H
    x0 = segment * tri_img_w
    y0 = tri * tri_img_h
    x1 = (segment + 1) * tri_img_w
    y1 = (tri + 1) * tri_img_h
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    r_tri = min((x1-x0), (y1-y0))//2 - 1
    # triangle_local_coords[led]からu,vを取得
    u, v = triangle_local_coords[led]
    px = cx + (u-0.5)*r_tri*1.732  # 1.732 ≒ √3
    py = cy + (1-v)*r_tri
    img_x = int(np.clip(px, x0, x1-1))
    img_y = int(np.clip(py, y0, y1-1))
    return img_x, img_y

import csv

def load_physical_led_positions(csv_path):
    led_positions = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Layer'].strip() == 'top' and row['Designator'].startswith('U'):
                cx = float(row['Mid X'])
                cy = float(row['Mid Y'])
                angle = float(row['Rotation'])
                led_positions.append({'cx': cx, 'cy': cy, 'angle': angle})
    return led_positions

def get_led_abs_pos(triangle, led, led_positions, triangle_local_coords):
    center = led_positions[triangle]
    u, v = triangle_local_coords[led]
    # 正三角形のローカル座標を一辺1で原点中心に変換
    lx = (u - 0.5) * 1.0 * np.sqrt(3) / 2
    ly = (v - 1/3) * 1.0
    # 回転
    theta = np.deg2rad(center['angle'])
    rx = lx * np.cos(theta) - ly * np.sin(theta)
    ry = lx * np.sin(theta) + ly * np.cos(theta)
    # 平行移動
    return center['cx'] + rx, center['cy'] + ry

def show_led_physical_emulation(img, mode="physical"):
    led_xs = []
    led_ys = []
    led_colors = []
    # --- 物理LEDインデックス: 全ポート分（5x16x36=2880） ---
    triangle_led_physical_index = []
    for seg in range(NUM_PORTS):
        port = []
        base = seg * TRIANGLES_PER_PORT * LEDS_PER_TRIANGLE
        for tri in range(TRIANGLES_PER_PORT):
            tri_leds = [base + tri*LEDS_PER_TRIANGLE + i + 1 for i in range(LEDS_PER_TRIANGLE)]
            port.append(tri_leds)
        triangle_led_physical_index.append(port)
    # --- 物理LED番号→三角形・ローカルLED番号への逆変換マップ ---
    led_map = dict()
    for seg in range(len(triangle_led_physical_index)):
        for tri in range(len(triangle_led_physical_index[seg])):
            for led in range(len(triangle_led_physical_index[seg][tri])):
                idx = triangle_led_physical_index[seg][tri][led] - 1
                led_map[idx] = (seg, tri, led)
    if mode == "2d":
        # --- 物理寸法に基づく三角形レイアウトをCSVから読込 ---
        triangle_layout = []
        with open('tools/triangle_piece_layout_mm.csv', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                triangle_layout.append({
                    'cx': float(row['x']),
                    'cy': float(row['y']),
                    'angle': float(row['angle'])
                })
        # --- デバッグ出力: 配置リストと段ごとの三角形番号 ---
        print("=== Triangle Layout Debug Info ===")
        for idx, tri in enumerate(triangle_layout):
            print(f"Triangle {idx+1}: center=({tri['cx']:.2f},{tri['cy']:.2f}), angle={tri['angle']}")
        row_start = 0
        for row, orientations in enumerate([
            [1],
            [2, 2, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [1, 2, 1],
            [1]
        ]):
            n = len(orientations)
            print(f"Row {row+1}: {[i+1 for i in range(row_start, row_start+n)]}")
            row_start += n

        # --- CSVから単一三角形（基板1枚分）のLED配置を取得 ---
        led_positions = load_physical_led_positions('tools/CPL-triangle_led5.csv')
        # --- 補正: LED座標を三角形中心基準に変換 ---
        centroid_x = sum([p['cx'] for p in led_positions]) / len(led_positions)
        centroid_y = sum([p['cy'] for p in led_positions]) / len(led_positions)
        for p in led_positions:
            p['cx'] -= centroid_x
            p['cy'] -= centroid_y
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1,1,1)
        for t_idx, tri in enumerate(triangle_layout):
            for led in range(LEDS_PER_TRIANGLE):
                # 基板上LEDローカル座標（U1基準）を取得
                base = led_positions[led]
                # 三角形全体配置の中心・回転を適用
                theta = np.deg2rad(tri['angle'])
                rx = (base['cx']) * np.cos(theta) - (base['cy']) * np.sin(theta)
                ry = (base['cx']) * np.sin(theta) + (base['cy']) * np.cos(theta)
                px = tri['cx'] + rx
                py = tri['cy'] + ry
                # サンプリング座標（画像）
                seg = t_idx // TRIANGLES_PER_PORT
                tri_local = t_idx % TRIANGLES_PER_PORT
                img_x, img_y = get_image_pixel_for_led(seg, tri_local, led)
                color = img[img_y, img_x] / 255.0
                # --- ガンマ補正でグラデーションを急峻に ---
                color = color ** 2.0
                if t_idx == 14 and led < 5:  # 15番目（三角形15）の先頭5LEDだけprint
                    print(f"Triangle 15 LED{led+1}: color={color}, img_x={img_x}, img_y={img_y}")
                led_xs.append(px)
                led_ys.append(py)
                led_colors.append(color)
            # --- 三角形番号を中心に描画 ---
            ax.text(tri['cx'], tri['cy'], str(t_idx+1), fontsize=16, color='red', ha='center', va='center', weight='bold')
        # --- LED点を小さく描画（s=10, alpha=0.9） ---
        # ax.scatter(led_xs, led_ys, c=led_colors, s=10, alpha=0.9, edgecolors='none')

        ax.scatter(led_xs, led_ys, c=led_colors, s=40, edgecolors='k')
        ax.set_aspect('equal')
        ax.axis('off')
        plt.title('M5Capsule 2D Triangular Grid Emulator (Icosahedron Layout)')
        plt.tight_layout()
        plt.show()
        return
    if mode == "3d":
        # 3D正二十面体配置
        from mpl_toolkits.mplot3d import Axes3D
        from math import sqrt
        phi = (1 + sqrt(5)) / 2
        vertices = [
            (-1,  phi, 0), (1,  phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1,  phi), (0, 1,  phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
        ]
        faces = [
            (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
            (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
            (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
            (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)
        ]
        led_xs, led_ys, led_zs, led_colors = [], [], [], []
        # --- Icosahedron（正二十面体）の20面それぞれにLEDを割り当てて3Dプロット ---
        for fidx, (i0,i1,i2) in enumerate(faces):
            # 各面（三角形）の3頂点座標を取得
            v0, v1, v2 = np.array(vertices[i0]), np.array(vertices[i1]), np.array(vertices[i2])
            for led in range(LEDS_PER_TRIANGLE):
                # 各LEDの三角形内での重心座標（u, v, w）を取得
                u, v = triangle_local_coords[led]
                w = 1 - u - v
                # 3頂点の重み付き和でLEDの3D座標を計算（面内の位置をIcosahedron面上に写像）
                pos = v0 * w + v1 * u + v2 * v
                # 球面上に正規化（半径1の球体に投影）
                pos = pos / np.linalg.norm(pos)
                # 3D座標リストに追加
                led_xs.append(pos[0])
                led_ys.append(pos[1])
                led_zs.append(pos[2])
                # 画像上の該当三角形エリアに対応するグリッド座標を計算
                gx = fidx % TRIANGLE_GRID_W  # 横方向のグリッド番号
                gy = fidx // TRIANGLE_GRID_W # 縦方向のグリッド番号
                img_x, img_y = get_image_pixel_for_led(gx, gy, led)
                # 画像から色を取得し、0-1に正規化
                color = img[img_y, img_x] / 255.0
                # ガンマ補正で色のコントラストを強調
                color = color ** 2.0  # 2Dと同じガンマ補正
                # 色リストに追加
                led_colors.append(color)
        # --- 3D散布図としてLEDを描画 ---
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(led_xs, led_ys, led_zs, c=led_colors, s=40, edgecolors='k')
        set_axes_equal(ax)
        ax.set_axis_off()
        ax.set_title('Icosahedron (20 faces) LED Layout Emulator')
        plt.tight_layout()
        plt.show()
        return
    # 従来の物理配置
    for seg in range(NUM_PORTS):
        for tri in range(TRIANGLES_PER_PORT):
            for led in range(LEDS_PER_TRIANGLE):
                img_x, img_y = get_image_pixel_for_led(seg, tri, led)
                color = img[img_y, img_x] / 255.0
                px = seg + triangle_local_coords[led][0]/1.2
                py = tri + (1.0-triangle_local_coords[led][1])/1.2
                led_xs.append(px)
                led_ys.append(py)
                led_colors.append(color)
    plt.figure(figsize=(10,12))
    plt.scatter(led_xs, led_ys, c=led_colors, s=40, edgecolors='k')
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.title('M5Capsule Physical LED Layout Emulator')
    plt.tight_layout()
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('imagefile')
    parser.add_argument('--mode', choices=['physical', '2d', '3d'], default='physical')
    args = parser.parse_args()
    path = args.imagefile
    if path.endswith('.h'):
        img = load_c_array_image(path)
    else:
        img = load_raw_rgb_image(path)
    show_led_physical_emulation(img, mode=args.mode)

if __name__ == '__main__':
    main()
