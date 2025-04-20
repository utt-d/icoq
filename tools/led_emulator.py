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
import csv

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
    global triangle_local_coords_side
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
    # 一辺長は固定値（17.425mm）
    side = 17.425
    triangle_local_coords_side = side
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
    global csv
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
        # --- グローバル回転・反転補正パラメータ ---
        theta_global = 0  # [deg] 必要に応じて調整
        flip_y = False    # TrueでY軸反転
        # --- 物理寸法に基づく三角形レイアウトをCSVから読込 ---
        triangle_layout = []
        with open('tools/triangle_piece_layout_mm.csv', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row['x'])
                y = float(row['y'])
                # グローバル回転
                theta = np.deg2rad(theta_global)
                x_rot = x * np.cos(theta) - y * np.sin(theta)
                y_rot = x * np.sin(theta) + y * np.cos(theta)
                # Y反転
                if flip_y:
                    y_rot = -y_rot
                triangle_layout.append({
                    'x': x_rot,
                    'y': y_rot,
                    'angle': float(row['angle']) + theta_global * (1 if not flip_y else -1),
                    'id': int(row['triangle_id']) if 'triangle_id' in row else None
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
                px = tri['x'] + rx
                py = tri['y'] + ry
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
            ax.text(tri['x'], tri['y'], str(t_idx+1), fontsize=16, color='red', ha='center', va='center', weight='bold')
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
        # Standard icosahedron vertices centered at the origin
        vertices = [
            (-1,  phi,  0),  # 0
            ( 1,  phi,  0),  # 1
            (-1, -phi,  0),  # 2
            ( 1, -phi,  0),  # 3
            ( 0, -1,  phi),  # 4
            ( 0,  1,  phi),  # 5
            ( 0, -1, -phi),  # 6
            ( 0,  1, -phi),  # 7
            ( phi,  0, -1),  # 8
            ( phi,  0,  1),  # 9
            (-phi,  0, -1),  #10
            (-phi,  0,  1)   #11
        ]
        # Standard icosahedron face definitions (each tuple is a triangle of vertex indices)
        faces = [
            (0, 11, 5), (5, 11, 4), (4, 9, 5), (3, 9, 4),
            (0, 5, 1), (1, 5, 9), (9, 8, 1), (3, 8, 9),
            (0, 1, 7), (7, 1, 8), (8, 6, 7), (3, 6, 8),
            (0, 7, 10), (10, 7, 6), (6, 2, 10), (3, 2, 6),
            (0, 10, 11), (11, 10, 2), (2, 4, 11), (3, 4, 2),
        ]

        led_xs, led_ys, led_zs, led_colors = [], [], [], []
        # --- 各面の重心に面番号を表示 ---
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        for fidx, (i0, i1, i2) in enumerate(faces):
            v0, v1, v2 = np.array(vertices[i0]), np.array(vertices[i1]), np.array(vertices[i2])
            centroid = (v0 + v1 + v2) / 3
            ax.text(centroid[0], centroid[1], centroid[2], str(fidx), color='blue', fontsize=12, ha='center', va='center')
            # Optionally: draw the triangle edges for clarity
            tri = np.array([v0, v1, v2, v0])
            ax.plot(tri[:,0], tri[:,1], tri[:,2], color='gray', alpha=0.5)
        # --- triangle_layoutをCSVから読み込む（2D/3D共通で必要） ---
        triangle_layout = []
        with open('tools/triangle_piece_layout_mm.csv', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get('triangle_id') or not row.get('x') or not row.get('y') or not row.get('angle'):
                    continue  # 空行や不正行をスキップ
                triangle_layout.append({
                    'triangle_id': int(row['triangle_id']),
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'angle': float(row['angle'])
                })
        # --- Icosahedron（正二十面体）の20面すべてに1ピース分（16枚）の物理配置を5ピース分割り当てて3Dプロット ---
        N_PIECES = 5
        TRI_PER_PIECE = 16 // N_PIECES  # 1ピースあたりの三角形枚数（ここでは4）
        TRI_PER_FACE = 4  # 1面あたり4枚の基板を割り当て
        # --- Icosahedronの頂点0を含む5面を「極」とし、そこから放射状に5ピースを反時計回りに展開 ---
        # 1. 頂点0を含む5面を抽出
        pole_faces = [fidx for fidx, (i0,i1,i2) in enumerate(faces) if 0 in (i0,i1,i2)]
        # 2. 5面の重心をXY平面に投影し、反時計回りにソート
        def face_centroid(face):
            v = [np.array(vertices[i]) for i in face]
            return np.mean(v, axis=0)
        pole_face_centroids = [face_centroid(faces[fidx]) for fidx in pole_faces]
        pole_angles = [np.arctan2(c[1], c[0]) for c in pole_face_centroids]
        pole_faces_sorted = [f for _, f in sorted(zip(pole_angles, pole_faces))]

        # --- 面・サブ面とtriangle_idの割当を外部CSVから読み込む ---
        import os
        face_sub_to_id = {}
        id_to_face_sub = {}
        assign_csv = 'tools/face_sub_assignment.csv'
        if os.path.exists(assign_csv):
            import csv
            with open(assign_csv, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    face = int(row['face'])
                    sub = int(row['sub'])
                    tid = int(row['triangle_id'])
                    face_sub_to_id[(face, sub)] = tid
                    id_to_face_sub[tid] = (face, sub)
        else:
            print(f"[WARN] {assign_csv} not found. Using default assignment.")
            idx = 0
            for face in range(4):
                for sub in range(4):
                    tid = idx + 1
                    face_sub_to_id[(face, sub)] = tid
                    id_to_face_sub[tid] = (face, sub)
                    idx += 1

        assigned = set()  # 割り当て済み小三角形
        all_subtris = []  # ピースごとに放射状に展開した小三角形リスト
        # Icosahedron全体の小三角形を面ごとに用意
        face_subtris_dict = {}
        for fidx, (i0,i1,i2) in enumerate(faces):
            v0, v1, v2 = np.array(vertices[i0]), np.array(vertices[i1]), np.array(vertices[i2])
            m01 = (v0 + v1) / 2
            m12 = (v1 + v2) / 2
            m20 = (v2 + v0) / 2
            face_subtris = [
                [v0, m01, m20],
                [m01, v1, m12],
                [m20, m12, v2],
                [m01, m12, m20]
            ]
            face_subtris_dict[fidx] = [np.stack(t) for t in face_subtris]
        # 4. triangle_piece_layout_mm.csvの各基板IDにIcosahedron上の(面番号, sub番号)をマッピングするテーブルを仮作成
        # 例: ID 1～16を面0,1,2,3・sub0～3に順に割り当てる（仮）
        id_to_face_sub = {}
        idx = 0
        for face in range(4):
            for sub in range(4):
                id_to_face_sub[idx+1] = (face, sub)  # IDは1始まり
                idx += 1
        # --- triangle_piece_layout_mm.csvの順で物理基板を割り当て ---
        # --- face_sub_assignment.csvの割当順でサブ三角形と基板を対応付け ---
        all_subtris = []
        triangle_layout_dict = {int(t['triangle_id']): t for t in triangle_layout}
        for row in face_sub_to_id:
            triangle_id = face_sub_to_id[row]
            if triangle_id not in triangle_layout_dict:
                continue
            tri_layout = triangle_layout_dict[triangle_id]
            fidx, subidx = row  # (face, sub)
            subtri3d = face_subtris_dict[fidx][subidx]
            all_subtris.append((fidx, subidx, subtri3d, tri_layout, triangle_id))
        # --- 各物理基板をIcosahedron上のサブ三角形に割り当て、LEDを3D座標へ配置 ---
        # --- 2D物理配置全体の重心を計算し、中心化補正 ---
        for subtris_idx, (fidx, subidx, subtri3d, tri_layout, triangle_id) in enumerate(all_subtris):
            angle = np.deg2rad(tri_layout['angle'])
            # 物理基板上の正三角形のローカル座標系（三頂点）
            # --- tri2dを物理基板実寸スケールで構成 ---
            # CPL-triangle_led5.csvからの1辺長を取得
            cpl_pts = np.array([triangle_local_coords[0], triangle_local_coords[9], triangle_local_coords[18]])
            # ただしtriangle_local_coordsは正規化(u,v)なので、物理LEDの絶対mm座標も取得する必要あり
            # ここではget_triangle_local_coords_from_cpl内でsideを返すようにして取得
            # 既に取得済みのsideをグローバル変数で使う
            side = triangle_local_coords_side
            # 物理基板の正三角形頂点（1辺=side, 上向き）
            # --- LED群物理座標（U1〜U36）の重心を計算 ---
            led_phys_xy = np.array([triangle_local_coords[led] for led in range(LEDS_PER_TRIANGLE)])
            # triangle_local_coordsは(u,v)正規化なので、物理mm座標に変換
            # ここで「正三角形の重心=LED群重心」となるように補正
            # 物理三角形の重心（基板中心）を原点とする
            led_phys_xy_mm = []
            for u, v in triangle_local_coords:
                # 正三角形座標系(u,v)→物理mm座標
                x = (u - 0.5) * side
                y = (v - 0.5) * side * np.sqrt(3)/2
                led_phys_xy_mm.append([x, y])
            led_phys_xy_mm = np.array(led_phys_xy_mm)
            led_centroid = np.mean(led_phys_xy_mm, axis=0)
            # tri2d: LED群重心=原点
            # 2D三角形3頂点（上・左下・右下）
            tri2d_pts = np.array([
                [0, side/np.sqrt(3)],
                [-0.5*side, -0.5*side/np.sqrt(3)],
                [0.5*side, -0.5*side/np.sqrt(3)]
            ])
            # 3D三角形3頂点（subtri3dのうち、Zが最大=上、次に左下/右下を距離で決定）
            subtri3d_pts = np.zeros_like(subtri3d)
            z_vals = subtri3d[:,2]
            idx_top = np.argmax(z_vals)
            idxs_rest = [i for i in range(3) if i != idx_top]
            # 2Dで左下/右下を判定
            xy_rest = subtri3d[idxs_rest,:2]
            # 原点から見て左下・右下を分ける
            if xy_rest[0,0] < xy_rest[1,0]:
                idx_left = idxs_rest[0]
                idx_right = idxs_rest[1]
            else:
                idx_left = idxs_rest[1]
                idx_right = idxs_rest[0]
            subtri3d_pts[0] = subtri3d[idx_top]
            subtri3d_pts[1] = subtri3d[idx_left]
            subtri3d_pts[2] = subtri3d[idx_right]
            # --- デバッグ: 頂点対応print ---
            print(f"[Triangle ID={tri_layout.get('id', tri_layout.get('triangle_id', '?'))}] 2D頂点={tri2d_pts.tolist()} 3D頂点={subtri3d_pts.tolist()}")
            # --- アフィン変換計算（3点対応）---
            # アフィン変換行列計算
            A, _, _, _ = np.linalg.lstsq(
                np.hstack([tri2d_pts, np.ones((3,1))]),
                subtri3d_pts,
                rcond=None
            )
            # --- 各LEDを変換 ---
            led_xyz = []
            for xy in led_phys_xy_mm:
                pt2d = np.array([xy[0], xy[1], 1.0])
                pt3d = pt2d @ A
                led_xyz.append(pt3d)
            led_xyz = np.array(led_xyz)
            # tri2d_pts, subtri3d_pts, led_xyzはこの後で使う
            # --- 以降はled_xyzのみを3D LED配置として利用 ---
            # --- 各LEDについて ---
            for led in range(LEDS_PER_TRIANGLE):
                # 3D LED座標はled_xyzから取得
                led_xs.append(led_xyz[led, 0])
                led_ys.append(led_xyz[led, 1])
                led_zs.append(led_xyz[led, 2])
                # --- サンプル画像内の対応ピクセル座標を取得 ---
                seg = (subtris_idx // TRIANGLES_PER_PORT) % NUM_PORTS
                tri_local = subtris_idx % TRIANGLES_PER_PORT
                img_x, img_y = get_image_pixel_for_led(seg, tri_local, led)
                # --- ピクセルから色情報を取得・ガンマ補正 ---
                if 0 <= img_y < img.shape[0] and 0 <= img_x < img.shape[1]:
                    color = img[img_y, img_x] / 255.0
                else:
                    color = np.array([0.0, 0.0, 0.0])
                color = color ** 2.0
                led_colors.append(color)

        # --- 3D散布図としてLEDを描画 ---
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        # Z値でalphaを調整（カメラ視点z>0を手前とする）
        zs = np.array(led_zs)
        alphas = 0.15 + 0.85 * (zs - zs.min())/(zs.max() - zs.min() + 1e-8)  # 手前ほどalpha=1, 奥ほどalpha=0.15
        led_rgba = [np.append(c, a) for c, a in zip(led_colors, alphas)]
        # scatterでRGBA配列を直接渡す
        ax.scatter(led_xs, led_ys, led_zs, c=led_rgba, s=18, edgecolors='k', linewidths=0.5, depthshade=True)

        # --- 各サブ三角形の重心位置に面番号・基板番号ラベルを表示 ---
        for fidx, subidx, subtri3d, tri_layout, triangle_id in all_subtris:
            # サブ三角形の重心座標を計算
            centroid = np.mean(subtri3d, axis=0)
            label = f"F{fidx}-S{subidx}\nID{triangle_id}"
            ax.text(centroid[0], centroid[1], centroid[2], label, color='black', fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

        # --- Icosahedronの面エッジをwireframeで描画 ---
        for (i0, i1, i2) in faces:
            v0, v1, v2 = np.array(vertices[i0]), np.array(vertices[i1]), np.array(vertices[i2])
            ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], color='gray', linewidth=0.8)
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], color='gray', linewidth=0.8)
            ax.plot([v2[0], v0[0]], [v2[1], v0[1]], [v2[2], v0[2]], color='gray', linewidth=0.8)
        set_axes_equal(ax)
        try:
            ax.set_box_aspect([1,1,1])  # matplotlib>=3.4
        except Exception:
            pass
        ax.set_axis_off()
        ax.set_title('Icosahedron (20 faces) LED Layout Emulator')
        plt.tight_layout()
        plt.show()
        return

def set_axes_equal(ax):
    '''3Dグラフの軸スケールを等しくするユーティリティ'''
    import numpy as np
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

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
