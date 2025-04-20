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
    # U1〜U36のtopレイヤー座標とDesignator番号を抽出
    # --- U1〜U36をDesignator番号順にソートして抽出 ---
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [
            row for row in reader
            if row['Layer'].strip() == 'top' and row['Designator'].startswith('U')
        ]
        rows.sort(key=lambda r: int(r['Designator'][1:]))
        led_xyz = [(float(row['Mid X']), float(row['Mid Y'])) for row in rows]
        led_designators = [row['Designator'] for row in rows]
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
        dx = x - cx
        dy = y - cy
        u = 0.5 - dx / side  # u成分を反転
        v = 0.5 + dy / (side * np.sqrt(3)/2)
        local_coords.append([u, v])
    return np.array(local_coords), led_designators


triangle_local_coords, triangle_led_designators = get_triangle_local_coords_from_cpl('tools/CPL-triangle_led5.csv')


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

def show_led_physical_emulation(img, mode="physical", label_orient=False):
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

        # --- サブ三角形ごとにアフィン変換行列を作成 ---
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
        # triangle_id→3Dアフィン変換行列
        triangle_id_to_affine = {}
        side = triangle_local_coords_side
        for subtris_idx, (fidx, subidx, subtri3d, tri_layout, triangle_id) in enumerate(all_subtris):
            tri2d_pts = np.array([
                [0, side/np.sqrt(3)],
                [-0.5*side, -0.5*side/np.sqrt(3)],
                [0.5*side, -0.5*side/np.sqrt(3)]
            ])
            subtri3d_pts = np.zeros_like(subtri3d)
            z_vals = subtri3d[:,2]
            idx_top = np.argmax(z_vals)
            idxs_rest = [i for i in range(3) if i != idx_top]
            xy_rest = subtri3d[idxs_rest,:2]
            if xy_rest[0,0] < xy_rest[1,0]:
                idx_left = idxs_rest[0]
                idx_right = idxs_rest[1]
            else:
                idx_left = idxs_rest[1]
                idx_right = idxs_rest[0]
            subtri3d_pts[0] = subtri3d[idx_top]
            subtri3d_pts[1] = subtri3d[idx_left]
            subtri3d_pts[2] = subtri3d[idx_right]
            A, _, _, _ = np.linalg.lstsq(
                np.hstack([tri2d_pts, np.ones((3,1))]),
                subtri3d_pts,
                rcond=None
            )
            triangle_id_to_affine[triangle_id] = A

        # === ここから全ポート分の物理LEDインデックスで3D座標・色を割り当てる ===
        # triangle_led_physical_index/led_mapを使い、物理LED番号順で全LEDを3D表示
        led_xs, led_ys, led_zs, led_colors = [], [], [], []
        triangle_led_physical_index = []
        for seg in range(NUM_PORTS):
            port = []
            base = seg * TRIANGLES_PER_PORT * LEDS_PER_TRIANGLE
            for tri in range(TRIANGLES_PER_PORT):
                tri_leds = [base + tri*LEDS_PER_TRIANGLE + i + 1 for i in range(LEDS_PER_TRIANGLE)]
                port.append(tri_leds)
            triangle_led_physical_index.append(port)
        led_map = dict()
        for seg in range(len(triangle_led_physical_index)):
            for tri in range(len(triangle_led_physical_index[seg])):
                for led in range(len(triangle_led_physical_index[seg][tri])):
                    idx = triangle_led_physical_index[seg][tri][led] - 1
                    led_map[idx] = (seg, tri, led)
        for idx in range(NUM_PORTS * TRIANGLES_PER_PORT * LEDS_PER_TRIANGLE):
            seg, tri, led = led_map[idx]
            triangle_id = seg * TRIANGLES_PER_PORT + tri + 1
            if triangle_id not in triangle_id_to_affine:
                continue
            A = triangle_id_to_affine[triangle_id]
            u, v = triangle_local_coords[led]
            x = (u - 0.5) * side
            y = (v - 0.5) * side * np.sqrt(3)/2
            pt2d = np.array([x, y, 1.0])
            pt3d = pt2d @ A
            led_xs.append(pt3d[0])
            led_ys.append(pt3d[1])
            led_zs.append(pt3d[2])
            img_x, img_y = get_image_pixel_for_led(seg, tri, led)
            # 物理LEDインデックスでHSVグラデーション色を付与
            import colorsys
            hue = led / float(LEDS_PER_TRIANGLE)
            grad_color = np.array(colorsys.hsv_to_rgb(hue, 1.0, 1.0))
            if 0 <= img_y < img.shape[0] and 0 <= img_x < img.shape[1]:
                base_color = img[img_y, img_x] / 255.0
            else:
                base_color = np.array([0.0, 0.0, 0.0])
            # 画像色とグラデーション色を掛け合わせて可視化
            color = (base_color * grad_color) ** 2.0
            led_colors.append(color)
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        zs = np.array(led_zs)
        alphas = 0.15 + 0.85 * (zs - zs.min())/(zs.max() - zs.min() + 1e-8)
        led_rgba = [np.append(c, a) for c, a in zip(led_colors, alphas)]
        ax.scatter(led_xs, led_ys, led_zs, c=led_rgba, s=18, edgecolors='k', linewidths=0.5, depthshade=True)
        # --- 各LEDに物理インデックス番号ラベルを表示 ---
        for i, (x, y, z) in enumerate(zip(led_xs, led_ys, led_zs)):
            led_label = triangle_led_designators[i % LEDS_PER_TRIANGLE] if 'triangle_led_designators' in globals() else str(i % LEDS_PER_TRIANGLE)
            ax.text(x, y, z, led_label, color='black', fontsize=12, ha='center', va='center')

        # --- 各基板（三角形）にtriangle_idの数字ラベルを表示 ---
        # triangle_layout_dict: triangle_id→dict
        for triangle_id, tri_layout in triangle_layout_dict.items():
            # 対応する3Dアフィン変換が存在する場合のみ
            if triangle_id not in triangle_id_to_affine:
                continue
            A = triangle_id_to_affine[triangle_id]
            # triangleの中心座標（2D原点）を3D変換
            pt2d = np.array([0, 0, 1.0])
            pt3d = pt2d @ A
            # ラベルの向きを基板の回転に合わせる場合
            if label_orient:
                # tri_layout['angle']は2D三角形の回転角度（degree）
                # 3D空間での向き合わせは難しいため、ここではmatplotlibのtextのrotation引数を利用（投影方向での回転）
                angle = tri_layout['angle']
            else:
                angle = 0
            ax.text(pt3d[0], pt3d[1], pt3d[2], str(triangle_id), color='red', fontsize=12, ha='center', va='center', rotation=angle, rotation_mode='anchor')
        # --- Icosahedronの面エッジをwireframeで描画 ---
        for (i0, i1, i2) in faces:
            v0, v1, v2 = np.array(vertices[i0]), np.array(vertices[i1]), np.array(vertices[i2])
            ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], color='gray', linewidth=0.8)
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], color='gray', linewidth=0.8)
            ax.plot([v2[0], v0[0]], [v2[1], v0[1]], [v2[2], v0[2]], color='gray', linewidth=0.8)
        set_axes_equal(ax)
        try:
            ax.set_box_aspect([1,1,1])
        except Exception:
            pass
        ax.set_axis_off()
        ax.set_title('Icosahedron (20 faces) LED Layout Emulator')
        plt.tight_layout()
        plt.show()
        return

def show_led_physical_emulation(img, mode="physical", label_orient=False):
    """
    LED物理配置エミュレーション表示。
    mode: "physical"/"2d"/"3d"
    label_orient: Trueで基板角度に合わせて数字ラベルを回転（3Dのみ対応）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import csv
    # --- 物理LEDインデックス: 全ポート分（5x16x36=2880） ---
    NUM_PORTS = 5
    TRIANGLES_PER_PORT = 16
    LEDS_PER_TRIANGLE = 36
    triangle_local_coords_side = globals().get('triangle_local_coords_side', 17.425)
    global triangle_local_coords
    if mode == "3d":
        from mpl_toolkits.mplot3d import Axes3D
        from math import sqrt
        phi = (1 + sqrt(5)) / 2
        vertices = [
            (-1,  phi,  0), ( 1,  phi,  0), (-1, -phi,  0), ( 1, -phi,  0),
            ( 0, -1,  phi), ( 0,  1,  phi), ( 0, -1, -phi), ( 0,  1, -phi),
            ( phi,  0, -1), ( phi,  0,  1), (-phi,  0, -1), (-phi,  0,  1)
        ]
        faces = [
            (0, 11, 5), (5, 11, 4), (4, 9, 5), (3, 9, 4),
            (0, 5, 1), (1, 5, 9), (9, 8, 1), (3, 8, 9),
            (0, 1, 7), (7, 1, 8), (8, 6, 7), (3, 6, 8),
            (0, 7, 10), (10, 7, 6), (6, 2, 10), (3, 2, 6),
            (0, 10, 11), (11, 10, 2), (2, 4, 11), (3, 4, 2),
        ]
        triangle_layout = []
        with open('tools/triangle_piece_layout_mm.csv', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get('triangle_id') or not row.get('x') or not row.get('y') or not row.get('angle'):
                    continue
                triangle_layout.append({
                    'triangle_id': int(row['triangle_id']),
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'angle': float(row['angle'])
                })
        import os
        face_sub_to_id = {}
        id_to_face_sub = {}
        assign_csv = 'tools/face_sub_assignment.csv'
        if os.path.exists(assign_csv):
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
        # --- triangle_3d_orient_correction.csvがあればtriangle_idごとに角度補正 ---
        orient_correction_csv = 'tools/triangle_3d_orient_correction.csv'
        orient_correction = {}
        if os.path.exists(orient_correction_csv):
            with open(orient_correction_csv, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    angle_key = 'angle_deg' if 'angle_deg' in row else ('correction_deg' if 'correction_deg' in row else None)
                    if row.get('triangle_id') and angle_key:
                        orient_correction[int(row['triangle_id'])] = float(row[angle_key])
        triangle_id_to_affine = {}
        triangle_id_to_theta_deg = {}
        side = triangle_local_coords_side
        for subtris_idx, (fidx, subidx, subtri3d, tri_layout, triangle_id) in enumerate(all_subtris):
            tri2d_pts = np.array([
                [0, side/np.sqrt(3)],
                [-0.5*side, -0.5*side/np.sqrt(3)],
                [0.5*side, -0.5*side/np.sqrt(3)]
            ])
            # triangle_idごとの角度補正（deg）
            # correction_degがあればそれを絶対角度として使う、なければtri_layout['angle']を使う
            if triangle_id in orient_correction:
                theta_deg = orient_correction[triangle_id]
                print(f"[DEBUG] triangle_id={triangle_id}: correction_deg (from CSV) = {theta_deg}")
            else:
                theta_deg = tri_layout.get('angle', 0.0)
                print(f"[DEBUG] triangle_id={triangle_id}: angle (from layout) = {theta_deg}")
            theta = np.deg2rad(theta_deg)
            rot2d = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            tri2d_pts = tri2d_pts @ rot2d.T

            subtri3d_pts = np.zeros_like(subtri3d)
            z_vals = subtri3d[:,2]
            idx_top = np.argmax(z_vals)
            idxs_rest = [i for i in range(3) if i != idx_top]
            xy_rest = subtri3d[idxs_rest,:2]
            if xy_rest[0,0] < xy_rest[1,0]:
                idx_left = idxs_rest[0]
                idx_right = idxs_rest[1]
            else:
                idx_left = idxs_rest[1]
                idx_right = idxs_rest[0]
            subtri3d_pts[0] = subtri3d[idx_top]
            subtri3d_pts[1] = subtri3d[idx_left]
            subtri3d_pts[2] = subtri3d[idx_right]
            # --- 反転対策: 3D三角形が時計回り(CW)なら左右を入れ替える ---
            v1 = subtri3d_pts[1] - subtri3d_pts[0]
            v2 = subtri3d_pts[2] - subtri3d_pts[0]
            cross = np.cross(v1, v2)
            if cross[2] > 0:
                subtri3d_pts[1], subtri3d_pts[2] = subtri3d_pts[2].copy(), subtri3d_pts[1].copy()
            A, _, _, _ = np.linalg.lstsq(
                np.hstack([tri2d_pts, np.ones((3,1))]),
                subtri3d_pts,
                rcond=None
            )
            triangle_id_to_affine[triangle_id] = A
            triangle_id_to_theta_deg[triangle_id] = theta_deg
        led_xs, led_ys, led_zs, led_colors = [], [], [], []
        triangle_led_physical_index = []
        for seg in range(NUM_PORTS):
            port = []
            base = seg * TRIANGLES_PER_PORT * LEDS_PER_TRIANGLE
            for tri in range(TRIANGLES_PER_PORT):
                tri_leds = [base + tri*LEDS_PER_TRIANGLE + i + 1 for i in range(LEDS_PER_TRIANGLE)]
                port.append(tri_leds)
            triangle_led_physical_index.append(port)
        led_map = dict()
        for seg in range(len(triangle_led_physical_index)):
            for tri in range(len(triangle_led_physical_index[seg])):
                for led in range(len(triangle_led_physical_index[seg][tri])):
                    idx = triangle_led_physical_index[seg][tri][led] - 1
                    led_map[idx] = (seg, tri, led)
        for idx in range(NUM_PORTS * TRIANGLES_PER_PORT * LEDS_PER_TRIANGLE):
            seg, tri, led = led_map[idx]
            triangle_id = seg * TRIANGLES_PER_PORT + tri + 1
            if triangle_id not in triangle_id_to_affine:
                continue
            A = triangle_id_to_affine[triangle_id]
            u, v = triangle_local_coords[led]
            x = (u - 0.5) * side
            y = (v - 0.5) * side * np.sqrt(3)/2
            pt2d = np.array([x, y, 1.0])
            pt3d = pt2d @ A
            led_xs.append(pt3d[0])
            led_ys.append(pt3d[1])
            led_zs.append(pt3d[2])
            img_x, img_y = get_image_pixel_for_led(seg, tri, led)
            # 物理LEDインデックスでHSVグラデーション色を付与
            import colorsys
            hue = led / float(LEDS_PER_TRIANGLE)
            grad_color = np.array(colorsys.hsv_to_rgb(hue, 1.0, 1.0))
            if 0 <= img_y < img.shape[0] and 0 <= img_x < img.shape[1]:
                base_color = img[img_y, img_x] / 255.0
            else:
                base_color = np.array([0.0, 0.0, 0.0])
            # 画像色とグラデーション色を掛け合わせて可視化
            color = (base_color * grad_color) ** 2.0
            led_colors.append(color)
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        zs = np.array(led_zs)
        alphas = 0.15 + 0.85 * (zs - zs.min())/(zs.max() - zs.min() + 1e-8)
        led_rgba = [np.append(c, a) for c, a in zip(led_colors, alphas)]
        ax.scatter(led_xs, led_ys, led_zs, c=led_rgba, s=18, edgecolors='k', linewidths=0.5, depthshade=True)
        # --- 各LEDに物理インデックス番号ラベルを表示 ---
        for i, (x, y, z) in enumerate(zip(led_xs, led_ys, led_zs)):
            led_label = triangle_led_designators[i % LEDS_PER_TRIANGLE] if 'triangle_led_designators' in globals() else str(i % LEDS_PER_TRIANGLE)
            ax.text(x, y, z, led_label, color='black', fontsize=12, ha='center', va='center')

        # --- 各基板（三角形）にtriangle_idの数字ラベルを表示（常に水平） ---
        for triangle_id, tri_layout in triangle_layout_dict.items():
            if triangle_id not in triangle_id_to_affine:
                continue
            A = triangle_id_to_affine[triangle_id]
            pt2d = np.array([0, 0, 1.0])
            pt3d = pt2d @ A
            # triangle_idをmagenta色でラベル表示（デバッグ用）
            ax.text(pt3d[0], pt3d[1], pt3d[2], f"ID:{triangle_id}", color='magenta', fontsize=14, ha='center', va='center', rotation=0)

            # --- 基板の物理的な向きを矢印で描画 ---
            if label_orient:
                theta_deg = triangle_id_to_theta_deg[triangle_id]  # アフィン変換と完全一致させる
                # 基板の「上方向」を「中心→U15」方向に修正
                # 三角形中心（ローカル[0,0,1]）
                pt2d_base = np.array([0, 0, 1.0])
                pt3d_base = pt2d_base @ A
                # U15方向ローカルベクトル（中心→U15）
                center_uv = np.mean(triangle_local_coords, axis=0)
                u15_uv = triangle_local_coords[14]
                vec2d_uv = u15_uv - center_uv
                vec2d_xy = vec2d_uv * side
                pt2d_tip = np.array([vec2d_xy[0], vec2d_xy[1], 1.0])
                pt3d_tip = pt2d_tip @ A
                # デバッグ出力: U15方向ベクトル
                vec3d = pt3d_tip - pt3d_base
                print(f"[DEBUG] triangle_id={triangle_id} theta_deg={theta_deg} U15_vec3d={vec3d}")
                # 矢印を描画
                ax.plot([pt3d_base[0], pt3d_tip[0]], [pt3d_base[1], pt3d_tip[1]], [pt3d_base[2], pt3d_tip[2]], color='blue', linewidth=1.2)

        for (i0, i1, i2) in faces:
            v0, v1, v2 = np.array(vertices[i0]), np.array(vertices[i1]), np.array(vertices[i2])
            ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], color='gray', linewidth=0.8)
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], color='gray', linewidth=0.8)
            ax.plot([v2[0], v0[0]], [v2[1], v0[1]], [v2[2], v0[2]], color='gray', linewidth=0.8)
        set_axes_equal(ax)
        try:
            ax.set_box_aspect([1,1,1])
        except Exception:
            pass
        ax.set_axis_off()
        ax.set_title('Icosahedron (20 faces) LED Layout Emulator')
        plt.tight_layout()
        plt.show()
        return
    # --- 2D/physicalモードは既存のまま（省略） ---
    # 必要ならここに2D/physicalのラベル表示拡張も実装可

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('imgfile')
    parser.add_argument('--mode', default='physical', choices=['physical','2d','3d'])
    parser.add_argument('--label_orient', action='store_true', help='基板角度に合わせて数字ラベルを回転')
    args = parser.parse_args()
    if args.imgfile.endswith('.h'):
        img = load_c_array_image(args.imgfile)
    else:
        img = load_raw_rgb_image(args.imgfile)
    show_led_physical_emulation(img, mode=args.mode, label_orient=args.label_orient)
