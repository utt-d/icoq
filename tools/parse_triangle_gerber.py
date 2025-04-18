import re
import csv

def parse_gerber_triangle_vertices(gerber_path):
    # KiCad GerberのX/Y座標抽出（単位: mm * 1e6）
    with open(gerber_path, encoding='utf-8') as f:
        lines = f.readlines()
    pts = []
    pat = re.compile(r'X(-?\d+)Y(-?\d+)D0[12]')
    for line in lines:
        m = pat.search(line)
        if m:
            x = int(m.group(1)) / 1e6
            y = int(m.group(2)) / 1e6
            pts.append((x, y))
    # 3頂点抽出
    uniq = []
    for pt in pts:
        if pt not in uniq:
            uniq.append(pt)
    return uniq[:3]

def triangle_center_and_angle(pts):
    # 外心（重心）
    cx = sum(x for x, _ in pts) / 3
    cy = sum(y for _, y in pts) / 3
    # 1辺目(水平基準)の角度
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
    # 一辺の長さ
    side = ((x1-x0)**2 + (y1-y0)**2)**0.5
    return cx, cy, angle, side

if __name__ == "__main__":
    import math
    gerber = "triangle_led5-EdgeCuts.gbr"
    pts = parse_gerber_triangle_vertices(gerber)
    cx, cy, angle, side = triangle_center_and_angle(pts)
    print(f"center=({cx:.3f},{cy:.3f}), angle={angle:.3f}, side={side:.3f}")
    print("vertices:", pts)
