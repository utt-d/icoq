import csv
import os

# 入力: triangle_piece_layout_mm.csv から triangle_id, angle を抽出
INPUT_CSV = os.path.join(os.path.dirname(__file__), 'triangle_piece_layout_mm.csv')
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), 'triangle_3d_orient_correction.csv')

rows = []
with open(INPUT_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            triangle_id = int(row['triangle_id'])
            angle = float(row['angle'])
            rows.append({'triangle_id': triangle_id, 'correction_deg': angle})
        except Exception:
            continue

# triangle_idでソート
rows.sort(key=lambda r: r['triangle_id'])

with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['triangle_id', 'correction_deg'])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Generated {OUTPUT_CSV} with {len(rows)} entries.")
