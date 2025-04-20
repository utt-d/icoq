import csv
import os

# デフォルト値（面数・サブ数・基板ID範囲）
NUM_FACES = 16 # 必要に応じて20に変更
NUM_SUBS = 4
START_ID = 1

# 任意の順番でtriangle_idを割り当てたい場合は、下記リストに指定
# 例: [5,3,1,2,4,6,7,8,9,10,11,12,13,14,15,16]
# 空リストの場合は昇順で自動割当
TRIANGLE_ID_ORDER = [1, 16, 3, 2, 4, 15, 13, 14, 12, 7, 5, 6, 9, 8, 11, 10]  # ←ここに任意の順を指定

# 出力ファイル名
ASSIGN_CSV = 'tools/face_sub_assignment.csv'

def main():
    # 既存ファイルがあれば上書き確認
    if os.path.exists(ASSIGN_CSV):
        ans = input(f"{ASSIGN_CSV} already exists. Overwrite? (y/N): ").strip().lower()
        if ans != 'y':
            print("Aborted.")
            return
    rows = []
    total = NUM_FACES * NUM_SUBS
    # 指定が足りなければ昇順で埋める
    ids = list(TRIANGLE_ID_ORDER) + [i for i in range(START_ID, START_ID+total) if i not in TRIANGLE_ID_ORDER]
    ids = ids[:total]
    idx = 0
    for face in range(NUM_FACES):
        for sub in range(NUM_SUBS):
            rows.append({'face': face, 'sub': sub, 'triangle_id': ids[idx]})
            idx += 1
    with open(ASSIGN_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['face', 'sub', 'triangle_id'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} assignments to {ASSIGN_CSV}")

if __name__ == '__main__':
    main()
