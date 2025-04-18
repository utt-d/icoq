# image_converter.py
"""
画像ファイル（PNG/JPG/BMP等）をM5Capsule用RGB24配列（Cヘッダファイル）に変換するツール。

使い方:
  python image_converter.py input.png output.h

出力は const uint8_t my_image[IMAGE_W*IMAGE_H*3] = {...}; の形で保存されます。

M5Capsuleプロジェクト本体と分離するため、src/ ではなく tools/ ディレクトリに配置します。
"""
import sys
from PIL import Image

# --- 変換先画像サイズ（M5Capsule用パネル解像度） ---
IMAGE_W = 180
IMAGE_H = 32


# --- 入力画像を読み込み、M5Capsule用RGB配列に変換して出力 ---
def convert_image(input_path, output_path):
    im = Image.open(input_path).convert("RGB").resize((IMAGE_W, IMAGE_H))
    pixels = list(im.getdata())
    # Cヘッダファイル形式で出力
    if output_path.endswith('.h'):
        with open(output_path, "w") as f:
            f.write(f"#pragma once\nconst uint8_t my_image[{IMAGE_W * IMAGE_H * 3}] = {{\n")
            for i, (r, g, b) in enumerate(pixels):
                f.write(f"{r},{g},{b},")
                if (i + 1) % 12 == 0:
                    f.write("\n")
            f.write("};\n")
        print(f"変換完了: {output_path}")
    # 生RGBバイナリ形式で出力
    elif output_path.endswith('.rgb'):
        arr = bytearray()
        for r, g, b in pixels:
            arr.extend([r, g, b])
        with open(output_path, "wb") as f:
            f.write(arr)
        print(f"変換完了: {output_path}")
    else:
        print("出力ファイルは .h または .rgb で指定してください")
        sys.exit(1)


# --- コマンドライン引数をパースし、画像変換を実行 ---
def main():
    if len(sys.argv) < 3:
        print("Usage: python image_converter.py input.png output.rgb|output.h")
        sys.exit(1)
    convert_image(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
