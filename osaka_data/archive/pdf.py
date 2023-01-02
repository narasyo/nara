import os
import img2pdf
from PIL import Image # img2pdfと一緒にインストールされたPillowを使います
from tqdm import tqdm

os.chdir("./osaka_data/archive/wasserstein_vel_UD=41*41_plot")

if __name__ == '__main__':
    pdf_FileName = "wasserstein_vel_UD.pdf" # 出力するPDFの名前
    png_Folder = "./" # 画像フォルダ
    extension  = ".png" # 拡張子がPNGのものを対象

    file = os.listdir(png_Folder)
    sorted_file = sorted(file)

    with open(pdf_FileName,"wb") as f:
        # 画像フォルダの中にあるPNGファイルを取得し配列に追加、バイナリ形式でファイルに書き込む
        f.write(img2pdf.convert([Image.open(png_Folder+j).filename for j in sorted_file if j.endswith(extension)]))