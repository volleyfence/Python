import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageOps

# このスクリプトのある場所をカレントディレクトリとする
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# テスト用
if not os.path.exists("TEST"):
    os.mkdir("TEST")
if os.path.exists("TEST/OCR.txt"):
    os.remove("TEST/OCR.txt")

# サブ関数
# 事前に読み込んだimg1の指定の範囲をimg2に貼り付けて返す
# img2：貼り付け先
# arr：座標情報
# name：画像の意味(テスト用)　日本語
# name2：画像の意味(テスト用)　英語
def img_copy(img2, arr, name, name2):
    x = arr[0]
    y = arr[1]
    w = arr[2]
    h = arr[3]
    x2 = arr[4]
    y2 = arr[5]
    w2 = arr[6]
    h2 = arr[7]
    
    tmp_part = img1[y:y+h, x:x+w]
    
    try:
        # ノイズ除去
        #tmp_part = cv2.medianBlur(tmp_part, 3)
        
        # 画像の幅が広いので右をトリミングする(上下中央の位置の座標で最後に黒となる位置を探してトリミングする)
        #--------------------------------------------------------------------------
        # 入力画像の一部を切り取る
        height, width = tmp_part.shape[:2]
        
        center_y = h // 2
        
        # 中央の行の画素データを取得
        row = tmp_part[center_y, :]
        
        # 黒の画素を見つける（しきい値を0とする）
        black_threshold = 10  # 多少の誤差を考慮して0ではなく10以下を黒とみなす
        
        # 右から走査（[::-1]で反転）
        black_x = np.where(row[::-1] <= black_threshold)[0]
        
        # 反転しているので元の座標に戻す
        first_black_x = width - 1 - black_x[0]
        
        # 右の余白をトリミングする(余裕を持たせる)
        tmp_part = tmp_part[:, 0:(first_black_x + 50)]
        #--------------------------------------------------------------------------
        
        cut_part = tmp_part
        
        # テスト確認用として指定範囲を画像ファイルに保存する
        #cv2.imwrite("TEST/" + name2 + ".jpg", cut_part)
        
        # トリミング後のサイズを取得
        h, w = cut_part.shape[:2]
        
        # 免許証の幅を超える場合はリサイズする
        if w > w2:
            aspect_ratio = w2 / float(w)
            w = w2
            h = int(h * aspect_ratio)
            cut_part = cv2.resize(cut_part, (w, h))
            cv2.imwrite("TEST/"+name2+"2.jpg", cut_part)
        
        # 白背景を透過させる
        # 白色部分をマスク化
        lower = np.array([200, 200, 200], dtype=np.uint8)  # 明るい白を閾値に
        upper = np.array([255, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(cut_part, lower, upper)  # 白部分を抽出
        alpha = cv2.bitwise_not(mask) / 255.0  # アルファ値（0〜1）
        
        # BGR部分取得
        cut_part_rgb = cut_part.copy()
        
        # 貼り付け先の領域を取得
        roi = img2[y2:y2+h, x2:x2+w]
        
        # アルファブレンドで合成
        for c in range(3):  # B, G, R の3チャンネル
            roi[:, :, c] = roi[:, :, c] * (1 - alpha) + cut_part_rgb[:, :, c] * alpha
        
        # img2 に合成結果を戻す
        img2[y2:y2+h, x2:x2+w] = roi
    except:
        pass
    
    return img2

def count_black_dots(arr):
    x = arr[0]
    y = arr[1]
    w = arr[2]
    h = arr[3]
    
    # 範囲を切り出し
    roi = img1[y:y+h, x:x+w]
    
    # グレースケールに変換
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 二値化（文字を黒とする）
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 中央横ラインの座標
    center_y = h // 2
    line_pixels = binary[center_y]  # 横1列
    
    # 黒（0）のピクセル数をカウント
    black_pixel_count = (line_pixels == 0).sum()
    
    return black_pixel_count

# 受け取った画像をA4サイズに調整する(300dpi)
def img_toA4(imgPath):
    # A4サイズ（300dpi）のピクセル数
    A4_W = 2481
    A4_H = 3507
    
    img = Image.open(imgPath)
    
    w, h = img.size
    
    # --- 縦が小さい → 上に白余白を追加 ---
    if h < A4_H:
        pad_top = A4_H - h
        img = ImageOps.expand(img, border=(0, pad_top, 0, 0), fill=255)
    
    # --- 縦が大きい → 下をトリミング ---
    elif h > A4_H:
        img = img.crop((0, 0, w, A4_H))
    
    # 保存
    img.save(imgPath, dpi=(300, 300))

img1Path = sys.argv[1]  # 入力画像のパス(引数)
img2Path = sys.argv[2]  # 出力画像のパス(引数)

# 現地調整用
X = 0
Y = 0

# 入力画像のトリミング範囲宣言
# x：入力画像から切り取る範囲のX軸の開始座標
# y：入力画像から切り取る範囲のy軸の開始座標
# w：入力画像から切り取る範囲の幅
# h：入力画像から切り取る範囲の高さ
# x2：出力画像に貼り付ける位置のx軸の開始座標
# y2：出力画像に貼り付ける位置のy軸の開始座標
# w2：出力画像に貼り付ける範囲の幅
# h2：出力画像に貼り付ける範囲の高さ
NAME      = [ 560 + X,  627 + Y,  920,   63,  160,   50,  680,   58]    # 氏名
BIRTHDAY  = [1815 + X,  543 + Y,  380,   60,  866,   50,  410,   58]    # 生年月日
HONSEKI   = [ 560 + X,  705 + Y, 1685,   60,  160,  113, 1135,   58]    # 本籍
ADDRESS   = [ 560 + X,  783 + Y, 1685,   60,  160,  176, 1135,   58]    # 住所
KOUHU1    = [1585 + X, 1119 + Y,  520,   54,  205,  239,  700,   54]    # 交付年月日・照会番号
KOUHU2    = [1585 + X, 1304 + Y,  520,   54,  205,  239,  700,   54]    # 記録等年月日
YUUKOU1   = [1586 + X, 1182 + Y,  520,   54,   70,  310,  770,   48]    # 有効期限
YUUKOU2   = [1586 + X, 1363 + Y,  520,   54,   70,  310,  770,   48]    # 有効期限
MENKYONO1 = [ 679 + X, 1179 + Y,  390,   54,  230,  583,  470,   60]    # 免許証番号
MENKYONO2 = [ 679 + X, 1362 + Y,  390,   54,  230,  583,  470,   60]    # 免許情報記録番号
SYUTOKU1  = [ 803 + X,  995 + Y,  300,   46,  135,  652,  355,   51]    # 二・小・原
SYUTOKU2  = [1386 + X,  995 + Y,  300,   46,  135,  700,  355,   51]    # その他
SYUTOKU3  = [1956 + X,  995 + Y,  300,   46,  135,  746,  355,   51]    # 第二種

# キャプチャー画像をA4サイズに調整
img_toA4(img1Path)

# 入力画像を読み込む(グレースケール)　※アスキーのみ対応
img1 = cv2.imread(img1Path)

# 出力画像を読み込む　※アスキーのみ対応
img2 = cv2.imread(img2Path)

# 氏名
img2 = img_copy(img2, NAME, "氏名", "NAME")

# 生年月日
img2 = img_copy(img2, BIRTHDAY, "生年月日", "BIRTHDAY")

# 本籍
img2 = img_copy(img2, HONSEKI, "本籍", "HONSEKI")

# 住所
img2 = img_copy(img2, ADDRESS, "住所", "ADDRESS")

#免許証と一体化マイナンバーのどちらの欄を読むか判断する
#判断基準は、切り取り範囲の上下中央の座標で黒が多いほうとする
count1 = 0
count2 = 0

# 交付日の比較
if count_black_dots(KOUHU1) >= count_black_dots(KOUHU2):
    count1 += 1
else:
    count2 += 1

# 有効期限の比較
if count_black_dots(YUUKOU1) >= count_black_dots(YUUKOU2):
    count1 += 1
else:
    count2 += 1

# 免許証番号の比較
if count_black_dots(MENKYONO1) >= count_black_dots(MENKYONO2):
    count1 += 1
else:
    count2 += 1

if count1 >= count2:
    # 交付年月日・照会番号
    img2 = img_copy(img2, KOUHU1, "交付日", "KOUHU")
    
    # 有効期限
    img2 = img_copy(img2, YUUKOU1, "有効期限", "YUUKOU")
    
    # 免許証番号
    img2 = img_copy(img2, MENKYONO1, "免許証番号", "MENKYONO")
else:
    # 記録等年月日
    img2 = img_copy(img2, KOUHU2, "交付日", "KOUHU")
    
    # 有効期限
    img2 = img_copy(img2, YUUKOU2, "有効期限", "YUUKOU")
    
    # 免許情報記録番号
    img2 = img_copy(img2, MENKYONO2, "免許証番号", "MENKYONO")

# 二・小・原
img2 = img_copy(img2, SYUTOKU1, "二・小・原", "SYUTOKU1")

# その他
img2 = img_copy(img2, SYUTOKU2, "その他", "SYUTOKU2")

# 第二種
img2 = img_copy(img2, SYUTOKU3, "第二種", "SYUTOKU3")

# OpenCVだとDPIが落ちるのでDPIを書き換えて保存
img_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img_rgb)
img.save(img2Path, dpi=(400, 400))


