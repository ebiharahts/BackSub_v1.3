
###############################################################
#  背景差分を用いた物体検知 BackSub_v1.3
#    Copyright © 2020 Retail AI Lab,Inc.
#
#  大きい領域抽出後の拡大処理機能を強化できるか検討
###############################################################

import cv2
import numpy as np

# (定数) ------------------------------------------------------------
# エッジ抽出
CANNY_EXT = 180  # エッジ拡張の許容度
CANNY_EDGE = 180  # エッジ判定のスレッショルド
# 短蓄、長蓄の合成比率
ACCUM_RATE_01 = 0.1    # 短期蓄積比率
ACCUM_RATE_001 = 0.04  # 長期蓄積比率
# フィルター最終処理
THRES_VAL_MIN = 40  # 2値化する際の最小値 ～255
# 外れ値除去
GS_VAL_MIN = 0.9  # GS値 = 標準偏差の何倍離れているか？
# 有効な検出領域と判断する最小面積
MIN_VALID_AREA_SIZE = 1500
# 有効な物体検知とみなす領域数の最大値 (超える場合はカゴ全体の振動と判断)
MAX_NAREA = 50
# 検出領域の面積合計が下記以下になったら検知完了と判断(軟らかい物体対応)
MIN_AREA_SIZE = 500
# マージ時のマージン (領域を広げて接触判定)
M_MARGIN = 3
# (定数) ------------------------------------------------------------

# 出力画像データを保管しておく辞書型変数
# dict0:動作中ずっと上書きされる、dictHoldFlagが立つとdict0をdict1にコピー
dict0, dict1 = {}, {}
# 何回目のファイル出力かを記録(出力ファイルの接頭ワードとして使う)
outCt = 0
# dict0からdict1へのコピーを指示するワンショットフラグ
dictHoldFlag = False  # 最大枠が更新される度にTrueになる

# ファイルに出力するか否かを決めるフラグ、mp4読み込み時はTrue
dictFileDumpFlag = True  # Trueになったら1回分の物体検出経過を一式出力
# 外部読み込みファイル名、スタート時は空欄
vfile = ''


# sキー入力で記録
def storeDict(name, img):  # 出力画像を辞書変数に保存
    global dict0, dict1, dictHoldFlag
    if dictHoldFlag:
        dict1 = dict0.copy()
        dictHoldFlag = False
    dict0[name] = img  # 同じ名前であれば何度も上書き
    return


# 中間画像を出力、ウィンドウ表示、ファイルダンプ(出力回数を頭に付けてファイル出力)
def writeDict():
    global dict1, outCt, dictFileDumpFlag

    for i in dict1.keys():
        cv2.imshow(i, dict1[i])

    if dictFileDumpFlag:
        for i in dict1.keys():
            name = "{:02d}".format(outCt)
            cv2.imwrite(name+'_'+i+'.png', dict1[i])
        print('Image files were stored!')
        outCt += 1  # ファイル出力後カウントを進める
    return


# 2つの領域間の最小距離を算出 (x1s,x1e,x0s,x1e)
def distance(s0, e0, s1, e1):
    if e0 < s1:
        dist = s1 - e0
    elif e1 < s0:
        dist = s0 - e1
    else:  # 領域位置に重複がある場合は距離ゼロ
        dist = 0
    return dist


# 検出された領域同士の距離を積算
def CheckDist(stats):
    global GS_VAL_MIN  # 外れ値ではないと判定するGSパラメータの最小値
    areaCnt = len(stats)
    if areaCnt <= 3:  # 領域数が3未満なら領域除去処理はskip
        return stats
    else:
        dist = [0]  # 積算距離のリスト
        stats2 = [[0, 0, 0, 0, 0]]  # 外れ値を取り除いた領域リスト
        for i in range(1, areaCnt):
            x, y, w, h, size = stats[i]
            sum = 0
            for j in range(1, areaCnt):
                xt, yt, wt, ht, sizet = stats[j]
                distX = distance(x, x+w, xt, xt+wt)
                distY = distance(y, y+h, yt, yt+ht)
                d1 = int(np.sqrt(distX**2 + distY**2))
                sum += d1
            dist.append(int(sum/(areaCnt-2)))

        # 積算距離の平均と分散を求める
        ave = int(np.mean(dist))
        sd = int(np.std(dist))
        if sd == 0:
            return stats
#         print('No=', areaCnt, 'ave=', ave, 'S.D.=', sd, 'ave+sd=', ave+sd)

        # 積算距離が平均より小さいものは無条件にgs値=0 ⇒ 有効
        for i in range(1, areaCnt):
            x, y, w, h, size = stats[i]
            if (100 < ave) & (60 < sd):  # バラけ過ぎ = カゴが動いたと判断
                stats2.append([x, y, w, h, -1])
            elif ave < dist[i]:  # 領域間距離平均が平均値より大きい
                gs = (dist[i]-ave)/sd  # グラブス・スミルノフ棄却検定
                if GS_VAL_MIN < gs:
                    # GS最小値より大きいGS値を持つものはサイズを-1に変更 ⇒ 外れ値と判断
                    stats2.append([x, y, w, h, -1])
                else:
                    stats2.append([x, y, w, h, size])
            else:
                gs = 0  # 他領域との平均距離が平均値より近いものは外れ値ではないと判断
                stats2.append([x, y, w, h, size])
    return stats2


# ******************************************************* カメラ起動の初期化
cap = cv2.VideoCapture(0)  # 起動時はカメラ入力

if not cap.isOpened():
    print('Camera is not connected, I cannot open it')
    import sys
    sys.exit(1)

# VSCでは下記設定はエラーになる
# print(cap.set(cv2.CAP_PROP_FPS, 2), end='')  # 2fpsにセット

ret, frame = cap.read()
height, width = frame.shape[:2]
print(' x=', width, ' y=', height)

# 初期化
frmAcc01 = None
frmAcc001 = None
nlbl1 = 0
PeakDetectedFlag = False
MS_maxSize = width*height  # 起動時、画像が安定するまでエリア検出を止める

# ************************************************************** main loop

while True:
    ret, frame = cap.read()
    if not ret:
        break

# 平滑化
    blu = cv2.GaussianBlur(frame, (5, 5), 0)

# エッジ抽出
    edges = cv2.Canny(blu, CANNY_EXT, CANNY_EDGE)
    if frmAcc01 is None:
        frmAcc01 = edges.copy().astype('float')
        frmAcc001 = edges.copy().astype('float')
        continue

# 1/10, 1/100積分器
    cv2.accumulateWeighted(edges, frmAcc01, ACCUM_RATE_01)
    cv2.accumulateWeighted(edges, frmAcc001, ACCUM_RATE_001)

# 差分画像生成
    frm01 = cv2.convertScaleAbs(frmAcc01)
    frm001 = cv2.convertScaleAbs(frmAcc001)
    frmSub = cv2.subtract(frm01, frm001)
#     storeDict('0.Acc01', frm01)
#     storeDict('1.Acc001', frm001)
#     storeDict('2.frmSub', frmSub)

# 差分画像のノイズ除去
    gaus = cv2.GaussianBlur(frmSub, (3, 3), 0)
#     storeDict('3.Gaussian', gaus)

# 2値化
    thres = cv2.threshold(gaus, THRES_VAL_MIN, 255, cv2.THRESH_BINARY)[1]
#     storeDict('4.thres', thres)

# クロージング/穴埋め処理 (膨張後、縮小)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
#     storeDict('5.closing', closing)

# 領域拡張 隙間を埋める
    dil = cv2.dilate(closing, kernel, iterations=2)
    cv2.imshow('6.Dilate_RT', dil)  # 途中状態を見るための別ウィンドウ
    storeDict('6.Dilate', dil)

# 領域抽出
    nArea, labeledImg, stats, center = \
        cv2.connectedComponentsWithStats(dil, 8, cv2.CV_32S)

# 領域検出がゼロなら検出完了、蓄積画像を出力してリセット
    if nArea == 1:  # 背景のみになったら物体検出完了
        MaxX, MaxY, MinX, MinY = 0, 0, width, height
        MS_maxSize = 0
        if PeakDetectedFlag:  # 物体検出済みの状態
            writeDict()  # 中間画像ファイルを出力する
            dict0.clear()
            dict1.clear()
#             dictFileDumpFlag = False
            PeakDetectedFlag = False

# 柔らかいものがあると確定まで時間がかかる、2個以上領域が残る状態でも合計面積500以下なら検知完了とする
    elif (2 <= nArea) & (nArea < MAX_NAREA):
        SUMsize = 0
        for i in range(1, nArea):  # 検出領域の面積合計算出
            x, y, w, h, size = stats[i]
            SUMsize += size

        # 検出領域面積合計が500以下なら物体検出完了と判断 (柔らかい物体対応)
        if PeakDetectedFlag & (SUMsize < MIN_AREA_SIZE):
            MaxX, MaxY, MinX, MinY = 0, 0, width, height
            MS_maxSize = 0
            writeDict()  # 蓄積済の中間画像ファイルを出力
            dict0.clear()
            dict1.clear()
#             dictFileDumpFlag = False
            PeakDetectedFlag = False

# 積算距離を用いた外れ値除外処理 (カゴ全体の振動を排除)、外れ値はSize=-1にセットされる
    stats = CheckDist(stats)

# 検出領域の中で最大サイズの領域を抽出
    SS_maxSize = 0  # SingleShot(検出1回分の最大サイズ)
    frame2 = frame.copy()
    frame3 = frame.copy()
    if (2 <= nArea) & (nArea < MAX_NAREA):  # 正常な検知エリアの数は2個から25個まで
        for i in range(1, nArea):
            x, y, w, h, size = stats[i]
            if size == -1:  # 外れ値
                frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h),
                                       (0, 0, 255), 1)  # 赤描画 GS値で除去
            else:
                frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h),
                                       (255, 255, 255), 1)  # 白描画 通常の枠
                if SS_maxSize < size:
                    SS_maxSize = size
                    MinX = x
                    MaxX = x+w
                    MinY = y
                    MaxY = y+h

    if 0 < SS_maxSize:  # 1個以上の有効データが存在、全て外れ値だと成立しない
        frame2 = cv2.rectangle(frame2, (MinX, MinY), (MaxX, MaxY),
                               (0, 255, 0), 1)  # 緑描画 最大サイズ
        cv2.imshow('7.Rect_RT', frame2)  # 画面観測用
        storeDict('7.Rect', frame2)

# 面積最大領域に重複する領域を併合して外枠拡張 (ひとつになるはずの領域が分割されたケースを救済)
        # 元のサイズからスタート
        MinX2, MaxX2, MinY2, MaxY2 = MinX, MaxX, MinY, MaxY
        # 元の領域にマージンを付けて拡張、多少の隙間があっても重複と判断
        MinX1, MaxX1, MinY1, MaxY1 = MinX-M_MARGIN, MaxX+M_MARGIN, MinY-M_MARGIN, MaxY+M_MARGIN
        for i in range(1, nArea):
            x, y, w, h, size = stats[i]
            if size == -1:
                frame3 = cv2.rectangle(frame3, (x, y), (x+w, y+h),
                                       (0, 0, 255), 1)  # 赤描画 GS値で除去
            else:
                if (MinX1 <= x) & (x <= MaxX1) & (MinY1 <= y) & (y <= MaxY1):
                    if MaxX2 < x+w:
                        MaxX2 = x+w
                    if MaxY2 < y+h:
                        MaxY2 = y+h
                if (MinX1 <= x+h) & (x+h <= MaxX1) & (MinY1 <= y) & (y <= MaxY1):
                    if x < MinX2:
                        MinX2 = x
                    if MaxY2 < y+h:
                        MaxY2 = y+h
                if (MinX1 <= x) & (x <= MaxX1) & (MinY1 <= y+h) & (y+h <= MaxY1):
                    if MaxX2 < x+w:
                        MaxX2 = x+w
                    if y < MinY2:
                        MinY2 = y
                if (MinX1 <= x+w) & (x+w <= MaxX1) & \
                   (MinY1 <= y+h) & (y+h <= MaxY1):
                    if x < MinX2:
                        MinX2 = x
                    if y < MinY2:
                        MinY2 = y
                frame3 = cv2.rectangle(frame3, (x, y), (x+w, y+h),
                                       (255, 255, 255), 1)  # 白描画 通常の枠

        MinX, MaxX, MinY, MaxY = MinX2, MaxX2, MinY2, MaxY2
        SS_maxSize = (MaxX-MinX)*(MaxY-MinY)
        frame3 = cv2.rectangle(frame3, (MinX, MinY), (MaxX, MaxY),
                               (0, 255, 0), 1)  # 緑描画 最大サイズ
        storeDict('8.RectMerged', frame3)

# 起動直後の検出結果をスキップ
    if MS_maxSize < SS_maxSize:  # 起動直後(検出1回目)は9999なので成立しない、不正画像抑制
        MS_maxSize = SS_maxSize
        if MIN_VALID_AREA_SIZE < MS_maxSize:  # 小さ過ぎる場合は描画しない(検出完了としない)
            frame5 = cv2.rectangle(frame, (MinX, MinY), (MaxX, MaxY),
                                   (0, 255, 255), 2)  # 黄描画
            storeDict('10.Detected', frame5)
            PeakDetectedFlag = True  # 検出完了フラグ
            dictHoldFlag = True  # 辞書ファイルのスナップショットを作成

    # qが押されたら終了
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    # sが押されたら物体検知の中間画像のファイルダンプを開始
    elif key & 0xFF == ord('s'):
        dictFileDumpFlag = True
        print('Image Store is started!')
    elif key & 0xff == ord('f'):  # カメラ入力からファイル入力に切替
        import os, tkinter, tkinter.filedialog, tkinter.messagebox
        # ファイル選択ダイアログの表示
        root = tkinter.Tk()
        root.withdraw()
        fTyp = [("","*")]
        iDir = os.path.abspath(os.path.dirname(__file__))
        # tkinter.messagebox.showinfo('動画入力','動画ファイルを選択してください！')
        vfile = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
        print('動画ファイル名=', vfile)
        cap.release()
        cap = cv2.VideoCapture(vfile)
        if not cap.isOpened():
            print('Video file has the problem!')
            import sys
            sys.exit(1)

cap.release()
cv2.destroyAllWindows()