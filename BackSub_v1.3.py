
###############################################################
#  背景差分を用いた物体検知 BackSub_v1.3
#    Copyright 2020 Retail AI X,Inc.
#    探索型領域拡張方式に変更 (2020.07.31)
#
# ◇使い方 (画面上でキー入力)
#  q = quit/終了
#  s = 処理結果をファイルにセーブ
#  f = カメラ入力から動画ファイル入力に切替(ファイル選択画面)
#
###############################################################

import cv2
import numpy as np

# (定数) ------------------------------------------------------------
# エッジ抽出
CANNY_EXT = 130  # エッジ拡張の許容度  180
CANNY_EDGE = 130  # エッジ判定のスレッショルド  180
# 短蓄、長蓄の合成比率
ACCUM_RATE_01 = 0.1    # 短期蓄積比率
ACCUM_RATE_001 = 0.04  # 長期蓄積比率
# フィルター最終処理
THRES_VAL_MIN = 40  # 2値化する際の最小値 ～255
# 外れ値除去
GS_VAL_MIN = 0.9  # GS値 = 標準偏差の何倍離れているか？
# 有効な検出領域と判断する最小面積
MIN_VALID_AREA_SIZE = 2500
# 有効な物体検知とみなす領域数の最大値 (超える場合はカゴ全体の振動と判断)
MAX_NAREA = 50
# 検出領域の面積合計が下記以下になったら検知完了と判断(軟らかい物体対応)
MIN_AREA_SIZE = 500
# 探索型拡張時のマージン (注目領域を広げて接触判定)
A_MARGIN = 10
# (定数) ------------------------------------------------------------

# 出力画像データを保管しておく辞書型変数
# dict0:動作中ずっと上書きされる、dictHoldFlagが立つとdict0をdict1にコピー
dict0, dict1 = {}, {}
# 何回目のファイル出力かを記録(出力ファイルの接頭ワードとして使う)
outCt = 0
# dict0からdict1へのコピーを指示するワンショットフラグ
dictHoldFlag = False  # 最大枠が更新される度にTrueになる
# ファイル出力を行うか否かを決めるフラグ、's'または'f'でダンプ開始
dictFileDumpFlag = False  # Trueになったら中間画像ファイルをダンプ
# ファイル入力切替フラグ
fileInputFlag = False


# 中間画像を上書き記録
def storeDict(name, img):  # 画像を辞書変数に保存
    global dict0, dict1, dictHoldFlag
    if dictHoldFlag:
        dict1 = dict0.copy()
        dictHoldFlag = False
    dict0[name] = img  # 同じ名前であれば何度も上書き
    return


# dictに溜まった画像をウィンドウ表示、ファイルセーブ(出力回数を頭に付けてファイル出力)
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


# 領域拡張処理: 指定領域の周囲に併合出来る領域があるかチェック
def AreaExpandSearch(stats, idx, mergin, validAreaList):
    x, y, w, h, size = stats[idx]
    # 元の領域にマージンを付けて拡張、多少の隙間があっても併合対象と判断
    MinX, MaxX, MinY, MaxY = x-mergin, x+w+mergin, y-mergin, y+h+mergin

    for i in range(1, len(stats)):
        x, y, w, h, size = stats[i]
        if (size != -1) & (i != idx) & (i not in validAreaList):
            if (MinX <= x) & (x <= MaxX) & (MinY <= y) & (y <= MaxY) or \
                (MinX <= x+w) & (x+w <= MaxX) & (MinY <= y) & (y <= MaxY) or \
                (MinX <= x) & (x <= MaxX) & (MinY <= y+h) & (y+h <= MaxY) or \
                (MinX <= x+w) & (x+w <= MaxX) & (MinY <= y+h) & (y+h <= MaxY):
                    validAreaList.append(i)
                    AreaExpandSearch(stats, i, mergin, validAreaList)
    return

# リストに蓄積された対象領域を包む外枠を決定
def OuterFrame(stats, validAreaList):
    global width, height, frame
    MinX, MaxX, MinY, MaxY = width, 0, height, 0
    frameOF = frame.copy()

    for i in validAreaList:
        x, y, w, h, size = stats[i]
        if x < MinX:
            MinX = x
        if MaxX < x+w:
            MaxX = x+w
        if y < MinY:
            MinY = y
        if MaxY < y+h:
            MaxY = y+h
        frameOF = cv2.rectangle(frameOF, (x, y), (x+w, y+h), (0, 255, 0), 1)  # 緑描画 最大サイズ

    storeDict('8.ExpandedArea', frameOF)     

    return(MinX, MaxX, MinY, MaxY)


# ******************************************************* カメラ起動の初期化
cap = cv2.VideoCapture(0)  # 起動時はカメラ入力
if not cap.isOpened():
    print('Camera is not connected, I cannot open it')
    import sys
    sys.exit(1)

# VScodeでは下記設定はエラーになる
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
            PeakDetectedFlag = False
            if fileInputFlag:  # ファイル入力時はキー入力を待って次に進む
                cv2.waitKey(0)

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
            PeakDetectedFlag = False
            if fileInputFlag:  # ファイル入力時はキー入力を待って次に進む
                cv2.waitKey(0)            

# 積算距離を用いた外れ値除外処理 (カゴ全体の振動を排除)、外れ値はSize=-1にセットされる
    stats = CheckDist(stats)

# 検出領域の中で最大サイズの領域を抽出
    SS_maxSize = 0  # SingleShot(検出1回分の最大サイズ)
    frame2 = frame.copy()
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
                    MAXidx = i

    if 0 < SS_maxSize:  # 1個以上の有効データが存在、全て外れ値だと成立しない
        frame2 = cv2.rectangle(frame2, (MinX, MinY), (MaxX, MaxY),
                               (0, 255, 0), 1)  # 緑描画 最大サイズ
        cv2.imshow('7.Rect_RT', frame2)  # 画面観測用
        storeDict('7.Rect', frame2)

# 領域拡張処理 (注目領域をマージン分広げて近接領域を併合)
        validAreaList = [MAXidx]
        AreaExpandSearch(stats, MAXidx, A_MARGIN, validAreaList)
        # print('validAreaList=', validAreaList)
        MinX, MaxX, MinY, MaxY = OuterFrame(stats, validAreaList)
        SS_maxSize = (MaxX-MinX)*(MaxY-MinY)   

# 起動直後の検出結果をスキップ
    if MS_maxSize < SS_maxSize:  # 起動直後(検出1回目)は9999なので成立しない、不正画像抑制
        MS_maxSize = SS_maxSize
        if MIN_VALID_AREA_SIZE < MS_maxSize:  # 小さ過ぎる場合は描画しない(検出完了としない)
            frame5 = cv2.rectangle(frame, (MinX, MinY), (MaxX, MaxY),
                                   (0, 255, 255), 2)  # 黄描画
            storeDict('9.Detected', frame5)
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
        fileInputFlag = True
        import os, tkinter, tkinter.filedialog, tkinter.messagebox
        # ファイル選択ダイアログの表示
        root = tkinter.Tk()
        root.withdraw()
        fTyp = [("","*")]
        iDir = os.path.abspath(os.path.dirname(__file__))
        # tkinter.messagebox.showinfo('動画入力','動画ファイルを選択してください！')
        vfile = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
        print('動画ファイル名=', vfile)
        print('Step execution ⇒  Push any key!')
        cap.release()
        # dictFileDumpFlag = True
        cap = cv2.VideoCapture(vfile)
        if not cap.isOpened():
            print('Video file has the problem!')
            import sys
            sys.exit(1)

cap.release()
cv2.destroyAllWindows()