import cv2
import os
import numpy as np
from natsort import natsorted
def display_and_find_corners(folder_path, chessboard_size=(11, 8), image_size=(2048, 2048)):
    """
    顯示資料夾中的每張圖片，並在棋盤格角點上繪製標記。

    :param folder_path: 圖片資料夾的路徑
    :param chessboard_size: 棋盤格的角點數量 (width, height)
    :param target_width: 調整後的目標寬度
    """
    # 確認資料夾存在
    if not os.path.exists(folder_path):
        print(f"資料夾 {folder_path} 不存在")
        return

    # 設定搜尋範圍、零區域、與準則
    winSize = (5, 5)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

    for filename in natsorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)

        # 確認檔案是圖片
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                # 使用 np.frombuffer 和 cv2.imdecode 讀取圖片，以支援非 ASCII 路徑
                with open(file_path, 'rb') as file:
                    img_array = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if image is not None:
                    # 調整圖片大小到指定的 image_size
                    resized_image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)

                    # 將圖片轉為灰階
                    grayimg = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)


                    # 尋找棋盤格角點
                    ret, corners = cv2.findChessboardCorners(grayimg, chessboard_size, None)

                    if ret:
                        # 增強角點的準確性
                        corners = cv2.cornerSubPix(grayimg, corners, winSize, zeroZone, criteria)

                        # 繪製角點
                        cv2.drawChessboardCorners(resized_image, chessboard_size, corners, ret)
                        result_image = cv2.resize(resized_image, (800,800), interpolation=cv2.INTER_AREA)
                        # 顯示帶有角點的圖片
                        cv2.imshow("Chessboard Corners", result_image)
                        print(f"在 {filename} 中找到並繪製棋盤格角點")
                        cv2.waitKey(0)  # 按任意鍵繼續到下一張圖片
                    else:
                        print(f"無法在 {filename} 中找到棋盤格角點")

                else:
                    print(f"無法讀取圖片：{file_path}")
            except Exception as e:
                print(f"讀取圖片時發生錯誤：{file_path}, 錯誤：{e}")
        else:
            print(f"跳過非圖片檔案：{filename}")

    # 關閉所有 OpenCV 視窗
    cv2.destroyAllWindows()

def find_intrinsic(folder_path, chessboard_size=(11,8), square_size=0.02, image_size=(2048, 2048)):
    """
    對指定資料夾中的所有棋盤格圖片進行角點檢測和相機校正。

    :param folder_path: 圖片資料夾的路徑
    :param chessboard_size: 棋盤格內角點的數量 (columns, rows)
    :param square_size: 每個棋盤格的實際大小（單位：米）
    :param image_size: 圖片的大小 (width, height)
    """

 # 準備棋盤格的三維座標
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # 將棋盤格的坐標縮放到實際尺寸

    # 初始化存儲點的列表
    object_points = []  # 3D points in real world space
    image_points = []   # 2D points in image plane

    # 遍歷資料夾中的所有圖片
    for image_name in natsorted(os.listdir(folder_path)):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"無法讀取圖片：{image_path}")
                continue
            resized_image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

            # 檢測棋盤格角點
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if ret:
                object_points.append(objp)
                # 提高角點精度
                criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                image_points.append(corners2)

            else:
                print(f"在圖片中未檢測到棋盤格角點：{image_name}")


    if len(object_points) < 1:
        print("沒有足夠的圖片來進行相機校正。")
        return

    # 進行相機校正
    ret, ins, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

    return ins, dist, rvecs, tvecs

