from matplotlib.pyplot import axis
import numpy as np
import cv2
import os
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow
tf = tensorflow.compat.v1
tf.disable_eager_execution()
kr = tf.keras

folders = ["circle", "square", "triangle"]

model = kr.models.load_model("model/classifier_20221217_124749.h5")
model.summary()

files = glob.glob("imgs/test/triangle/*.png", recursive = True)
print(files)
for file_index, file in enumerate(files):
    """テスト画像の読込 + 二値化"""
    img_raw = cv2.imread(file)
    img_result = img_raw
    img_gray = cv2.cvtColor(img_raw,cv2.COLOR_BGR2GRAY)
    img_bin = cv2.bitwise_not(cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,20))
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    img_bin = cv2.dilate(img_bin, kernel, iterations=3) # 線が細すぎるので膨張処理しておく
    img_bin = cv2.erode(img_bin, kernel, iterations=2)
    

    """輪郭検出"""
    contours,hierarchy = cv2.findContours(img_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)

    """画像のトリミング + サイズ変更"""
    def img_convert(img_bin, upper, lower, left,right):
        img = img_bin[upper:lower, left:right]
        img = cv2.resize(img, dsize = (20,20))
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
        img = cv2.dilate(img, kernel)
        return img

    shape_position = []
    predict_images = []
    print(len(contours))
    for cnt in contours:
        x_list = []
        y_list = []
        for pos in cnt:
            x_list.append(pos[0][0])
            y_list.append(pos[0][1])
        
        """トリミングの範囲決め"""
        upper = min(y_list)
        lower = max(y_list)
        left = min(x_list)
        right = max(x_list)

        """予測用データの作成"""
        img_clip = img_convert(img_bin,upper,lower,left,right)
        img_clip = img_clip / 255
        predict_images.append(img_clip)
        shape_position.append([upper, lower, left, right])
        cv2.imshow("img", img_clip)
        cv2.waitKey(0)

    predict_images = np.asarray(predict_images)
    predict_images = predict_images.reshape(len(contours),20,20,1)

    predict_result = model.predict(predict_images).argmax(axis=1)
    print(predict_result)

    for i, pos in enumerate(shape_position):
        upper, lower, left, right = pos
        cv2.putText(img_result,folders[predict_result[i]],(left,upper - 10),fontFace = cv2.FONT_HERSHEY_COMPLEX,fontScale = 1,color = (255,0,0),thickness=2,lineType=cv2.LINE_AA)
        cv2.rectangle(img_result,(left,upper),(right,lower),(255,0,0),thickness = 2)

    cv2.imwrite("result_img/{}/result_".format("triangle") + str(file_index) + ".jpg", img_result)
