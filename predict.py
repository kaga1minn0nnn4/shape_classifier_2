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

def main():
    model = kr.models.load_model("model/classifier_20221217_124749.h5")
    folders = ["circle", "square", "triangle"]
    input_shape = (20, 20)
    num_of_img = 60

    result_handwrite = []
    result_nohandwrite = []
    
    for i, folder in enumerate(folders):
        files = glob.glob("imgs/test/{}/*.png".format(folder), recursive = True)
        test_imgs = []
        for j, file in enumerate(files):
            img_raw = cv2.imread(file)
            img_converted = convert_image(img_raw)
            cv2.imwrite("result_img/{}/result_".format(folder) + str(j) + ".jpg", img_converted)
            test_imgs.append(img_converted / 255.)
        predict_results = model.predict(np.asarray(test_imgs).reshape(len(files), input_shape[0], input_shape[1], 1)).argmax(axis=1)

        for j, pred_result in enumerate(predict_results):
            if j < 10:
                result_handwrite.append(1 if pred_result == i else 0)
            else:
                result_nohandwrite.append(1 if pred_result == i else 0)

    accuracy_handwrite = result_handwrite.count(1) / len(result_handwrite)
    accuracy_nohandwrite = result_nohandwrite.count(1) / len(result_nohandwrite)
    accuracy = (result_handwrite.count(1) + result_nohandwrite.count(1)) / num_of_img
    print("total accuracy : {} , handwrite accuracy : {} , not handwrite accuracy : {}".format(accuracy, accuracy_handwrite, accuracy_nohandwrite))

def clip_image(img_bin, upper, lower, left,right):
    img = img_bin[upper:lower, left:right]
    img = cv2.resize(img, dsize = (20,20), interpolation=cv2.INTER_AREA)
    return img

def convert_image(img_raw):
    img_gray = cv2.cvtColor(img_raw,cv2.COLOR_BGR2GRAY)
    img_bin = cv2.bitwise_not(cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,20))
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    img_bin = cv2.dilate(img_bin, kernel, iterations=3)
    img_bin = cv2.erode(img_bin, kernel, iterations=3)

    contours, _ = cv2.findContours(img_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)

    max_area = 0
    max_index = 0
    max_contour = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt, False)
        if area > max_area:
            max_index = i
            max_contour = cnt
            max_area = area

    x_list = []
    y_list = []
    for pos in max_contour:
        x_list.append(pos[0][0])
        y_list.append(pos[0][1])

    img_brank = np.zeros((img_raw.shape[0], img_raw.shape[1], 1), np.uint8)
    img_c = cv2.drawContours(img_brank, contours, max_index, (255, 255, 255), 3)

    upper = min(y_list)
    lower = max(y_list)
    left = min(x_list)
    right = max(x_list)

    img_clip = clip_image(img_c,upper,lower,left,right)

    return img_clip

if __name__ == "__main__":
    main()
