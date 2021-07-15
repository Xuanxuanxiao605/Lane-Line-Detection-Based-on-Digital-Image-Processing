import numpy as np
import cv2 as cv

# 对图像预处理，使用canny边缘检测处理图像
def do_canny_detection(frame):
    # RGB 转灰度图
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # 高斯滤波器
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    # Canny 处理
    canny_detection = cv.Canny(blur, 50, 150)

    # 在canny检测结果中划分感兴趣区域
    # imshape = frame.shape
    # polygons = np.array([[(2 * imshape[1] / 11, 10 * imshape[0] / 11), (5 * imshape[1] / 13, 5 * imshape[0] / 11),
    #                       (7 * imshape[1] / 13, 5 * imshape[0] / 11), (8 * imshape[1] / 11, 10 * imshape[0] / 11)]],
    #                     dtype=np.int32)  # 对应左下、左上、右上、右下
    # mask = canny_detection
    # cv.polylines(mask, [polygons], True, (255, 255, 255), 3)
    # cv.imshow('mask', mask)
    # cv.waitKey(0)
    return canny_detection

 # 提取感兴趣区域:感兴趣区域的设置为梯形
def do_segment(frame):
        # height = frame.shape[0]
        # width = frame.shape[1]
        # polygons = np.array([[(0, height), (width, height), (int(width/2), int(height/2))]])
        imshape=frame.shape
        polygons = np.array([[(2*imshape[1]/11, 10*imshape[0]/11), (5 * imshape[1] / 13, 5* imshape[0] / 11),
                   (7 * imshape[1] / 13, 5 * imshape[0] / 11), (8*imshape[1]/11, 10*imshape[0]/11)]], dtype=np.int32)#对应左下、左上、右上、右下
        mask = np.zeros_like(frame)
        # 在全黑 mask 上画白色多边形（内部填充）
        cv.fillPoly(mask, polygons, 255)
        # cv.imshow('mask',mask)
        # 按位与，保留感兴趣的区域
        segment = cv.bitwise_and(frame, mask)
        return segment
# 原图上划分感兴趣区域
# def do_seg(frame):
#     imshape = frame.shape
#     polygons = np.array([[(2 * imshape[1] / 11, 10 * imshape[0] / 11), (5 * imshape[1] / 13, 5 * imshape[0] / 11),
#                           (7 * imshape[1] / 13, 5 * imshape[0] / 11), (8 * imshape[1] / 11, 10 * imshape[0] / 11)]],
#                         dtype=np.int32).reshape((-1,1,2))  # 对应左下、左上、右上、右下
#     mask = frame
#     cv.polylines(mask,[polygons],True,(0,255,0),3)
#     cv.imshow('mask', mask)
#     cv.waitKey(0)
# 霍夫变换提取车道线参数
def do_hough(frame):
    lines = cv.HoughLinesP(frame, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)
    parameter_slope = []     #记录符合斜率条件的直线
    parameter_all = []       #记录霍夫直线检测到的直线信息
    parameter = []   # 记录筛选过后直线信息
    parameters1 = []  #记录左右两条车道线的四条边缘线
    parameters = []   #记录平均后两条车道线的直线信息
    mask = np.zeros_like(frame)
    mask1 = np.zeros_like(frame)
    mask2 = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            parameter_all.append([slope, x1, y1, x2, y2])
            # 1.根据斜率筛选直线
            # if -20 <= slope <= -0.5 or 0.5 <= slope <= 20:
            if -20 <= slope <= -0.5 or 0.5 <= slope <= 20:
                parameter_slope.append([slope, x1, y1, x2, y2])
        parameter = sorted(parameter_slope)
        parameter_all = sorted(parameter_all)
        m = len(parameter_slope)  # 记录霍夫直线检测直线条数n 及控制斜率后的直线斜率 m
        n = len(parameter_all)
        ####################################################################################################
        #  可视化
        #  绘制霍夫直线检测到的直线
        for n in range(len(parameter_all)):
            polygons_all = np.array([(parameter_all[n][1], parameter_all[n][2]), (parameter_all[n][3], parameter_all[n][4])])
            cv.polylines(mask, [polygons_all], False, (255, 255, 255))
        cv.imshow('all_lines', mask)
        #  绘制斜率控制后的直线
        for n in range(len(parameter_slope)):
            polygons = np.array([(parameter_slope[n][1], parameter_slope[n][2]), (parameter_slope[n][3], parameter_slope[n][4])])
            cv.polylines(mask1, [polygons], False, (255, 255, 255))
        cv.imshow('slope_lines', mask1)

        # 2.根据车道线的几何特征进行边缘筛选
        if m >= 2:
            if parameter[0][0] < 0 and parameter[1][0] < 0:  # and parameter[1][0]-parameter[0][0] < 0.1: 左车道线
                parameters1.append([parameter[0][1],parameter[0][2],parameter[0][3],parameter[0][4]])
                parameters1.append([parameter[1][1], parameter[1][2], parameter[1][3], parameter[1][4]])
                slope = (parameter[0][0] + parameter[1][0]) / 2
                x1 = int((parameter[0][1] + parameter[1][1]) / 2)
                y1 = int((parameter[0][2] + parameter[1][2]) / 2)
                x2 = int((parameter[0][3] + parameter[1][3]) / 2)
                y2 = int((parameter[0][4] + parameter[1][4]) / 2)
                parameters.append([slope, x1, y1, x2, y2])
            if parameter[n - 2][0] > 0 and parameter[n - 1][0] > 0:  # and parameter[i-1][0]-parameter[i-2][0] < 0.1: 右车道线
                parameters1.append([parameter[n-2][1], parameter[n-2][2], parameter[n-2][3], parameter[n-2][4]])
                parameters1.append([parameter[n - 1][1], parameter[n - 1][2], parameter[n - 1][3], parameter[n - 1][4]])
                slope = (parameter[n - 2][0] + parameter[n - 1][0]) / 2
                x1 = int((parameter[n - 2][1] + parameter[n - 1][1]) / 2)
                y1 = int((parameter[n - 2][2] + parameter[n - 1][2]) / 2)
                x2 = int((parameter[n - 2][3] + parameter[n - 1][3]) / 2)
                y2 = int((parameter[n - 2][4] + parameter[n - 1][4]) / 2)
                parameters.append([slope, x1, y1, x2, y2])  # 存放右左车道线的斜率及点坐标值

        # 绘制筛选后的四条车道边缘线
        for i in range(len(parameters1)):
            polygons2 = np.array([(parameters1[i][0], parameters1[i][1]), (parameters1[i][2], parameters1[i][3])])
            cv.polylines(mask2, [polygons2], False, (255, 255, 255))
        cv.imshow('four_lines', mask2)

        return parameters

# 绘制车道线
def draw_lines(frame,parameters, color=[0,0,255], thickness=2):
    #绘制左车道线
    left_x1_set = parameters[0][1]
    left_y1_set = parameters[0][2]
    left_x2_set = parameters[0][3]
    left_y2_set = parameters[0][4]
    cv.line(frame,(left_x1_set,left_y1_set),(left_x2_set,left_y2_set),color, thickness)

    # 绘制右车道线
    right_x1_set = parameters[1][1]
    right_y1_set = parameters[1][2]
    right_x2_set = parameters[1][3]
    right_y2_set = parameters[1][4]
    cv.line(frame, ( right_x1_set,  right_y1_set), ( right_x2_set,  right_y2_set), color, thickness)

def  weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    return cv.addWeighted(initial_img, a, img, b, c)

# 虚实线检测
def solid_dotted_judge(frame, lines):
    if len(lines) > 0:
        for line in lines:
            slope, x1, y1, x2, y2 = line
            k = abs(y2 - y1)
            font = cv.FONT_HERSHEY_SIMPLEX
            # 右车道线
            if slope >0:
                if k >= 60:
                    cv.putText(frame, 'Right Solid', (int((x1 + x2) / 2)+50, int((y1 + y2) / 2)), font, 0.5, (0, 255, 0), 2)
                else:
                    cv.putText(frame, 'Right Dotted', (int((x1 + x2) / 2)+50, int((y1 + y2) / 2)), font, 0.5, (0, 255, 0), 2)
            # 左车道线
            else:
                if k >= 60:
                    cv.putText(frame, 'Left Solid', (int((x1 + x2) / 2)+50, int((y1 + y2) / 2)), font, 0.5, (0, 255, 0), 2)
                else:
                    cv.putText(frame, 'Left Dotted', (int((x1 + x2) / 2)+50, int((y1 + y2) / 2)), font, 0.5, (0, 255, 0), 2)

# 黄白线检测
def color_judge(frame, frame1, lines):
    frame2 = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    L = []
    if len(lines) > 0:
        for i in range(len(lines)):
            L.append(lines[i])
        for line in L:
            # print(line)
            slope, x1, y1, x2, y2= line
            p = 0
            for i in range(1, 11):
                y = int(y1 + (y2 - y1) * i / 11)
                x = int(x1 + (x2 - x1) * i / 11)
                # print(x, y)
                h, s, v = frame2[y][x]
                if s > 40:
                    p += 1
            if p >= 3:
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(frame1, 'Yellow', (int((x1 + x2) / 2), int((y1 + y2) / 2)), font, 0.5, (0, 255, 255), 2)
            else:
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(frame1, 'White', (int((x1 + x2) / 2), int((y1 + y2) / 2)), font, 0.5, (255, 255, 255), 2)
# 处理图片
def process_image(image):

    alpha = 0.8  # 原图像权重
    beta = 1.  # 车道线图像权重
    lambda_ = 0.

    # canny边缘检测
    canny_image = do_canny_detection(image)
    cv.imshow('do_canny_image', canny_image)

    # 提取感兴趣区域
    ROI_image = do_segment(canny_image)
    cv.imshow('do_segment_image', ROI_image)


    # 霍夫变换提取出车道线参数
    lines = do_hough(ROI_image)
    # 生成与原图大小一致的纯黑图片
    line_image = np.zeros_like(image)

    # 绘制车道线线段
    draw_lines(line_image, lines, thickness=10)
    cv.imshow('do_hough_image', line_image)



    # 图像融合
    final_image = weighted_img(image, line_image, alpha, beta, lambda_)
    # 虚实线的判断
    solid_dotted_judge(final_image,lines)
    # 黄白线判断
    color_judge(image,final_image,lines)

    # 显示检测效果图
    cv.imshow('final_image', final_image)
    cv.waitKey(0)
    # 保存效果图
    cv.imwrite(str(save_path) + str(canny_name), canny_image)
    cv.imwrite(str(save_path) + str(roi_name), ROI_image)
    cv.imwrite(str(save_path) + str(hough_name), line_image)
    cv.imwrite(str(save_path) + str(final_name), final_image)
    cv.destroyAllWindows()


if __name__ == '__main__':

    save_path = 'D:/lane-detection-cv/demo1_result_image/simple1/'
    image_path='D:/lane-detection-cv/test-image/simple1/'
    img='02100'   # 输入测试图片名称
    image_name=img+'.jpg'
    canny_name = img+'do_canny.jpg'
    roi_name = img+'do_roi.jpg'
    hough_name = img+'do_hough.jpg'
    final_name = img+'result_img.jpg'
    # 指定测试图片路径
    image = cv.imread(image_path+'/'+image_name)
    # 原图显示roi
    # do_seg(image)
    # 处理图片
    process_image(image)

