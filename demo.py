# 项目介绍：基于数图像处理的
#
#
#
#
#
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# 灰度图转换
def grayscale(image):
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)

# 高斯滤波
# size为高斯核大小，即高斯滤波器的尺寸；0是高斯标准差，一般默认为0
def gaussian_blur(image, kernel_size):
    return cv.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Canny边缘检测
# low_threshold低阈值； high_threshold高阈值
def canny(edge_image, low_threshold, high_threshold):
    return cv.Canny(image, low_threshold, high_threshold)

#  生成感兴趣区域
# 步骤：1.生成一个与原image大小一致的的mask矩阵，初始化为全0
# 2.对照原图在mask上构建感兴趣区域
# 3.利用opencv中cv.fillpoly()函数对所限定的多边形轮廓进行填充，填充为1，即全白。
# 4.利用opencv中cv.bitwise()函数与canny边缘检测后的图像按位与，保留原图相中对应感兴趣区域内的白色像素值，剔除黑色像素值

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)  # 生成维度与image大小一致的mask矩阵，并为其初始化为全0,即构建出全黑的mask图像

    # 填充顶点vertices中间区域
    if len(image.shape) > 2:    # image.shape是一个1*3的矩阵，len()方法，返回的是矩阵中列元素的个数
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

# cv.fillPoly()用于一个被多边形轮廓所限定的区域内进行填充。mask：图像；vertices：顶点坐标的小数点位数；ignore_mask_color:多边形的颜色
    cv.fillPoly(mask, vertices, ignore_mask_color)
    cv.imshow('maskRIO',mask)
# cv.bitwise_and()函数是对二进制数据进行与操作，即对图像每个像素值进行二进制与操作
# 利用掩膜进行与操作，即掩膜图像白色区域是对需要处理图像像素的保留，黑色区域是对需要处理图像像素的剔除。
    masked_image = cv.bitwise_and(image, mask)
    return masked_image

# 原图像与车道线图像按照a:b比例融合
def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    return cv.addWeighted(initial_img, a, img, b, c)  # cv.addWeighted()函数是将两张相同大小，相同类型的图片融合


# 绘制车道线
def draw_lines(image, lines, color=[0,255,0], thickness=2):

    right_y_set = []
    right_x_set = []
    right_slope_set = []

    left_y_set = []
    left_x_set = []
    left_slope_set = []

    slope_min = .35  # 斜率低阈值
    slope_max = .85  # 斜率高阈值
    middle_x = image.shape[1] / 2  # 图像中线x坐标
    max_y = image.shape[0]  # 最大y坐标

    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)    # 一次多项式拟合相当于线性拟合，即拟合成直线
            slope = fit[0]  # 斜率

            if slope_min < np.absolute(slope) <= slope_max:

                # 将斜率大于0且线段X坐标在图像中线右边的点存为右边车道线
                if slope > 0 and x1 > middle_x and x2 > middle_x:
                    right_y_set.append(y1)
                    right_y_set.append(y2)
                    right_x_set.append(x1)
                    right_x_set.append(x2)
                    right_slope_set.append(slope)

                # 将斜率小于0且线段X坐标在图像中线左边的点存为左边车道线
                elif slope < 0 and x1 < middle_x and x2 < middle_x:
                    left_y_set.append(y1)
                    left_y_set.append(y2)
                    left_x_set.append(x1)
                    left_x_set.append(x2)
                    left_slope_set.append(slope)

    # 绘制左车道线
    if left_y_set:
        lindex = left_y_set.index(min(left_y_set))  # 图像中的最高点，对应坐标系中y的最小值
        left_x_top = left_x_set[lindex]
        left_y_top = left_y_set[lindex]
        lslope = np.median(left_slope_set)   # 计算斜率的平均值

    # 根据斜率计算车道线与图片下方交点作为起点
    left_x_bottom = int(left_x_top + (max_y - left_y_top) / lslope)

    # 绘制线段
    cv.line(image, (left_x_bottom, max_y), (left_x_top, left_y_top), color, thickness)

    # 绘制右车道线
    if right_y_set:
        rindex = right_y_set.index(min(right_y_set))  # 最高点
        right_x_top = right_x_set[rindex]
        right_y_top = right_y_set[rindex]
        rslope = np.median(right_slope_set)

    # 根据斜率计算车道线与图片下方交点作为起点
    right_x_bottom = int(right_x_top + (max_y - right_y_top) / rslope)

    # 绘制线段
    cv.line(image, (right_x_top, right_y_top), (right_x_bottom, max_y), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    # rho：线段以像素为单位的距离精度
    # theta : 像素以弧度为单位的角度精度(np.pi/180较为合适)
    # threshold : 霍夫平面累加的阈值
    # minLineLength : 线段最小长度(像素级)
    # maxLineGap : 最大允许断裂长度
    lines = cv.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def process_image(image):

    kernel_size = 5      # 高斯滤波器大小size
    canny_low_threshold = 75     # canny边缘检测低阈值
    canny_high_threshold = canny_low_threshold * 3    # canny边缘检测高阈值

    rho = 1       # 霍夫像素单位
    theta = np.pi / 180    # 霍夫角度移动步长
    hof_threshold = 20   # 霍夫平面累加阈值threshold
    min_line_len = 30    # 线段最小长度
    max_line_gap = 60    # 最大允许断裂长度

    alpha = 0.8   # 原图像权重
    beta = 1.     # 车道线图像权重
    lambda_ = 0.

    imshape = image.shape  # 获取图像大小，返回的值是H,W,C（H是图像的高，W是图像的宽，C表示图像有几层）
    # print(imshape) #(459, 867, 3) 代表的意思就是该图高459，宽867，有bgr三层

    # 灰度图转换
    gray = grayscale(image)
    # cv.imshow('gray_image', gray)

    # 高斯滤波
    blur_gray = gaussian_blur(gray, kernel_size)
    # cv.imshow('blur_gray_image', blur_gray)

    # Canny边缘检测
    edge_image = canny(blur_gray, canny_low_threshold, canny_high_threshold)
    cv.imshow('edge_image', edge_image)

    # 绘制感兴趣区域
    vertices = np.array([[(0, imshape[0]), (9 * imshape[1] / 20, 11 * imshape[0] / 18),
                          (11 * imshape[1] / 20, 11 * imshape[0] / 18), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edge_image, vertices)

    cv.imshow('masked_edges_image', masked_edges)

    # 基于霍夫变换的直线检测
    lines = hough_lines(masked_edges, rho, theta, hof_threshold, min_line_len, max_line_gap)
    line_image = np.zeros_like(image)   # 构造一个与image矩阵大小维度一致的矩阵，并初始化为全0（纯黑）

    # 绘制车道线线段
    draw_lines(line_image, lines, thickness=10)
    cv.imshow('line_image',line_image)
    # 图像融合
    final_image = weighted_img(image, line_image, alpha, beta, lambda_)
    # cv.imshow('final_image',final_image)
    return final_image

if __name__ == '__main__':
    # cap = cv.VideoCapture("./test_videos/solidYellowLeft.mp4")
    # while(cap.isOpened()):
    #     _, frame = cap.read()
    #     processed = process_image(frame)
    #     cv.imshow("image", processed)
    #     cv.waitKey(1)
    save_path='D:/lane-detection-cv/result-image/'

    # 测试图片的路径
    image = cv.imread('D:/lane-detection-cv/test-image/demo/03605.jpg')
    # 显示原始图像
    #cv.imshow('original_image', image)

    # 显示图像融入的最终效果图
    final_image = process_image(image)
    # 显示检测效果图
    cv.imshow('final_image',final_image)
    cv.waitKey(0)
    # 保存效果图
    cv.imwrite(str(save_path)+'3.jpg',final_image)
    cv.destroyAllWindows()