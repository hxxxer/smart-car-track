import cv2
import numpy as np

from TrackDrawer.utils import TURN_COS_THRESHOLD


def add_black_border(image, border_size=10):
    """给图像添加黑色边框"""
    return cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                              value=[0, 0, 0])


def find_edges(image):
    """边缘检测"""
    # 使用Canny算子进行边缘检测
    edges = cv2.Canny(image, 50, 150)

    # 形态学操作改善边缘
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    return edges


def find_contours(image):
    """找到隐形眼镜的轮廓"""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 选择最大的轮廓
    if contours:
        lens_contour = max(contours, key=cv2.contourArea)
        return lens_contour
    return None
    # return contours


def get_turn_points(contour: np.ndarray, x_threshold_func, y_threshold_func):
    contour_points = contour.reshape(-1, 2)
    turn_points = np.empty((0, 2), dtype=contour.dtype)
    n = len(contour_points)

    def calculate_angle(v1, v2):
        """计算两个向量之间的夹角的余弦值。"""
        # 向量的点积
        dot_product = np.dot(v1, v2)
        # 向量的模
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        # 计算cosine值
        cosine_theta = dot_product / (norm_v1 * norm_v2)
        # 确保cosine值在有效范围内
        cosine_theta = max(min(cosine_theta, 1.0), -1.0)
        return cosine_theta

    for i in range(n):
        if x_threshold_func(contour_points[i][0]) or y_threshold_func(contour_points[i][1]):
            continue
        # 取点
        p_i_minus_2 = contour_points[(i - 7) % n]
        p_i_minus_1 = contour_points[(i - 1) % n]
        p_i_plus_1 = contour_points[(i + 1) % n]
        p_i_plus_2 = contour_points[(i + 7) % n]

        l1 = p_i_minus_2 - p_i_minus_1
        l2 = p_i_plus_2 - p_i_plus_1
        # 计算斜率
        angle_cos = calculate_angle(l1, l2)
        # print(angle_cos)

        # 比较斜率
        if angle_cos >= TURN_COS_THRESHOLD:
            turn_points = np.vstack((turn_points, contour_points[i]))

    def remove_close_points(points, threshold):
        # 计算所有点对之间的距离
        distances = np.sum(np.abs(points[:, np.newaxis] - points[np.newaxis, :]), axis=2)

        # 初始化一个标记数组，用于标记是否保留该点
        keep = np.ones(len(points), dtype=bool)

        # 遍历所有点
        for i in range(len(points)):
            if keep[i]:
                # 找到与当前点距离小于阈值的其他点
                close_points = np.where(distances[i] < threshold)[0]

                # 保留一个点，其他点标记为不保留
                keep[close_points] = False
                keep[i] = True  # 保留当前点

        return points[keep]

    turn_points = remove_close_points(turn_points, 30)
    turn_points = np.reshape(turn_points, (-1, 1, 2))
    return turn_points


def get_largest_region(image, max_contour):
    """
    根据最大轮廓获取只有相应赛道为白色的图像。

    参数:
    image (numpy.ndarray): 原始图像。
    max_contour (numpy.ndarray): 最大面积的轮廓，由cv2.findContours找到并筛选出。

    返回:
    numpy.ndarray: 相应赛道为白色的图像。
    """

    # 创建一个与原图大小相同的空白掩码
    mask = np.zeros_like(image)
    # 在掩码上填充最大轮廓
    cv2.fillPoly(mask, [max_contour], (255, 255, 255))

    return mask


def calculate_midpoints(contour: np.ndarray):
    """
    计算赛道中线点
    :param
    contour (numpy.ndarray)
    """
    # mid_points = []
    # for y in range(edges.shape[0]):
    #     row = edges[y, :]
    #     indices = np.where(row == 255)[0]
    #     if len(indices) > 0:
    #         x_min = indices.min()
    #         x_max = indices.max()
    #         mid_x = (x_min + x_max) // 2
    #         mid_points.append((mid_x, y))
    # return mid_points
    contour_points = contour.reshape(-1, 2)
    # 按y坐标分组点，并记录每行的最大和最小x坐标
    points_by_row = {}
    for x, y in contour_points:
        if y not in points_by_row:
            points_by_row[y] = {'min_x': x, 'max_x': x}
        else:
            # 更新当前行的最小和最大x坐标
            points_by_row[y]['min_x'] = min(x, points_by_row[y]['min_x'])
            points_by_row[y]['max_x'] = max(x, points_by_row[y]['max_x'])

    # 计算每行的中点
    midpoints = []
    for y, values in points_by_row.items():
        # 计算x坐标的平均值作为中点
        midpoint_x = (values['min_x'] + values['max_x']) / 2
        midpoints.append((midpoint_x, y))

    return midpoints


def remove_black_border(image, border_size=10):
    """移除之前添加的黑色边框"""
    return image[border_size:-border_size, border_size:-border_size]
