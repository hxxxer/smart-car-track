import cv2
import numpy as np

from TrackDrawer.utils.predict import LinePredictor
from TrackDrawer.utils.cv2F import (add_black_border,
                        find_contours,
                        get_turn_points,
                        calculate_midpoints,
                        remove_black_border)


def main_process_cv2(image, processed_image, border_size=10):
    """
        处理图像，计算赛道中线并确定赛道类型。

        :return: 处理后的图像和赛道类型
        """
    try:
        # 加黑框
        bordered_image_original = add_black_border(image, border_size)
        bordered_image = add_black_border(processed_image, border_size)
        # cv2.imwrite("temp.jpg", bordered_image)
        # 边缘检测
        # edges = find_edges(bordered_image)
        # 寻找轮廓
        max_contour = find_contours(bordered_image)
        # max_contour_list = max_contour.tolist()
        # with open("temp.txt", "w") as f:
        #     json.dump(max_contour_list, f)
        # 寻找拐点
        turn_points = get_turn_points(max_contour,
                                      lambda x: x <= border_size + 20 or x >= border_size + image.shape[1] - 20,
                                      lambda y: y <= border_size + 20 or y >= border_size + image.shape[0] - 20)
        # 绘制轮廓
        cv2.drawContours(bordered_image_original, [max_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(bordered_image_original, turn_points, -1, (255, 0, 0), 5)
        # temp_img = np.zeros_like(bordered_image)
        # cv2.drawContours(temp_img, max_contour, -1, (255, 255, 255), 1)
        # cv2.imwrite("temp.jpg", temp_img)

        # part_image = get_largest_region(bordered_image, max_contour)
        # 获取中线点
        mid_points = calculate_midpoints(max_contour)
        # mid_points = get_midpoints_from_largest_contour(contours)
        # 拟合中线
        predictor = LinePredictor()
        model, mid_points = predictor.fit_polynomial(mid_points)
        # 绘制中线
        if mid_points is not None:
            cv2.polylines(bordered_image_original, [np.int32(mid_points)], False, (0, 0, 255), 2)
        # 获取赛道类型
        track_type = predictor.determine_track_type()
        # 最后处理
        result_image = remove_black_border(bordered_image_original)

        return result_image, track_type
    except Exception as e:
        return e
