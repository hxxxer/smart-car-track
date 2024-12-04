import cv2
import numpy as np

from TrackDrawer.utils.seed_grow import (find_turning_points,
                                         find_edges,
                                         get_mid_points,
                                         determine_race_type)


def main_process_seed_grow(image, processed_image):
    try:
        # 获取左右边线、左右丢边数、是否为十字路口
        left_points, right_points, left_lose_count, right_lose_count, cross_judge = find_edges(processed_image)

        # 获取四个拐点
        left_upper_turn, left_lower_turn = find_turning_points(left_points)
        right_upper_turn, right_lower_turn = find_turning_points(right_points)

        # 根据是否为十字路口以及四个拐点是否存在，获取中线
        if cross_judge and all(len(t) for t in [left_upper_turn, left_lower_turn, right_upper_turn, right_lower_turn]):
            mid_points = get_mid_points(left_points, right_points, left_upper_turn,
                                        left_lower_turn, right_upper_turn, right_lower_turn)
        else:
            cross_judge = False
            mid_points = get_mid_points(left_points, right_points)

        # 判断道路类型
        race_type = determine_race_type(left_lose_count, right_lose_count, cross_judge)

        # 绘制左右轮廓、拐点
        cv2.polylines(image, (np.array(left_points + right_points, dtype=int)).reshape(-1, 1, 2), True, (0, 255, 0), 2)
        cv2.polylines(image,
                      (np.array([left_upper_turn + left_lower_turn + right_upper_turn + right_lower_turn],
                                dtype=int)).reshape(-1, 1, 2),
                      True, (255, 0, 0), 5)
        # 绘制中线
        if mid_points is not None:
            cv2.polylines(image, [np.int32(mid_points)], False, (0, 0, 255), 2)

        return image, race_type
    except Exception as e:
        return e
