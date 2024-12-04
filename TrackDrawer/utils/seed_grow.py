import cv2
import numpy as np


def find_edges(image):
    """获取与边界有关的一系列信息"""
    height, width = image.shape
    seed_point = width // 2

    upper_half_height = height // 2
    left_lose_count = 0
    right_lose_count = 0
    left_points = []
    right_points = []
    cross_judge = 0

    for y in range(height - 1, 0, -1):
        row = image[y, :]

        # 寻找左右边界点
        left_edge = np.max(np.where(row[:seed_point + 1] == 0)) if 0 in row[:seed_point + 1] else None
        right_edge = seed_point + np.min(np.where(row[seed_point:] == 0)) if 0 in row[seed_point:] else None

        # 寻找左右丢边和左右边线
        left_points.append((0, y) if left_edge is None else (left_edge, y))
        left_lose_count += (1 if y < upper_half_height and left_edge is None else 0)

        right_points.append((width - 1, y) if right_edge is None else (right_edge, y))
        right_lose_count += 1 if y < upper_half_height and right_edge is None else 0

        # 更新种子点
        seed_point = ((0 if left_edge is None else left_edge) + (width - 1 if right_edge is None else right_edge)) // 2

        # 根据是否同时存在左右丢边判断十字路口
        cross_judge = 1 if left_edge is None and right_edge is None else cross_judge

    return left_points, right_points, left_lose_count, right_lose_count, cross_judge


def find_turning_points(points: list, turn_threshold=5):
    """寻找拐点"""
    points_np = np.array(points)
    diffs: np.ndarray = np.abs(points_np[:-1, 0] - points_np[1:, 0])

    upper_turn = points[np.max(np.where(diffs > turn_threshold)) + 1] if np.any(diffs > turn_threshold) else tuple()
    lower_turn = points[np.min(np.where(diffs > turn_threshold))] if np.any(diffs > turn_threshold) else tuple()

    return upper_turn, lower_turn


def get_mid_points_4turn(left_points, right_points, left_upper_turn, left_lower_turn, right_upper_turn, right_lower_turn):
    """4个拐点情况下，获取左右边线"""
    def generate_line(p1, p2, points: np.ndarray):
        points = points[::-1]
        """p1 < p2"""
        y1, y2 = points[p1][1], points[p2][1]
        x1, x2 = points[p1][0], points[p2][0]

        y_points = np.arange(y1, y2)
        x_points = np.linspace(x1, x2, len(y_points), dtype=int)

        line_points = list(zip(x_points, y_points))

        points = np.delete(points, np.s_[p1:p2], axis=0)
        points = np.insert(points, p1, line_points, axis=0)

        return points.tolist()

    left_points = generate_line(left_upper_turn[1], left_lower_turn[1], np.array(left_points))
    right_points = generate_line(right_upper_turn[1], right_lower_turn[1], np.array(right_points))

    return left_points, right_points


def get_mid_points_0turn(left_points, right_points):
    """无拐点情况下获取左右边线"""
    return left_points, right_points


def get_mid_points(left_points, right_points, *args):
    """根据传入拐点数量不同，选择不同的中线函数"""
    func_dict = {
        0: get_mid_points_0turn,
        4: get_mid_points_4turn
    }

    if len(args) in func_dict:
        left_points, right_points = func_dict[len(args)](left_points, right_points, *args)

        mid_points = [((left_edge[0] + right_edge[0]) // 2, left_edge[1]) for left_edge, right_edge in zip(left_points, right_points)]

        return mid_points
    else:
        print(f"Unsupported number of arguments: {len(args)}")


def determine_race_type(left_lose_count, right_lose_count, cross_judge):
    """根据左右丢边数和十字路口判断道路类型"""
    if cross_judge == 0:
        if left_lose_count > right_lose_count:
            return "Left Turn"
        elif right_lose_count > left_lose_count:
            return "Right Turn"
        else:
            return "Straight"
    else:
        return "Cross"
