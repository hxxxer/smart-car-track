from enum import Enum, auto

from .cv2_contour import main_process_cv2
from .seed_grow import main_process_seed_grow
from .utils.image_processing import (
    load_image,
    preprocess_image,
    add_type_text,
)
import cv2
import os


class ProcessMethods(Enum):
    CV2 = auto()
    SEED_GROW = auto()


process_methods = {
    ProcessMethods.CV2: main_process_cv2,
    ProcessMethods.SEED_GROW: main_process_seed_grow,
}


class TrackDrawer:
    def __init__(self, image_path, method, border_size=10):
        """
        初始化 TrackDrawer 类。

        :param image_path: 图像文件的路径
        """
        self.image_path = image_path
        self.method = method
        self.border_size = border_size
        # self._process_methods = {
        #     ProcessMethods.CV2: main_process_cv2,
        #     ProcessMethods.SEED_GROW: main_process_seed_grow,
        # }

    def all_process_image(self):
        try:
            if self.method not in process_methods:
                raise AttributeError("方法不存在")
            # 载入图片
            image = load_image(self.image_path)
            # 预处理图片
            processed_image = preprocess_image(image)

            result_image, text = process_methods[self.method](image, processed_image)
            print(f"图片 {self.image_path} 类型 : {text}")

            # 添加文字
            result_image = add_type_text(result_image, text)

            return result_image
        except Exception as e:
            print(f"图片 {self.image_path} 发生错误 : {e}")
            return None

    def save_result(self, result_image, output_dir):
        """
        保存处理后的图像。

        :param result_image: 处理后的图像
        :param output_dir: 输出目录
        """
        if result_image is not None:
            try:
                name, ext = os.path.splitext(os.path.basename(self.image_path))
                output_file_name = f"{name}_deal{ext}"
                output_path = os.path.join(output_dir, output_file_name)
                cv2.imwrite(output_path, result_image)
            except Exception as e:
                print(f"图片 {self.image_path} 保存发生错误 ：{e}")
                return None


def batch_process_images(image_paths, method, output_dir=""):
    """
    批量处理图片。

    :param method: 处理方法
    :param image_paths: 包含图片路径的列表
    :param output_dir: 输出目录
    """
    no_output_path = output_dir == ""

    if not no_output_path and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_path in image_paths:
        drawer = TrackDrawer(image_path, method)

        result_image = drawer.all_process_image()
        if result_image is None:
            print("处理出错：", image_path)
            continue
        if no_output_path:
            drawer.save_result(result_image, os.path.dirname(image_path))
        else:
            drawer.save_result(result_image, output_dir)
