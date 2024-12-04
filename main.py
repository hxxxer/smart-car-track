import time

from TrackDrawer.track_drawer import batch_process_images, ProcessMethods
from pathlib import Path


def get_jpg_files(directory):
    directory_path = Path(directory)
    jpg_files = [str(file) for file in directory_path.rglob('*.jpg') if not file.name.lower().endswith('_deal.jpg')]
    return jpg_files


if __name__ == "__main__":
    st = time.time()

    # 指定图片路径列表和输出目录
    file_dir = "track_data/GRAY"
    image_paths = get_jpg_files(file_dir)
    output_dir = "deal_image"
    # image_paths = [
    #     # r"track_data/GRAY/ring_left/ring_left1.jpg",
    #     # r"track_data/GRAY/cross/cross_middle/cross_arrive.jpg",
    #     # r"t1.jpg"
    #     r"track_data\GRAY\straightway\straightway_right.jpg"
    # ]
    # output_dir = ""

    # 调用批量处理函数
    batch_process_images(image_paths, ProcessMethods.SEED_GROW, output_dir)
    # batch_process_images(image_paths, ProcessMethods.CV2, output_dir)

    et = time.time()
    print("Time used: ", et - st)
