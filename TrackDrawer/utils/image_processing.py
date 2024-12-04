import cv2


def load_image(image_path):
    """加载图像并检查是否成功"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image


def preprocess_image(image):
    """预处理图像，包括转换为灰度图、应用中值滤波、二值化处理等"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_blur = cv2.medianBlur(gray, 5)
    _, binary = cv2.threshold(median_blur, 150, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return opening


def add_type_text(image, track_type):
    # 获取图像的高度和宽度
    height, width, _ = image.shape
    # font = cv2.FONT_ITALIC
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 0)
    thickness = 2

    # 计算文本的大小
    (text_width, text_height), _ = cv2.getTextSize(track_type, font, font_scale, thickness)

    # 计算文本的起始位置，使其位于图像底部正中间
    x = (width - text_width) // 2
    y = height - 10  # 距离底部10个像素

    # 在图像上添加文本
    cv2.putText(image, track_type, (x, y), font, font_scale, font_color, thickness)

    return image
