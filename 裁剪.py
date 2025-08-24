def crop_center(image, crop_width, crop_height):
    h, w = image.shape[:2]
    start_x = (w - crop_width) // 2
    start_y = (h - crop_height) // 2
    cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
    return cropped_image
# 使用示例
cropped = crop_center(img, 300, 200)



