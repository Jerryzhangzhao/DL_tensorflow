import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

'''
# resize and crop
tf.image.resize_bicubic
tf.image.central_crop
tf.image.crop_and_resize
tf.image.random_crop()
tf.image.pad_to_bounding_box

# adjust color
tf.image.adjust_brightness
tf.image.adjust_hue()
tf.image.adjust_contrast()
tf.image.adjust_gamma
tf.image.adjust_saturation()

# encode and decode
tf.image.encode_png
tf.image.decode_bmp

# flip
tf.image.flip_left_right
tf.image.flip_up_down
tf.image.rot90

tf.image.non_max_suppression()

# color chanel transform
tf.image.yuv_to_rgb()
tf.image.rgb_to_hsv

# draw
tf.image.draw_bounding_boxes

tf.image.transpose_image
'''


def image_process(image, operation="crop", code=0):
    if operation == "crop":
        image_result = tf.image.central_crop(image, central_fraction=0.5)

    if operation == "rgb2hsv":
        image = tf.to_float(image)
        image_result = tf.image.rgb_to_hsv(image)

    if operation == "flip":
        if code == 0:
            image_result = tf.image.flip_up_down(image)
        elif code == 1:
            image_result = tf.image.flip_left_right(image)

    if operation == "resize":
        if code == 0:
            image_result = tf.image.resize_images(image, (100, 100))
            # image_result=tf.to_int32(image_result)
            image_result = tf.cast(image_result, dtype=tf.uint8)

    return image_result


with tf.Session() as sess:
    image_raw_data = tf.gfile.GFile("images/panda.png", "rb").read()
    image = tf.image.decode_png(image_raw_data, )

    image_result = image_process(image, "resize")
    plt.imshow(image_result.eval())
    plt.show()

    # load image using PIL.Image
    # im = Image.open("images/panda.png")
    # plt.imshow(im)
    # plt.show()
