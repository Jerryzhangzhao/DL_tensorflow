import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def image_process(image, operation="crop", code=0):
    if operation == "crop":
        if code == 0:
            image_result = tf.image.central_crop(image, central_fraction=0.5)
        elif code == 1:
            image_result = tf.random_crop(image, [200, 200, 2], 3)

    elif operation == "rgb2xxx":
        if code == 0:
            image = tf.to_float(image)
            image_result = tf.image.rgb_to_hsv(image)
        elif code == 1:
            image_result = tf.image.rgb_to_grayscale(image)
        # elif code == 2:
        # image_result = tf.image.rgb_to_yuv(image)

    elif operation == "flip":
        if code == 0:
            image_result = tf.image.flip_up_down(image)
        elif code == 1:
            image_result = tf.image.flip_left_right(image)
        elif code == 2:
            image_result = tf.image.rot90(image, k=1)
        elif code == 3:
            image_result = tf.image.transpose_image(image)

    elif operation == "resize":
        if code == 0:
            image_result = tf.image.resize_images(image, (100, 100), method=tf.image.ResizeMethod.BILINEAR)
            # image_result=tf.to_int32(image_result)
            image_result = tf.cast(image_result, dtype=tf.uint8)
        elif code == 1:
            image_result = tf.image.resize_image_with_crop_or_pad(image, 400, 550)

    elif operation == "adjust":
        if code == 0:
            image_result = tf.image.adjust_saturation(image, 0.5)
        if code == 1:
            image_result = tf.image.adjust_hue(image, 0.01)
        if code == 2:
            image_result = tf.image.adjust_brightness(image, 0.5)
        if code == 3:
            image_result = tf.image.adjust_contrast(image, 0.5)

    elif operation == "draw_boxes":
        # `boxes` are encoded as `[y_min, x_min, y_max, x_max]` format[batch,number,box]
        # expand_dims: insert a batch dimension
        image1 = tf.expand_dims(tf.image.convert_image_dtype(image, tf.float32), 0)
        boxes = [[[0.2, 0.2, 0.5, 0.7], [0.3, 0.3, 0.8, 0.6]]]
        images_result = tf.image.draw_bounding_boxes(image1, boxes)
        image_result = images_result[0]

    else:
        image_result = image
    return image_result


with tf.Session() as sess:
    image_raw_data = tf.gfile.GFile("images/panda.png", "rb").read()
    image = tf.image.decode_png(image_raw_data)
    image_shape = tf.shape(image)

    print("original image shape: ", image_shape.eval(), " dtype: ", image.dtype)

    operation = "rgb2xxx"
    code = 1
    image_processed = sess.run(image_process(image, operation, code))

    processed_shape = tf.shape(image_processed)
    print("processed image shape: ", processed_shape.eval(), " dtype: ", image_processed.dtype)

    plt.subplot(221)
    plt.imshow(image.eval())
    if operation == "rgb2xxx" and code == 1:
        plt.subplot(222)
        plt.imshow(image_processed[:, :, 0], plt.cm.gray)
    else:
        plt.subplot(222)
        plt.imshow(image_processed, plt.cm.gray)

    plt.show()

    # load image using PIL.Image
    # im = Image.open("images/panda.png")
    # plt.imshow(im)
    # plt.show()
