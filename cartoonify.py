###Joel van der Boom 209668862 UN:joelvdb
###ex5

import copy, math
import ex5_helper as helper
import sys


def rotate_90_degrees_clockise(arr):
    """
    :param arr: list
    :return: list rotated 90_degress_clockwise
    """
    rows = len(arr)
    cols = len(arr[0])
    arr2 = set_new_matrix(rows, cols)
    for x in range(rows):
        for y in range(cols):
            arr2[y][rows - x - 1] = arr[x][y]
    return arr2


def quantize_pixel(pixel_value, n):
    """
    :param pixel_value: pixel
    :param n: range of colors
    :return: the color in range
    """
    qpixel = (math.floor((pixel_value * n) / 255)) * 255 / n
    return round(qpixel)


def set_new_matrix(a: int, b: int, c=0, def_value=-1):
    """
    :param a: size of dimension in list
    :param b:size of dimension in list
    :param c:size of dimension in list default=0
    :param def_value: value to put in list default=-1
    :return: new list
    """
    if c == 0:
        return [[def_value for j in range(a)] for i in range(b)]
    return [[[def_value for j in range(a)] for i in range(b)] for k in range(c)]


def get_red_constant():
    return 0.299


def get_blue_constant():
    return 0.114


def get_green_constant():
    return 0.587


def screen_arr(arr1, kernel_arr):
    """
    :param arr1: squared list
    :param karnel_arr: kerenel list
    :return: the avg of the list after the kernel is screened
    """
    sum = 0
    for index_row, row in enumerate(arr1):
        for index_column, column in enumerate(row):
            arr1[index_row][index_column] = arr1[index_row][index_column] * kernel_arr[index_row][index_column]
            sum += arr1[index_row][index_column]
    if (sum > 255):
        return 255
    if (sum < 0):
        return 0
    return round(sum)


def get_new_list_main_in_middle(kernel, image, cell, index_cell_x, index_cell_y, distance_from_main):
    """

    :param kernel: list
    :param image: list
    :param cell: value 0f pixel
    :param index_cell_x:
    :param index_cell_y:
    :param distance_from_main: the distance from the cell
    :return:get the new pixel value after the kernel is screened
    """
    demo_array = set_new_matrix(len(kernel), len(kernel))
    for i in range(len(kernel)):
        for j in range(len(kernel)):
            x = i - distance_from_main + index_cell_x
            y = j - distance_from_main + index_cell_y
            if x < 0 or y < 0 or x > len(image) - 1 or y > len(
                    image[0]) - 1:  # if the screen is out of range the value will be the value of cell
                demo_array[i][j] = cell
            else:
                demo_array[i][j] = image[x][y]

    return demo_array


def separate_channels(image):
    """
    :param image: list 3d
    :return: list seperated that each color R/G/B in diffrent list
    """
    rows = len(image)
    columns = len(image[0])
    channels = len(image[0][0])
    demo_array = set_new_matrix(columns, rows, channels)
    for channel in range(channels):
        for row in range(rows):
            for column in range(columns):
                demo_array[channel][row][column] = image[row][column][channel]

    return demo_array


def combine_channels(channels):
    """
    :param channels: 3 same sized 2d lists
    :return: colored image list
    """
    rows = len(channels[0])
    columns = len(channels[0][0])
    channels_num = len(channels)  # rgb
    demo_array = set_new_matrix(channels_num, columns, rows)
    for channel in range(channels_num):
        for row in range(rows):
            for column in range(columns):
                demo_array[row][column][channel] = channels[channel][row][column]

    return demo_array


def RGB2grayscale(colored_image):
    """
    :param colored_image: 3d list
    :return: 2d list- black and white
    """
    new_image = set_new_matrix(len(colored_image[0]), len(colored_image))
    for index_r, row in enumerate(colored_image):
        for index_c, column in enumerate(row):
            red = column[0] * get_red_constant()
            green = column[1] * get_green_constant()
            blue = column[2] * get_blue_constant()
            new_image[index_r][index_c] = round(red + green + blue)
    return new_image


def blur_kernel(size):
    """
    :param size: size of kernel
    :return: 2d list sized sizeXsize all same values 1 / (size ** 2)
    """
    sqr = 1 / (size ** 2)
    arr = set_new_matrix(abs(size), abs(size), 0, sqr)
    return arr


def apply_kernel(image, karnel):
    """
    # 1.move through each pixel
    # 2.get the blured value of it
    # 3.put it in an array that should be the same size
    :param image: list
    :param karnel: kernel list
    :return: blured image list
    """
    new_image = set_new_matrix(len(image[0]), len(image))
    distance_from_center = len(karnel) // 2
    for index_row, row in enumerate(image):
        for index_cell, cell in enumerate(row):
            new_cube_list_main_in_middle = get_new_list_main_in_middle(karnel, image, cell, index_row, index_cell,
                                                                       distance_from_center)
            screened_pixel = screen_arr(new_cube_list_main_in_middle, karnel)
            new_image[index_row][index_cell] = screened_pixel
    return new_image


def bilinear_interpolation(image, y, x):
    """
    :param image: 2d list
    :param y:
    :param x:
    :return: the "avg" of 4 points depends on the x y ratio
    """
    x_zero = round(x * (len(image[0]) - 2))
    y_zero = round(y * (len(image) - 2))
    a = image[y_zero][x_zero]
    d = image[y_zero + 1][x_zero + 1]
    b = image[y_zero + 1][x_zero]
    c = image[y_zero][x_zero + 1]
    total = round(a * (1 - x) * (1 - y) + b * y * (1 - x) + c * x * (1 - y) + d * x * y)
    return total


def resize(image, new_height, new_width):
    """
    :param image: 2d list
    :param new_height:
    :param new_width:
    :return: 2d list (image) in new size
    """
    copy_image = copy.deepcopy(image)
    new_image = set_new_matrix(new_width, new_height)
    for index_row, row in enumerate(new_image):
        for index_column, column in enumerate(row):
            new_image[index_row][index_column] = bilinear_interpolation(copy_image, (index_row / (new_height - 1)),
                                                                        index_column / (new_width - 1))
    return new_image


def rotate_90(image, direction):
    """
    :param image: list
    :param direction: the desired direction to rotate
    :return: rotated image
    """
    rotated_image = []
    if direction == 'R':
        rotated_image = rotate_90_degrees_clockise(image)
    elif direction == 'L':
        rotated_image = rotate_90_degrees_clockise(rotate_90_degrees_clockise(rotate_90_degrees_clockise(image)))
    return rotated_image


def screen_threshold(list, block_size):
    r = block_size ** 2 - 1
    list_copy = copy.deepcopy(list)
    sum = 0
    for index_row, row in enumerate(list):
        for index_column, column in enumerate(row):
            if not (index_row == index_column and index_row == block_size // 2):
                list_copy[index_row][index_column] = list[index_row][index_column] * (1 / r)
                sum += list_copy[index_row][index_column]
    return round(sum)


def set_threshold(image, block_size):
    threshold = copy.deepcopy(image)
    for index_row in range(len(image)):
        for index_column in range(len(image[0])):
            new_cell_list = get_new_list_main_in_middle(blur_kernel(block_size), image, image[index_row][index_column],
                                                        index_row, index_column,
                                                        (block_size // 2))
            new_cell = screen_threshold(new_cell_list, block_size)
            threshold[index_row][index_column] = new_cell
    return threshold


def get_edges(image, blur_size, block_size, c):
    edged = copy.deepcopy(image)
    blur_image = apply_kernel(image, blur_kernel(blur_size))
    threshold = set_threshold(blur_image, block_size)
    for row in range(len(blur_image)):
        for column in range(len(blur_image[0])):
            if blur_image[row][column] < threshold[row][column] - c:
                edged[row][column] = 0
            else:
                edged[row][column] = 255
    return edged


def quantize(image, N):
    """
    :param image: 2d list
    :param N: natural number
    :return: new image seperated to N colors
    """
    new_image = set_new_matrix(len(image[0]), len(image))
    for index_row, row in enumerate(image):
        for index_cell, cell in enumerate(row):
            new_image[index_row][index_cell] = quantize_pixel(cell, N)
    return new_image


def quantize_colored_image(image, N):
    """
    :param image: 3d list
    :param N: natural number
    :return: new image seperated to N colors
    """
    channels = separate_channels(image)
    for index, c in enumerate(channels):
        channels[index] = quantize(c, N)
    return combine_channels(channels)


def add_mask(image1, image2, mask):
    """
    :param image1: list of pixels
    :param image2: list of pixels
    :param mask: list of values 1 or 0
    :return: new image combination of two images
    NOTE: the if statements checks if the images are multi color or 1 color
    """
    if isinstance(image1[0][0], int) and isinstance(image2[0][0], int):
        new_image = set_new_matrix(len(image1[0]), len(image1))
        for index_row, row in enumerate(image1):
            for index_column, column in enumerate(row):
                new_image[index_row][index_column] = round(
                    image1[index_row][index_column] * mask[index_row][index_column] +
                    image2[index_row][index_column] * (
                            1 - mask[index_row][index_column]))

    if isinstance(image2[0][0], list) and isinstance(image1[0][0], int):
        new_image = set_new_matrix(len(image1[0][0]), len(image1[0]), len(image1))
        image1, image2 = image2, image1
    if isinstance(image1[0][0], list) and isinstance(image2[0][0], int):
        new_image = set_new_matrix(len(image1[0][0]), len(image1[0]), len(image1))
        for index_row, row in enumerate(image1):
            for index_column, column in enumerate(row):
                for index_cell, cell in enumerate(column):
                    new_image[index_row][index_column][index_cell] = round(
                        image1[index_row][index_column][index_cell] * (1 - mask[index_row][index_column]) +
                        image2[index_row][index_column] * (
                            mask[index_row][index_column]))

    if isinstance(image2[0][0], list) and isinstance(image1[0][0], list):
        new_image = set_new_matrix(len(image1[0][0]), len(image1[0]), len(image1))
        for index_row, row in enumerate(image1):
            for index_column, column in enumerate(row):
                for index_cell, cell in enumerate(column):
                    new_image[index_row][index_column][index_cell] = round(
                        image1[index_row][index_column][index_cell] * mask[index_row][index_column] +
                        image2[index_row][index_column][index_cell] * (
                                1 - mask[index_row][index_column]))

    return new_image


def edge_to_mask(arr):
    """
    :param arr: list that is already showing only edges
    :return: mask kind of list
    """
    new_arr = copy.deepcopy(arr)
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j] == 255:
                new_arr[i][j] = 0
            else:
                new_arr[i][j] = 1

    return new_arr


def cartoonify(image, blur_size, th_block_size, th_c, quant_num_shades):
    """
    :param image: 3d list
    :param blur_size:
    :param th_block_size:
    :param th_c:
    :param quant_num_shades:
    :return:cartoon image
    """
    copy_image = copy.deepcopy(image)
    bnw = RGB2grayscale(image)
    edges = get_edges(bnw, blur_size, th_block_size, th_c)
    quanetized = quantize_colored_image(copy_image, quant_num_shades)
    mask = edge_to_mask(edges)
    masked = add_mask(quanetized, edges, mask)
    return masked


if __name__ == '__main__':
    if len(sys.argv) != 8:
        sys.exit()
    else:
        image_source = sys.argv[1]
        cartoon_dest = sys.argv[2]
        max_im_size = int(sys.argv[3])
        blur_size = int(sys.argv[4])
        th_block_size = int(sys.argv[5])
        th_c = int(sys.argv[6])
        quant_num_shades = int(sys.argv[7])
        loaded_image = helper.load_image(image_source)
        if max_im_size < len(loaded_image) or max_im_size < len(loaded_image[0]):
            if len(loaded_image) < len(loaded_image[0]):  # im_size will be the width
                ratio = len(loaded_image) / len(loaded_image[0])
                x, y = round(max_im_size * ratio), max_im_size  # value in resize

            else:  # im size will be the length
                ratio = len(loaded_image[0]) / len(loaded_image)
                x, y = max_im_size, round(max_im_size * ratio)  # value in resize

            separated_image = separate_channels(loaded_image)
            for index, i in enumerate(separated_image):
                separated_image[index] = resize(i, x, y)
            combined_image = combine_channels(separated_image)
            cartoon = cartoonify(combined_image, blur_size, th_block_size, th_c, quant_num_shades)
        else:
            cartoon = cartoonify(loaded_image, blur_size, th_block_size, th_c, quant_num_shades)
        helper.save_image(cartoon, cartoon_dest)
