"""
This module aligns sherds in RGB images to the respective depth images.

version: 1.0.5
Last Edited: 02-03-22
"""
import argparse
import math
import os
from math import radians

import cv2 as cv
import numpy as np


# TODO: Add documentation.
def create_result(sherd_img, depth_img):
    """
    This function creates the final image containing the cropped rgb and depth image.
    :param sherd_img: cropped Sherd image
    :param depth_img: cropped Depth image
    :return: a ndarray
    """
    padding = 100

    RGB_B = cv.copyMakeBorder(sherd_img, padding, padding, padding, padding, cv.BORDER_CONSTANT, value=(0, 0, 0))
    D_B = cv.copyMakeBorder(depth_img, padding, 0, padding, padding, cv.BORDER_CONSTANT, value=(0, 0, 0))

    rr, rc, _ = RGB_B.shape
    dr, dc, _ = D_B.shape

    height, width = rr, rc + dc

    result = np.zeros((height, width, 3), dtype=np.uint8)

    result[0:rr, 0:rc, :] = RGB_B
    result[0:dr, rc:rc + dc, :] = D_B

    return result


def first_agr(tup):
    """
    This function returns the first element in the given tuple.
    :param tup: the specified tuple.
    """
    return tup[0]


def match_helper(sherd_shape, depth_shape, inx):
    """
    This method is a helper method for the contour_match method.
    :param sherd_shape: contour from RGB
    :param depth_shape: contour from depth
    :param inx: index
    :return: The similarity, depth name, the depth shape, index in a tuple.
    """
    sim = cv.matchShapes(sherd_shape, depth_shape[inx][0], cv.CONTOURS_MATCH_I3, 0)
    return sim, depth_shape[inx][1], depth_shape[inx][0], inx


def contour_match(rbg_contours, depth_contours):
    """
    This function maps the RGB contours to depth images using the shape matching method to compare contours.
    :param rbg_contours: the contours from the rgb image
    :param depth_contours: the contours from the depth image
    :return: a list
    """
    c_results = []

    for j in rbg_contours:
        temp = [match_helper(j, depth_contours, d) for d in range(len(depth_contours))]

        # Maps the rgb to the depth mask with the lowest similarity
        if temp:
            dp_ob = min(temp, key=first_agr)
            c_results.append((j, dp_ob[1], dp_ob[0], dp_ob[2]))

            # Deletes depth values that have already been assigned
            del depth_contours[dp_ob[3]]

    return c_results


def crop(img, contours):
    """
    This function crops the image based on the given contours.
    :param img: the RGB image to crop from
    :param contours: the sherd contours
    :return: an Image
    """
    x, y, w, h = cv.boundingRect(contours)
    new_img = img[y:y + h, x:x + w]

    return new_img


def resize_img(img, w, h):
    """
    This method downsizes the image.
    :param img: An image.
    :param w: Target image width.
    :param h: Target image height.
    :return: resized image.
    """
    v_scale = h / img.shape[0]
    h_scale = w / img.shape[1]

    scale = max(v_scale, h_scale)
    dim = (int(scale * img.shape[1]), int(scale * img.shape[0]))
    n_img = cv.resize(img, dim, interpolation=cv.INTER_LINEAR)

    return n_img


def rotate_img(img, theta):
    """
    This function rotates a given image about the centre by a given angle.
    :param img: An image.
    :param theta: A number to represent the angle.
    :return: An image.
    """
    if (theta % 360.0) == 0.0:
        result = img
    else:
        h, w, _ = np.shape(img)
        img_centre = (w // 2, h // 2)

        rot = cv.getRotationMatrix2D(img_centre, theta, 1)
        rad = radians(theta)
        sin = math.sin(rad)
        cos = math.cos(rad)
        b_w = int((h * abs(sin)) + (w * abs(cos)))
        b_h = int((h * abs(cos)) + (w * abs(sin)))

        rot[0, 2] += ((b_w / 2) - img_centre[0])
        rot[1, 2] += ((b_h / 2) - img_centre[1])
        result = cv.warpAffine(img, rot, (b_w, b_h), flags=cv.INTER_LINEAR)

    return result


def overlay_img(r_img, d_img):
    # TODO: Finish this function.
    target_height = r_img.shape[0]
    target_width = r_img.shape[1]

    d_height = d_img.shape[0]
    d_width = d_img.shape[1]

    # Get margins
    margin_height = (d_height - target_height) // 2
    margin_width = (d_width - target_width) // 2

    roi = r_img[margin_width:margin_width + target_width, margin_height:margin_height + target_height]
    temp = cv.addWeighted(roi, 0.8, d_img, 0.2, 0)
    r_img[margin_width:margin_width + target_width, margin_height:margin_height + target_height] = temp

    return r_img


def add_cm_scale(img, s=1):
    """
    This function adds a 1cm scale to the bottom of the given image.
    :param s: An integer that represents the scale to reduce by.
    :param img: An image.
    :return: An image.
    """
    padding = 100
    temp_img = cv.copyMakeBorder(img, 0, padding, 0, 0, cv.BORDER_CONSTANT, value=(0, 0, 0))

    h, w, _ = temp_img.shape
    x = int(w // 2)

    # Draw 1 cm scale
    base_height = h - 50
    fin_height = h - 65
    base_width = x - 127
    fin_width = x + (126 * s)

    cv.line(temp_img, (base_width, base_height), (fin_width, base_height), (255, 255, 255), 2)  # Baseline
    cv.line(temp_img, (base_width, base_height), (base_width, fin_height), (255, 255, 255), 2)  # Leg One
    cv.line(temp_img, (fin_width, base_height), (fin_width, fin_height), (255, 255, 255), 2)  # Leg Two

    # Add text
    cv.putText(temp_img, "1cm", (base_width - 14, base_height + 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
               cv.LINE_AA, False)

    return temp_img


def get_centre(pts):
    """
    This function calculates the centre of the given contour.
    :param pts: A contour.
    :return: A tuple. i.e (x, y)
    """
    mm = cv.moments(pts)
    x = int(mm['m10'] / mm['m00'])
    y = int(mm['m01'] / mm['m00'])

    return x, y


def get_max_distance(pts, centre):
    """
    This function calculates the furthest point in the given contour from the centre of the contour.
    :param pts: A contour.
    :param centre: A tuple to represent the centre point. i.e (x, y)
    :return: An int Distance, A tuple of the max point.
    """
    max_d = 0.0
    min_d = 10000
    max_pt = None
    min_pt = None

    for d in range(len(pts)):
        for x, y in pts[d]:
            # Find distance from centre
            temp_d = math.sqrt(math.pow((x - centre[0]), 2) + math.pow((y - centre[1]), 2))

            # Check if distance is greater than previous max distance
            if temp_d > max_d:
                max_d = temp_d
                max_pt = (x, y)

            # Check if distance is less than previous min distance
            if temp_d < min_d:
                min_d = temp_d
                min_pt = (x, y)

    return max_d, max_pt, min_d, min_pt


def get_angle(rgb_pts, depth_pts, cen_pts):
    """
    This function calculates the angle between the three points: furthest rgb point, the furthest depth point,
    and the centre point of the contour.
    :param rgb_pts: A tuple, furthest rgb point of a contour.
    :param depth_pts: A tuple, the furthest depth point of a contour.
    :param cen_pts: A tuple, centre point of the contour.
    :return: An float angle.
    """
    angle = math.degrees(
        math.atan2(depth_pts[1] - cen_pts[1], depth_pts[0] - cen_pts[0]) - math.atan2(rgb_pts[1] - cen_pts[1],
                                                                                      rgb_pts[0] - cen_pts[0]))
    angle = angle + 360 if angle < 0 else angle

    return angle


def find_square(cnt):
    """
    This method determines if the contour is a square.
    :param cnt: the contour
    :return: a contour
    """
    area = cv.contourArea(cnt)

    if area > 10000:
        per = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.01 * per, True)

        if len(approx) == 4:
            return cnt


def block_quads(A, img, conts):
    """
    This function returns a mask with the quadrilateral shapes blocked out.
    :param A: Original RGB image
    :param img: Image used to combine with blocked out mask
    :param conts: Quadrilateral contours
    :return: A binary mask.
    """
    mask = np.zeros(A.shape, np.uint8)

    # Holds the contours of non-sherd shapes
    square_cont = tuple(map(find_square, conts))
    square_cont = tuple(x for x in square_cont if x is not None)

    if len(square_cont) > 0:
        cv.drawContours(mask, square_cont, -1, (255, 255, 255), cv.FILLED)
        square_cont = sorted(square_cont, key=cv.contourArea, reverse=False)

        # Creates new binary image without non-sherd shapes.
        BW = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        combine_mask = cv.subtract(BW, mask)
        combine_mask = cv.cvtColor(combine_mask, cv.COLOR_BGR2GRAY)
        return combine_mask

    return None


def get_img_contours(img):
    """
    This function finds the contour in an image and sorts them.
    :param img: An image
    :return: Contours
    """
    temp_contours, _ = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    temp_contours = sorted(temp_contours, key=cv.contourArea, reverse=True)

    return temp_contours


def align(filename, rgb_dir, depth_dir, output_path):
    """
    This function pairs the splits and pairs sherds from RGB image with the respective depth images.
    :param filename: Name of RGB image
    :param rgb_dir: Path to directory with RGB images
    :param depth_dir: Path to directory with depth images
    :param output_path: Path to directory to store results
    :return: None
    """

    # Retrieves the sherd's ID from the filename.
    ending = 'ext' if 'ext' in filename else 'int' if 'int' in filename else None

    # Splits the sherd id from sherd name.
    scan_name = filename.split('_')[0] if ending else filename.split('.')[0]

    print(f'Processing: {scan_name}')

    # Read image
    A = cv.rotate(cv.imread(os.path.join(rgb_dir, filename)), cv.ROTATE_180)
    _, BW = cv.threshold(cv.cvtColor(cv.GaussianBlur(A, (31, 31), 0), cv.COLOR_BGR2GRAY), 20, 255, cv.THRESH_BINARY)

    # Find initial contours and block non-sherd shapes
    temp_conts = get_img_contours(BW)
    mask = block_quads(A, BW, temp_conts)

    sherd_cont = get_img_contours(mask)
    sherd_cont = [cont for cont in sherd_cont if cv.contourArea(cont) > 100000]

    # Retrieve depth images
    print("--- Retrieving Depth images.")
    dp_files = os.listdir(depth_dir)
    dp_files = list(filter(lambda f: scan_name in f, dp_files))
    dp_contours = []

    # Reads the depth image, convert it to a binary image, and finds the contours.
    for d in dp_files:
        if ending is None or ending == "ext":
            d_img = cv.imread(os.path.join(depth_dir, d))
        else:
            d_img = cv.flip(cv.imread(os.path.join(depth_dir, d)), 1)

        _, dp_BW = cv.threshold(cv.cvtColor(cv.GaussianBlur(d_img, (31, 31), 0), cv.COLOR_BGR2GRAY), 20, 255,
                                cv.THRESH_BINARY)
        t_cont = get_img_contours(dp_BW)
        # Store the contours with the depth image name
        dp_contours.append((t_cont[0], d))

    # Determines if the number of Sherds in the RGB image matches the number of depth images
    if len(sherd_cont) == len(dp_contours):
        # Pairs the RGB contours to the matching depth contours
        print("--- Matching RGB contours to depth contours.")
        results = contour_match(sherd_cont, dp_contours)
    else:
        print(
            f"ERROR: Number of sherds for {scan_name} do not match the number of depth images. Sherd: {len(sherd_cont)} Depth: {len(dp_contours)}")
        return

    print(f"--- {len(results)} results were matched.")
    print("--- Creating result images.")

    # Creating result image
    for r in range(len(results)):
        # Creates output directory if path does not exist
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # Constructs result image
        r_img = crop(A, results[r][0])

        # Flips depth/mask images for internal sherd rgb images
        if ending is None or ending == "ext":
            d_img = cv.imread(os.path.join(depth_dir, results[r][1]))
        else:
            d_img = cv.flip(cv.imread(os.path.join(depth_dir, results[r][1])), 1)

        t_img = resize_img(r_img, d_img.shape[1], d_img.shape[0])
        _, t_BW = cv.threshold(cv.cvtColor(cv.GaussianBlur(t_img, (31, 31), 0), cv.COLOR_BGR2GRAY), 20, 255,
                               cv.THRESH_BINARY)
        cont, _ = cv.findContours(t_BW, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        cont = sorted(cont, key=cv.contourArea, reverse=True)
        f_cnt = cont[0]

        # Retrieve angles from the depth and rgb images; calculates the angle of rotation
        r_cen = get_centre(f_cnt)
        d_cen = get_centre(results[r][3])

        cen = ((r_cen[0] + d_cen[0]) / 2, (r_cen[1] + d_cen[1]) / 2)

        r_max_d, r_max_pt, r_min_d, r_min_pt = get_max_distance(f_cnt, r_cen)
        d_max_d, d_max_pt, d_min_d, d_min_pt = get_max_distance(results[r][3], d_cen)

        # Max_theta: Angle calculated from the max depth, RGB, and centre point
        # Min_theta: Angle calculated from the min depth, RGB, and centre point
        max_theta = get_angle(r_max_pt, d_max_pt, cen)
        min_theta = get_angle(r_min_pt, d_min_pt, cen)

        # VISUAL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Uncomment code below to see visual of the sherd contours, centre, maxima, and minima points.
        # Note: RGB image is scaled to roughly match depth image.

        # Draw Contours
        # cv.drawContours(t_img, f_cnt, -1, (255, 255, 255), 2)
        # cv.drawContours(d_img, results[r][3], -1, (0, 204, 255), 2)

        # RED -> MAX POINT
        # BLUE -> CENTRE
        # GREEN -> MIN POINT
        # cv.circle(t_img, r_max_pt, 8, (0, 0, 255), -1)
        # cv.circle(t_img, r_cen, 8, (255, 0, 0), -1)
        # cv.circle(t_img, r_min_pt, 8, (0, 255, 0), -1)
        #
        # cv.circle(d_img, d_max_pt, 8, (0, 0, 255), -1)
        # cv.circle(d_img, d_cen, 8, (255, 0, 0), -1)
        # cv.circle(d_img, d_min_pt, 8, (0, 255, 0), -1)
        #
        # cv.imshow("RGB", t_img)
        # cv.imshow("Depth", d_img)
        # cv.waitKey()
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # Rotate the RGB image with respect to the Depth image
        # rot_img = rotate_img(r_img, -min_theta)
        # result_img = create_result(rot_img, d_img)

        # Rotate the Depth image with respect to the RGB image
        rot_img = rotate_img(d_img, max_theta)

        # Add Scale if option was selected
        if args["scale"]:
            bh, bw, _ = r_img.shape
            sh, sw, _ = rot_img.shape

            scale = max(int(sh/bh), int(sw/bw))

            rot_img = add_cm_scale(rot_img, scale)
            r_img = add_cm_scale(r_img)

        result_img = create_result(r_img, rot_img)

        if ending is not None:
            cv.imwrite(f"{output_path}/{scan_name}_{ending}_{r + 1}.png", result_img)
        else:
            cv.imwrite(f"{output_path}/{scan_name}_{r + 1}.png", result_img)


# Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directories", required=True, nargs='+',
                help="The rgb and depth image directories. Usage: python align.py -d <rgb_dir_path> <depth_dir_path>")
ap.add_argument("-o", "--output_dir", required=False, default="./output", help="The path to store results.")
ap.add_argument("-l", "--overlay", required=False, help="Enables overlay visualization.", action="store_true")
ap.add_argument("-s", "--scale", required=False, help="Enables scale to be visible on result images.",
                action="store_true")
args = vars(ap.parse_args())

# Checks arguments
if 1 < len(args["directories"]) < 3:
    # Retrieve RGB files
    rgb_files = os.listdir(args["directories"][0])
    rgb_files.sort()

    # Run RGB images through "align" function
    for i in rgb_files:
        align(i, args["directories"][0], args["directories"][1], args["output_dir"])
    print("Done.")
else:
    print(f"ERROR: User supplied {len(args['directories'])} argument, but this programme requires 2 argument.")
