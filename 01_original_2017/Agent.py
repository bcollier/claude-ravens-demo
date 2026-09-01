# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
from PIL import Image, ImageFilter, ImageChops, ImageStat
import numpy as np
import copy
import random
import time


class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        pass



    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints 
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self, problem):
        # try:
            # it sounds like my code goes here.
        print(problem.name)

        if (problem.problemType == "3x3"): # & ("challenge" not in problem.problemSetName.lower()):
            return solve3x3(problem)


        #if not problem.hasVerbal:
            # print("Not yet configured for visual-only problems")
        #    return guess_or_pass()

        #return verbal_solve(problem)
        return -1

        #except:
        #    print("Error in solve, returning -1")
        #    return -1


def img_to_black_white(problem, image_letter):
    # https://stackoverflow.com/questions/9506841/using-python-pil-to-turn-a-rgb-image-into-a-pure-black-and-white-image
    image_source = problem.figures[image_letter].visualFilename

    # im_raw = Image.open(image_source)

    # im_raw.show()

    # im_blur = im_raw.filter(ImageFilter.GaussianBlur)

    # im_blur.show()

    # im = im_blur.convert('L').point(lambda x: 0 if x < 255 else 255, '1')

    im = Image.open(image_source).convert('L').point(lambda x: 0 if x < 255 else 255, '1')

    # im.show()

    # im_blur = im.filter(ImageFilter.GaussianBlur)

    # returns as white = 255, black = 0
    return im


def get_dpr(im):
    # I tried this the "easy" way with numpy, so many problems with it I found it better to just use pillow and
    # go pixel by pixel through the image, it is not the most efficient way to do it obviously
    dark_pixel_count = 0

    for x in range(im.width):
        for y in range(im.height):
            pixel_value = im.getpixel((x, y))  # gets numerical value of pixel, 255 or 0

            if pixel_value == 0:
                dark_pixel_count += 1

    # percentage of pixels in the image that are dark
    dark_pixel_ratio = dark_pixel_count / (im.width * im.height)

    return dark_pixel_ratio


def get_dpr_from_img(problem, image_letter):
    im = img_to_black_white(problem, image_letter)
    return get_dpr(im)


def get_ipr(im1, im2):
    # I tried this the "easy" way with numpy, so many problems with it I found it better to just use pillow and
    # go pixel by pixel through the image, it is not the most efficient way to do it obviously
    intersectional_pixel_count = 0
    dark_pixel_count_1 = 0
    dark_pixel_count_2 = 0

    for x in range(im1.width):
        for y in range(im1.height):
            pixel_value1 = im1.getpixel((x, y))  # gets numerical value of pixel, 255 or 0
            pixel_value2 = im2.getpixel((x, y))

            if (pixel_value1 == 0) & (pixel_value2 == 0):
                intersectional_pixel_count += 1

            if (pixel_value1 == 0):
                dark_pixel_count_1 += 1

            if (pixel_value2 == 0):
                dark_pixel_count_2 += 1

    # percentage of dark pixels in the image that are dark for both images
    intersectional_pixel_ratio = (intersectional_pixel_count / (dark_pixel_count_1 + dark_pixel_count_2)) / 0.50

    return intersectional_pixel_ratio


def get_ipr_from_imgs(problem, image_letter1, image_letter2):
    im1 = img_to_black_white(problem, image_letter1)
    im2 = img_to_black_white(problem, image_letter2)
    return get_ipr(im1, im2)


def score_dpr(problem, dpr_scores, dpr_row, dpr_col, dpr_diag):
    g_dpr = get_dpr_from_img(problem, "G")
    h_dpr = get_dpr_from_img(problem, "H")
    c_dpr = get_dpr_from_img(problem, "C")
    f_dpr = get_dpr_from_img(problem, "F")

    a_dpr = get_dpr_from_img(problem, "A")
    e_dpr = get_dpr_from_img(problem, "E")

    gh_pred_dpr = (h_dpr - g_dpr) + h_dpr
    cf_pred_dpr = (f_dpr - c_dpr) + f_dpr
    ae_pred_dpr =  (e_dpr - a_dpr) + e_dpr

    #  loop values 1 through 8 for potential solutions
    for i in range(1, 9):
        i_dpr = get_dpr_from_img(problem, str(i))

        # scores is max value of 10 for best fit
        # for each 1% error in prediction, the score goes down by 1 point, so predict 14% dpr and actual is 13% the
        # score would be a 9 / 10. the minimum score is 1 for those with high error
        dpr_scores[i, dpr_row] = max(10 - abs(gh_pred_dpr - i_dpr) * 100, 1)
        dpr_scores[i, dpr_col] = max(10 - abs(cf_pred_dpr - i_dpr) * 100, 1)
        dpr_scores[i, dpr_diag] = max(10 - abs(ae_pred_dpr - i_dpr) * 100, 1)

    return dpr_scores


def score_ipr(problem, ipr_scores, ipr_row, ipr_col, ipr_diag):
    gh_ipr = get_ipr_from_imgs(problem, "G", "H")
    cf_ipr = get_ipr_from_imgs(problem, "C", "F")
    ae_ipr = get_ipr_from_imgs(problem, "A", "E")

    #  loop values 1 through 8 for potential solutions
    for i in range(1, 9):
        hi_ipr = get_ipr_from_imgs(problem, "H", str(i))
        fi_ipr = get_ipr_from_imgs(problem, "F", str(i))
        ei_ipr = get_ipr_from_imgs(problem, "E", str(i))

        hi_gh_ipr_diff = abs(gh_ipr - hi_ipr)
        fi_cf_ipr_diff = abs(cf_ipr - fi_ipr)
        ei_ae_ipr_diff = abs(ae_ipr - ei_ipr)

        # scores is max value of 10 for best fit
        # for each 10% error in IPR difference, the score goes down by 1 point, lowest values is 1
        # example: GH IPR is .90, HI IPR is .80, score would be 9 / 10
        # example: CF IPR is .80, FI IPR is .40, score would be 6 / 10

        ipr_scores[i, ipr_row] = max(10 - hi_gh_ipr_diff * 10, 1)
        ipr_scores[i, ipr_col] = max(10 - fi_cf_ipr_diff * 10, 1)
        ipr_scores[i, ipr_diag] = max(10 - ei_ae_ipr_diff * 10, 1)

    return ipr_scores


def score_obj_cnt(problem, obj_cnt_scores, obj_cnt_row, obj_cnt_col):
    g_obj_cnt = problem.figures["G"].objects.__len__()
    h_obj_cnt = problem.figures["H"].objects.__len__()
    c_obj_cnt = problem.figures["C"].objects.__len__()
    f_obj_cnt = problem.figures["F"].objects.__len__()

    #  loop values 1 through 8 for potential solutions
    for i in range(1, 9):
        i_obj_cnt = problem.figures[str(i)].objects.__len__()

        # scores is max value of 10 for best fit, minimum value of 1
        # for each object different than predicted, we subtract 2 points from the score
        # example: G has 1 obj, H as 2 obj, we predict 3 but I has 4 leads to score of 8
        # example: C has 3 obj, F has 2, we predict 1 object but I has 3, leads to score of 6
        obj_cnt_scores[i, obj_cnt_row] = max(10 - abs((h_obj_cnt - g_obj_cnt) + h_obj_cnt - i_obj_cnt) * 2, 1)
        obj_cnt_scores[i, obj_cnt_col] = max(10 - abs((f_obj_cnt - c_obj_cnt) + f_obj_cnt - i_obj_cnt) * 2, 1)

    return obj_cnt_scores


def get_dark_pixel_centroid(im):
    # get the x , y center for all the dark pixels
    x_total = 0
    y_total = 0
    dark_pixel_count = 0

    for x in range(im.width):
        for y in range(im.height):
            pixel_value = im.getpixel((x, y))  # gets numerical value of pixel, 255 or 0

            if pixel_value == 0:
                x_total += x
                y_total += y
                dark_pixel_count += 1

    # remove issue of divide by zero
    if dark_pixel_count == 0:
        dark_pixel_count += 1

    # percentage of pixels in the image that are dark
    dark_pixel_centroid_x = x_total / dark_pixel_count
    darK_pixel_centroid_y = y_total / dark_pixel_count

    # this should return the x, y coordinates of the center or average of where the dark pixels are
    dark_pixel_centroid_xy = (dark_pixel_centroid_x, darK_pixel_centroid_y)

    return dark_pixel_centroid_xy


def get_dark_pixel_centroid_from_img(problem, image_letter):
    im = img_to_black_white(problem, image_letter)
    return get_dark_pixel_centroid(im)


def score_dark_pixel_centroids(problem, dpix_center_scores, dpix_cntr_row, dpix_cntr_col):
    g_cntr = get_dark_pixel_centroid_from_img(problem, "G")
    h_cntr = get_dark_pixel_centroid_from_img(problem, "H")
    c_cntr = get_dark_pixel_centroid_from_img(problem, "C")
    f_cntr = get_dark_pixel_centroid_from_img(problem, "F")

    gh_x_pred = (h_cntr[0] - g_cntr[0]) + h_cntr[0]
    gh_y_pred = (h_cntr[1] - g_cntr[1]) + h_cntr[1]
    cf_x_pred = (f_cntr[0] - c_cntr[0]) + f_cntr[0]
    cf_y_pred = (f_cntr[1] - c_cntr[1]) + f_cntr[1]

    #  loop values 1 through 8 for potential solutions
    for i in range(1, 9):
        i_cntr = get_dark_pixel_centroid_from_img(problem, str(i))

        # scores is max value of 10 for best fit, minimum value of 1
        row_pred_distance = abs(gh_x_pred - i_cntr[0]) + abs(gh_y_pred - i_cntr[1])
        col_pred_distance = abs(cf_x_pred - i_cntr[0]) + abs(cf_y_pred - i_cntr[1])

        if row_pred_distance < 5:
            adj_row_score = 10
        elif row_pred_distance < 20:
            adj_row_score = 7
        elif row_pred_distance < 60:
            adj_row_score = 3
        else:
            adj_row_score = 1

        if col_pred_distance < 5:
            adj_col_score = 10
        elif col_pred_distance < 20:
            adj_col_score = 7
        elif col_pred_distance < 60:
            adj_col_score = 3
        else:
            adj_col_score = 1

        dpix_center_scores[i, dpix_cntr_row] = adj_row_score
        dpix_center_scores[i, dpix_cntr_col] = adj_col_score

    # if problem.name == "Basic Problem C-07":
    #     print("Improve this problem")

    return dpix_center_scores


def compare_images_similarity(problem, image_letter1, image_letter2):
    im1 = img_to_black_white(problem, image_letter1)
    im2 = img_to_black_white(problem, image_letter2)

    img_diff = ImageChops.difference(im1, im2)

    img_stats = ImageStat.Stat(img_diff)

    dark_pix_cnt = img_stats.h[000]
    white_pix_cnt = img_stats.h[255]

    err_perc = white_pix_cnt / dark_pix_cnt

    return err_perc


def check_identity(problem, identity_scores, identity_check_col, eliminate_col):

    # error less than 3% is identical
    identical_threshold = .03

    # gh_ident_score = compare_images_similarity(problem, "G", "H")
    # cf_ident_score = compare_images_similarity(problem, "C", "F")
    # ae_ident_score = compare_images_similarity(problem, "A", "E")

    gh_ident = compare_images_similarity(problem, "G", "H") < identical_threshold
    cf_ident = compare_images_similarity(problem, "C", "F") < identical_threshold
    ae_ident = compare_images_similarity(problem, "A", "E") < identical_threshold

    if gh_ident or cf_ident or ae_ident:
        if gh_ident:
            compare_letter = "H"
        elif cf_ident:
            compare_letter = "F"
        elif ae_ident:
            compare_letter = "E"

        #  loop values 1 through 8 for potential solutions
        for i in range(1, 9):
            if compare_images_similarity(problem, compare_letter, str(i)) < identical_threshold:
                # iscore = compare_images_similarity(problem, compare_letter, str(i))
                identity_scores[i, identity_check_col] = 10

    # if we don't have identity in rows and columns or diagnols, eliminate identity choices from solution set
    if not gh_ident:
        for i in range(1, 9):
            if (compare_images_similarity(problem, "G", str(i)) < identical_threshold) or (compare_images_similarity(problem, "H", str(i)) < identical_threshold):
                identity_scores[i, eliminate_col] = 10
                print("row eliminating choice " + str(i) + " from problem ", problem.name)

    if not cf_ident:
        for i in range(1, 9):
            if (compare_images_similarity(problem, "C", str(i)) < identical_threshold) or (compare_images_similarity(problem, "F", str(i)) < identical_threshold):
                identity_scores[i, eliminate_col] = 10
                print("col eliminating choice " + str(i) + " from problem ", problem.name)

    if not ae_ident:
        for i in range(1, 9):
            if (compare_images_similarity(problem, "A", str(i)) < identical_threshold) or (compare_images_similarity(problem, "E", str(i)) < identical_threshold):
                identity_scores[i, eliminate_col] = 10
                print("diag eliminating choice " + str(i) + " from problem ", problem.name)

    return identity_scores


def compare_images_similarity_imgsrc(im1, im2):

    img_diff = ImageChops.difference(im1, im2)

    img_stats = ImageStat.Stat(img_diff)

    dark_pix_cnt = img_stats.h[000]
    white_pix_cnt = img_stats.h[255]

    err_perc = white_pix_cnt / dark_pix_cnt

    return err_perc


def check_addition(problem, input1, input2, output):
    # error less than 3% is identical
    identical_threshold = .03

    im1 = ImageChops.invert(img_to_black_white(problem, input1))
    im2 = ImageChops.invert(img_to_black_white(problem, input2))
    im3 = ImageChops.invert(img_to_black_white(problem, output))

    dp = ImageChops.add(im1, im2)

    diff = compare_images_similarity_imgsrc(dp, im3)

    image_addition_true = diff < identical_threshold

    return image_addition_true


def get_addition_score(problem, input1, input2, output):


    im1 = ImageChops.invert(img_to_black_white(problem, input1))
    im2 = ImageChops.invert(img_to_black_white(problem, input2))
    im3 = ImageChops.invert(img_to_black_white(problem, output))

    dp = ImageChops.add(im1, im2)

    diff = compare_images_similarity_imgsrc(dp, im3)

    return diff


def score_image_addition(problem, img_addition_scores, addition_match_col, eliminate_col):
    min_index = -1
    min_value = 100

    if check_addition(problem, "A", "B", "C") and check_addition(problem, "D", "E", "F"):
        for i in range(1, 9):
            if check_addition(problem, "G", "H", str(i)):
                potential_best_fit = get_addition_score(problem, "G", "H", str(i))

                if potential_best_fit < min_value:
                    img_addition_scores[i, addition_match_col] = 10
                    img_addition_scores[i, eliminate_col] = 0

                    if min_index >= 0:
                        img_addition_scores[min_index, addition_match_col] = 0

                    min_index = i
                    min_value = potential_best_fit

    return img_addition_scores



def solve3x3(problem):
    print("Predicting solution for: " + problem.name)
    start_time = time.time()

    # create structure similar to
    # PotentialSolution     DPR_ROW     DPR_COL     IPR_ROW     IPR_COL     OBJ_CNT_ROW  OBJ_CNT_COL   WeightedScore
    # 1                     8.1         9.1         4.5             7.3     1.1             2.5             6.2

    # just a way of naming what scores are in what columns
    DPR_ROW = 1
    DPR_COL = 2
    DPR_DIAG = 3

    IPR_ROW = 4
    IPR_COL = 5
    IPR_DIAG = 6

    OBJ_CNT_ROW = 7
    OBJ_CNT_COL = 8

    DPIX_CNTR_ROW = 9
    DPIX_CNTR_COL = 10

    IDENTITY_CHECK = 11

    ELIMINATE = 12

    ADDITION_MATCH = 13

    SUBTRACTION_MATCH = 14

    INTERSECTION_MATCH = 15

    WEIGHTED_SCORE = 16


    # weighting scores by columns
    DPR_ROW_WT = 0.50
    DPR_COL_WT = 0.20
    IPR_ROW_WT = 0.10
    IPR_COL_WT = 0.10
    OBJ_CNT_ROW_WT = 0
    OBJ_CNT_COL_WT = 0
    DPIX_CNTR_ROW_WT = 0.0
    DPIX_CNTR_COL_WT = 0.0
    DPR_DIAG_WT = 0.50
    IPR_DIAG_WT = 0.15
    IDENTITY_CHECK_WT = 5
    ELIMINATE_WT = -10

    ADDITION_MATCH_WT = 5

    SUBTRACTION_MATCH_WT = 5

    INTERSECTION_MATCH_WT = 5


    # used to hold scores at each stage
    solution_scores = np.array([[0.00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # STEP 1: Get DPR Scores
    solution_scores = score_dpr(problem, solution_scores, DPR_ROW, DPR_COL, DPR_DIAG)

    # STEP 2: Get IPR Scores
    solution_scores = score_ipr(problem, solution_scores, IPR_ROW, IPR_COL, IPR_DIAG)

    # No longer used in Project 3
    # STEP 3: Get ObjCountScores
    # if problem.hasVerbal:
    #     solution_scores = score_obj_cnt(problem, solution_scores, OBJ_CNT_ROW, OBJ_CNT_COL)
    #     OBJ_CNT_ROW_WT = 0.10
    #     OBJ_CNT_COL_WT = 0.10

    # STEP 3: Check for identity on rows, col, or diagnols
    solution_scores = check_identity(problem, solution_scores, IDENTITY_CHECK, ELIMINATE)

    # STEP 4: Dark Pixel Centroids
    solution_scores = score_dark_pixel_centroids(problem, solution_scores, DPIX_CNTR_ROW, DPIX_CNTR_COL)

    # STEP 5: Image Addition
    # & ("challenge" not in problem.problemSetName.lower()):
    if "problem d" not in problem.name.lower():
        solution_scores = score_image_addition(problem, solution_scores, ADDITION_MATCH, ELIMINATE)

    # STEP 6: Image Subtraction

    # STEP 7: Image Intersection

    # STEP 8: Weight Scores
    for i in range(1, 9):
        solution_scores[i, WEIGHTED_SCORE] = (DPR_ROW_WT * solution_scores[i, DPR_ROW] +
                                              DPR_COL_WT * solution_scores[i, DPR_COL] +
                                              IPR_COL_WT * solution_scores[i, IPR_COL] +
                                              IPR_COL_WT * solution_scores[i, IPR_COL] +
                                              OBJ_CNT_ROW_WT * solution_scores[i, OBJ_CNT_ROW] +
                                              OBJ_CNT_COL_WT * solution_scores[i, OBJ_CNT_COL] +
                                              DPIX_CNTR_ROW_WT * solution_scores[i, DPIX_CNTR_ROW] +
                                              DPIX_CNTR_COL_WT * solution_scores[i, DPIX_CNTR_COL] +
                                              DPR_DIAG_WT * solution_scores[i, DPR_DIAG] +
                                              IPR_DIAG_WT + solution_scores[i, IPR_DIAG] +
                                              IDENTITY_CHECK_WT * solution_scores[i, IDENTITY_CHECK] +
                                              ELIMINATE_WT * solution_scores[i, ELIMINATE] +
                                              ADDITION_MATCH_WT * solution_scores[i, ADDITION_MATCH] +
                                              SUBTRACTION_MATCH_WT * solution_scores[i, SUBTRACTION_MATCH] +
                                              INTERSECTION_MATCH_WT * solution_scores[i, INTERSECTION_MATCH])

    solution_scores = sort_array_by_col(solution_scores, WEIGHTED_SCORE)

    best_solution_number = int(solution_scores[0,0])

    # STEP 6: Return Prediction
    print("Best solution is: " + str(best_solution_number) + " with a score of: " +
          str(round(solution_scores[0, WEIGHTED_SCORE], 2)))

    run_time = round(time.time() - start_time, 2)

    output_file_name = problem.name + "_solution_scores.csv"
    # np.savetxt(output_file_name, solution_scores, delimiter=",")

    print("Solution took " + str(run_time) + " seconds\n")

    # f = open('combined_solution_scores.csv', 'ab')
    # np.savetxt(f, solution_scores, delimiter=",")
    # f.close()

    # if problem.name == "Basic Problem D-04":
    #     print("Improve this problem")



    return best_solution_number


def sort_array_by_col(a, coln):
    # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
    return a[a[:, coln].argsort()[::-1]]


def guess_or_pass3x3():
    return random.randint(1, 8)