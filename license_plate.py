# import necessary packages
from imutils import paths
from ANPR import ANPR
import numpy as np
import argparse
import cv2
from PIL import Image

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True, default="/license_plates",
#                 help="path to input directory of images")
# ap.add_argument("-c", "--clear_border", type=int, default=-1,
#                 help="whether or to clear border pixels before OCR'ing")
# ap.add_argument("-p", "--psm", type=int, default=7,
#                 help="default PSM mode for OCR'ing license plates")
# ap.add_argument("-d", "--debug", type=int, default=-1,
#                 help="whether or not to show additional visualizations")
# ap.add_argument("-r", "--resize", type=int, default=1,
#                 help="whether or not to resize the image")
# args = vars(ap.parse_args())
#
# # initialize the ANPR class
# anpr = ANPR.ANPR(debug=args["debug"] > 0)

if __name__ == "__main__":

    # initialize the ANPR class
    anpr = ANPR.ANPR(debug=False, resize=True)
    # set the model of net
    anpr.load_models("ANPR")

    # grab all images paths in the input directory
    imagePaths = sorted(list(paths.list_images("license_plates/")))
    # file to record the detection result
    file_success = open("report/success.txt", 'w')
    file_failed = open("report/failed.txt", 'w')

    index = 1
    success_num = 0
    # loop over all image paths in the input directory
    for imagePath in imagePaths:
        print('----------------------------------------')
        print('Image {} : {} \n'.format(index, imagePath))

        # set the image to be detected
        anpr.set_image(imagePath)
        # recognize the license plate
        success, lp_list = anpr.detection()

        # record result to file.
        if success is True:
            success_num += 1
            # result = ''.join(lp_list)
            # print(result)
            print('[DETECTION] detect result is {}'.format(lp_list[0]))
            file_success.write('filename: {}, license: {}\n'.format(imagePath, lp_list[0]))
        else:
            print('[DETECTION] detect failed')
            file_failed.write('filename: {}\n'.format(imagePath))

        index += 1
        print('----------------------------------------\n')

    file_success.close()
    file_failed.close()

    print('----------------------------------------\n')
    print('Total image : {}'.format(index - 1))
    print('Success result : {}'.format(success_num))
    print('accuracy : {}'.format(success_num / float(index - 1)))
