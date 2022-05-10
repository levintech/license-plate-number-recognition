# -----------------------------------------------------------
# German License Plate Detection
#
# Client        : Caglar A.
# Author        : Vladislav K.
# File Name     : ANPR.py
# Class Name    : ANPR
# summary       : This is the class to detect the license plate.
# -----------------------------------------------------------

# import necessary packages
import numpy as np
import cv2
from ANPR.local_utils import detect_lp
from PIL import Image, ImageOps

from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import pytesseract

# Begin Constants
DEFAULT_SIZE = 224
D_MAX = 800
D_MIN = 350
# End Constants


class ANPR:
    """
    the class to detect the german license plate from the image.
    """

    # Begin Built-in Methods...
    def __init__(self, debug=False, path="", resize="True"):
        """
        Initialize class scope variables...

        :param debug: (boolean) flag to see if the processed images should be showed or not in each step.
        :param path: (string) the path of the image located.
        :param resize: (boolean) flag to see if the image should be resize or not.
        """

        self.debug = debug
        self.original_image = None
        self.path = path
        self.resize = resize
        self.region_net = None
        self.extract_net = None
        self.extract_labels = None

    # End Built-in Methods...

    def debug_show(self, title, image, wait_key=False):
        """
        The function is to check the see if we are debug mode, if so show the image.

        :param title: (string) the desired opencv window title.
        :param image: () the image to display inside the opencv window
        :param wait_key: (boolean) flag to see if the display should wait the keypress

        :return: none
        """

        if self.debug:
            cv2.imshow(title, image)

            # check to see if we should wait keypress
            if wait_key:
                cv2.waitKey(0)

    def load_models(self, path):
        """
        the function is to load the net's model from the path

        :param path: (string) the path of the model file

        :return: none
        """

        try:
            # Load the region net's model
            with open('%s/region.json' % path, 'r') as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json, custom_objects={})
            model.load_weights('%s/region.h5' % path)
            print("[INFO] region model loaded successfully...")
            self.region_net = model

            # Load model architecture, weight and labels
            json_file = open('%s/MobileNets_character_recognition.json' % path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights("%s/License_character_recognition_weight.h5" % path)
            print("[INFO] extract model loaded successfully...")
            self.extract_net = model

            labels = LabelEncoder()
            labels.classes_ = np.load('%s/license_character_classes.npy' % path)
            print("[INFO] extract labels loaded successfully...")
            self.extract_labels = labels
        except Exception as e:
            print('[ERROR] loading models contains issue. \n {}'.format(e))

    def preprocess_image(self):
        """
        the function is load the image from path and resize them.

        :return: none
        """

        # load image from path
        self.original_image = cv2.imread(self.path)
        # convert image to RGB color format
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.original_image = self.original_image / 255

        # resize the image along with the self.resize value whether or not
        if self.resize:
            cv2.resize(self.original_image, (DEFAULT_SIZE, DEFAULT_SIZE))

    def set_image(self, path):
        """
        the function to set the image's path

        :param path: (string) the path of the image
        :return: none
        """

        self.path = path

    @staticmethod
    def is_valid(region_list, region):
        """
        the function is to compare two region's overlap

        :param region_list: (tuple) list of the region
        :param region: (tuple) current region
        :return: (boolean) the result is Ture if any region contain current one. otherwise False
        """

        if len(region_list) == 0:
            return True

        (cur_x, cur_y, cur_w, cur_h) = cv2.boundingRect(region)

        for reg in region_list:
            (x, y, w, h) = cv2.boundingRect(reg)
            if (cur_x >= x and cur_y >= y) and ((cur_x + cur_w) <= (x + w) and (cur_y + cur_h) <= (y + h)):
                return False

        return True

    @staticmethod
    def split_position(contours):
        """
        the function to grab the contour of each digit from left to right

        :param contours: (list) the list of the contours of characters
        :return split_pos: (list) the list of the split positions
        """

        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        ch_width = 0
        for bound in boundingBoxes:
            ch_width += bound[2]
        ch_width /= len(boundingBoxes)

        split_pos = []
        split_size = []
        for nI in range(0, len(boundingBoxes) - 1):
            if (boundingBoxes[nI + 1][0] - (boundingBoxes[nI][0] + boundingBoxes[nI][2])) > ch_width / 3:
                split_size.append(boundingBoxes[nI + 1][0] - (boundingBoxes[nI][0] + boundingBoxes[nI][2]))
                split_pos.append(nI + 1)

        try:
            (split_pos, split_size) = zip(*sorted(zip(split_pos, split_size),
                                                  key=lambda b: b[1], reverse=True))
        except ValueError:
            print("[DETECTION] split is failed")

        return split_pos

    @staticmethod
    def predict_from_model(image, model, labels):
        """
        the function to predict the label with extract model & labels

        :param image: (array) the around image per each character
        :param model: (keras model) model of the extract net
        :param labels: (LabelEncoder) labels of the extract net

        :return prediction: (string) predicted label.
        """
        image = cv2.resize(image, (80, 80))
        image = np.stack((image,) * 3, axis=-1)
        prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis, :]))])

        return prediction

    @staticmethod
    def build_tesseract_options(psm=10, is_numeric=False):
        """
        the function is to set tesseract options.

        :param psm: (number) PSM mode
        :param is_numeric: (boolean) characters to OCR
        :param alphanumeric: (number) characters to OCR

        :return: (string) the string contains tesseract options
        """

        # set the PSM mode
        options = " --psm {} --oem 3".format(psm)

        # # set the language mode

        # tell Tesseract to only ocr numeric OR alphabet characters
        if is_numeric is True:
            # set the trained data
            options += " -l digits"
        else:
            # set the trained data
            options += " -l leu"
            # whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            # options += " -c tessedit_char_whitelist={}".format(whitelist)

        # return the built option string
        return options

    def detection(self):
        """
        the function to recognize the license plate from.

        :return: none
        """

        license_list = []

        # preprocess the image
        self.preprocess_image()
        # detect the plate's region from the image
        vehicle, LpImg, cor = self.detect_plate()

        # (x, y, w, h) = cv2.boundingRect(cor)
        # crop = self.original_image[y:y + h, x:x + w]
        # cv2.imwrite('crop', crop)

        # check if there is at least one of the plate region
        if len(LpImg):
            print('[DETECTION] {} license plates detected from the image'.format(len(LpImg)))
            index = 0

            for lp in LpImg:
                # Scales, calculates absolute values, and converts the result to 8-bit.
                plate_image = cv2.convertScaleAbs(lp, alpha=255.0)
                self.debug_show("Plate image", plate_image, self.debug)
                # cv2.imwrite('{}_{}.png'.format(self.path[:-4], index), plate_image)

                text = self.extract_license(plate_image)
                if text == "failed":
                    pass
                elif 5 <= len(text) <= 11:
                    license_list.append(text)
                index += 1

            if len(license_list) > 0:
                success = True
            else:
                success = False
        else:
            success = False

        return success, license_list

    def detect_plate(self, d_max=D_MAX, d_min=D_MIN):
        """
        the function to detect license plate region from the image

        :param d_max: (int)
        :param d_min: (int)

        :return vehicle: (array) the input image(normal image)
        :return LpImg: (list) the list of the detected plate's regions
        :return cor: (array)
        """

        vehicle = self.original_image
        ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
        side = int(ratio * d_min)
        bound_dim = min(side, d_max)
        _, LpImg, _, cor = detect_lp(self.region_net, vehicle, bound_dim, lp_threshold=0.5)

        return vehicle, LpImg, cor

    def extract_license(self, plate_image):
        """
        the function to extract the license from the plate image

        :param plate_image: (image) color scale image contains license plate

        :return final_string: (string) the license text.
        """

        # convert to grayscale and blur the image
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        # Applied inverse thresh_binary
        binary = cv2.threshold(blur, 200, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        threshold_mor = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3)
        self.debug_show("Binary", binary, self.debug)

        # find contours from image
        candi_contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundingBoxes = [cv2.boundingRect(c) for c in candi_contours]

        nI = 0
        (candi_contours, boundingBoxes) = zip(*sorted(zip(candi_contours, boundingBoxes),
                                                      key=lambda b: b[1][nI], reverse=False))
        # creat a copy version "test_roi" of plat_image to draw bounding box
        test_roi = plate_image.copy()
        cv2.imwrite('1.png', threshold_mor)

        # Initialize a list which will be used to append real character contours
        candi_character = []
        # Initialize a list which will be used to append character image
        crop_characters = []

        # define standard width and height of character
        digit_w, digit_h = 20, 40

        for cont in candi_contours:
            (x, y, w, h) = cv2.boundingRect(cont)
            ratio = h / w
            if (1 <= ratio <= 3.5) & self.is_valid(candi_character, cont):  # Only select contour with defined ratio
                if h / plate_image.shape[0] >= 0.4:  # Select contour which has the height larger than 50% of the plate
                    # add the con tour to the list
                    candi_character.append(cont)

        boundingBoxes = [cv2.boundingRect(c) for c in candi_character]
        if len(boundingBoxes) == 0:
            return "failed"

        ch_height = 0
        for bound in boundingBoxes:
            ch_height += bound[3]
        ch_height /= len(boundingBoxes)

        contours = []
        for cont in candi_character:
            (x, y, w, h) = cv2.boundingRect(cont)
            if h >= (ch_height * 0.8):
                contours.append(cont)
                # Draw bounding box around digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Separate number and gibe prediction
                curr_num = threshold_mor[y:y + h, x:x + w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 230, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                character_img = np.zeros((60, 30), dtype="uint8")
                character_img[10:50, 5:25] = curr_num
                crop_characters.append(character_img)

        print("[DETECTION] {} letters detected from license plate region.".format(len(crop_characters)))
        self.debug_show("Characters", test_roi, self.debug)

        if len(contours) > 2:
            split_pos = self.split_position(contours)
        else:
            split_pos = []

        print('[DETECTION] license plate number divided into {} parts'.format(len(split_pos) + 1))

        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        # options = self.build_tesseract_options(7)
        # lpText = pytesseract.image_to_string(threshold_mor, config=options)

        final_string = ''
        for i, character in enumerate(crop_characters):
            pil_image = Image.fromarray(character)
            pil_image = ImageOps.invert(pil_image)
            # kernel = np.ones((5, 5), np.uint8)
            # cv2.dilate(pil_image, kernel, iterations=1)

            # setup the options of tesseract.
            if len(split_pos) < 2:
                options = self.build_tesseract_options()
            elif i < split_pos[1]:
                options = self.build_tesseract_options()
            else:
                options = self.build_tesseract_options(is_numeric=True)

            lpText = pytesseract.image_to_string(pil_image, config=options)
            # pil_image.save("letters/{}.png".format(lpText[0]))
            final_string += lpText[0]

            # title = np.array2string(self.predict_from_model(character, self.extract_net, self.extract_labels))
            # final_string += title.strip("'[]")

        if len(split_pos) >= 2:
            final_string = final_string[:(split_pos[0])] + '-' + final_string[(split_pos[0]):]
            final_string = final_string[:(split_pos[1] + 1)] + ' ' + final_string[(split_pos[1] + 1):]
            print('[DETECTION] license plate number is {}'.format(final_string))
        else:
            final_string = "failed"
            print('[DETECTION] license plate detect is failed')

        return final_string
