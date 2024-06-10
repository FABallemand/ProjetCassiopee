import os
import numpy as np
import cv2
import imutils
from keras.models import load_model
import screen_brightness_control as sbc

# global variables
bg = None

label_to_class = {0: "blank",
                  1: "fist",
                  2: "five",
                  3: "ok",
                  4: "thumbsdown",
                  5: "thumbsup"}

label_to_action = {0: "",
                   1: "Minimum brightness",
                   2: "Maximum brightness",
                   3: "",
                   4: "Decrease brightness",
                   5: "Increase brightness"}


def _load_weights():
    model = load_model("model.keras")
    print(model.summary())
    # print(model.get_weights())
    # print(model.optimizer)
    return model


def run_avg(image, accum_weight):
    global bg

    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, accum_weight)


def segment(image, threshold=25):
    global bg

    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def get_predicted_label(model, img):
    global label_to_class

    img = cv2.resize(img, (100, 100))
    img = img.reshape(1, 100, 100, 1)

    prediction = model.predict(img)

    predicted_label = np.argmax(prediction)
    
    return predicted_label


def main():
    # load model
    model = _load_weights()

    # initialize accumulated weight
    accum_weight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # get camera frame rate
    fps = int(camera.get(cv2.CAP_PROP_FPS))

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    # volume percentage
    brightness = 50
    sbc.set_brightness(brightness)

    # Text to print
    text = ""
    text_position = (50, 500)

    # define calibration function
    def calibration():
        # to get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        print("[STATUS] calibrating...")
        for i in range(30):
            # get the current frame
            (grabbed, frame) = camera.read()

            # resize the frame
            frame = imutils.resize(frame, width=700)
            
            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)

            # get and process ROI
            roi = frame[top:bottom, right:left]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.GaussianBlur(roi, (7, 7), 0)

            # perform calibration
            run_avg(roi, accum_weight)
        print("[STATUS] calibration successfull")

    # perform calibration once
    calibration()
    
    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)
        
        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get and process ROI
        roi = frame[top:bottom, right:left]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.GaussianBlur(roi, (7, 7), 0)
        
        # segment the hand region
        hand = segment(roi)
        if hand is not None:
            # unpack the thresholded image and segmented region
            (thresholded, segmented) = hand

            # draw the segmented region and display the frame
            cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

            # prediction
            if num_frames % (fps / 6) == 0:
                # make prediction
                predicted_label = get_predicted_label(model, thresholded)
                predicted_class = label_to_class[predicted_label]
                predicted_action = label_to_action[predicted_label]
                text = predicted_class + " - " + predicted_action

                # adjust brightness
                if predicted_class == "Fist":
                    brightness = 0
                    sbc.set_brightness(brightness)
                elif predicted_class == "Five":
                    brightness = 100
                    sbc.set_brightness(brightness)
                elif predicted_class == "Thumbsup":
                    brightness += 2
                    sbc.set_brightness(brightness)
                elif predicted_class == "Thumbsdown":
                    brightness -= 2
                    sbc.set_brightness(brightness)

            # display prediction
            cv2.putText(clone, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 4)
                    
            # show the thresholded image
            cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        # display the frame with segmented hand
        cv2.imshow("Projet Cassiopee 24", clone)

        # perform calibration
        if num_frames % (fps / 1000) == 0:
            calibration()
            num_frames = 0

        # increment the number of frames
        num_frames += 1

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

    # free up memory
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Change working directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    main()