import cv2
from scipy.spatial import distance as dist
import imutils
from imutils import perspective, contours
import numpy as np
import time



def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


# initialize camera

cap = cv2.VideoCapture("Test_Video.mp4")
start_time=time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        orig = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        (cnts, _) = contours.sort_contours(cnts)

        pixles_to_size = None

        for c in cnts:
            if cv2.contourArea(c) < 100:
                continue

            bbox = cv2.minAreaRect(c)
            bbox = cv2.cv.BoxPoints(bbox) if imutils.is_cv2() else cv2.boxPoints(bbox)
            box = np.array(bbox, dtype="int")

            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if pixles_to_size is None:
                pixles_to_size = 4  # change value as per camera calibration

            dimA = dA / pixles_to_size
            dimB = dB / pixles_to_size

            area = dimA * dimB

            # --------------------- MAKE CLASS HERE ------------------------------------

            if area <= 300:
                cv2.putText(orig, "object is Pin", (int(tltrX - 45), int(tltrY - 25)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 0, 0), 2)

            if 550 >= area > 300:
                cv2.putText(orig, "object 2", (int(tltrX - 45), int(tltrY - 25)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 0, 125), 2)

            if 550 < area <= 1150:
                cv2.putText(orig, "object is sharpener", (int(tltrX - 45), int(tltrY - 25)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 255), 2)

            if 1150< area >1300:
                cv2.putText(orig, "object 4", (int(tltrX-45), int(tltrY - 25)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,(180, 150, 45), 2)

            if 1300 < area <= 2000:
                cv2.putText(orig, "object is cap", (int(tltrX - 45), int(tltrY - 25)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 255, 0), 2)

            if area > 2000 :
                cv2.putText(orig, "object is connector", (int(tltrX - 45), int(tltrY - 25)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 255, 0), 2)


            cv2.putText(orig, "{:.2f}mm".format(dimB), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255), 2)
            cv2.putText(orig, "{:.2f}mm".format(dimA), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2)

        # print("-----%s seconds---" % (time.time()-start_time))

        orig = cv2.resize(orig, (780, 940))
        frame = cv2.resize(frame, (780, 940))
        cv2.imshow('frame', np.hstack([orig, frame]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break