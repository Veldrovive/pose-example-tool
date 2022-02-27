import cv2
import time
import math
import numpy as np

vid = cv2.VideoCapture(0)
cv2.namedWindow("Visual", cv2.WINDOW_AUTOSIZE)

# Get the start time
start_time = time.time()
countdown_time = 3
while True:
    ret, frame = vid.read()
    cv2.imshow("Visual", frame)
    
    count = math.ceil(countdown_time - (time.time() - start_time))
    print(count)
    if count == 0:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
