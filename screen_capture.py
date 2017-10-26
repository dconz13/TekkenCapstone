# To use this code the PS4 Remote Play window must be in the top left corner of the desired screen.
# Otherwise the offsets for the leftHPCapture and rightHPCapture variables must be updated to wrap
# the hp bars.

import time
import cv2
import mss
#from mss.windows import MSS as mss
import numpy as np
from PIL import Image

def capture_img():
    with mss.mss() as sct:
        # Part of the screen to capture
        screen = {'top': 40, 'left': 0, 'width': 640, 'height': 350}
        leftHPCapture = {'top': 62, 'left': 84, 'width': 201, 'height': 12}
        rightHPCapture = {'top': 62, 'left': 360, 'width': 201, 'height': 12}

        prevLeft = np.array(sct.grab(leftHPCapture))
        prevRight = np.array(sct.grab(rightHPCapture))
        # convert to grayscale
        prevLeft = np.dot(prevLeft[...,:3], [0.299,0.587,0.114])
        prevRight = np.dot(prevRight[...,:3], [0.299,0.587,0.114])
        # round up negatives to 0
        prevLeft = prevLeft.clip(min=0)
        prevRight = prevRight.clip(min=0)

        last_time = time.time()
        while 'Screen capturing':
            # Get raw pixels from the screen, save it to a Numpy array
            #currScreen = np.array(sct.grab(screen))
            currScreen = sct.grab(screen)
            currLeft = np.array(sct.grab(leftHPCapture))
            currRight = np.array(sct.grab(rightHPCapture))

            #convert to grayscale
            currLeft = np.dot(currLeft[...,:3], [0.299,0.587,0.114])
            currRight = np.dot(currRight[...,:3], [0.299,0.587,0.114])
            # get the difference in previous vs current
            diffLeft = prevLeft - currLeft
            diffRight = prevRight - currRight
            # round negative values up to 0
            diffLeft = diffLeft.clip(min=0)
            diffRight = diffRight.clip(min=0)
            # Check for hit on left
            if((diffLeft > 120).sum() > 10):
                print('Left was hit')
            # Check for hit on right
            if((diffRight > 120).sum() > 10):
                print('Right was hit')
            # Set previous frame data to current frame data
            prevLeft = currLeft
            prevRight = currRight

            img = Image.frombytes('RGB', currScreen.size, currScreen.rgb)
            img = img.resize((85,85), Image.LANCZOS)
            currScreen = np.array(img)
            # Uncomment to display the picture
            cv2.imshow('OpenCV/Numpy normal', currScreen)

            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            # print('fps: {}'.format(1 / (time.time()-last_time)))
            # last_time = time.time()
            #Sleep to reduce fps to 2 fps
            time.sleep(0.4)

if __name__ == "__main__":
    capture_img()
