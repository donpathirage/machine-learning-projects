import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
rect, digits, x, y = [], [], 0, 0
font = cv2.FONT_HERSHEY_SIMPLEX
kernel = np.ones((3,3),np.uint8)

def show_webcam(model, mirror=False):
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        rect = []
        _, img = cam.read()

        if mirror:
            img = cv2.flip(img, 1)

        im_copy = img.copy()

        gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        blur = cv2.GaussianBlur(gray,(21,21),0)
        blur = cv2.medianBlur(blur, 11)
        th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)


        dilation = cv2.dilate(th3, kernel, iterations = 2)



        _, contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            x = c[:,0][:,0]
            y = c[:,0][:,1]
            minx = np.min(x)
            maxx = np.max(x)
            miny = np.min(y)
            maxy = np.max(y)

            rect.append([minx, maxx, miny, maxy])

        digits = []
        r2 = []
        for r in rect:
            try:
                digit = dilation[r[2]-5 : r[3]+5, r[0]-5 : r[1]+5]
                digit = cv2.erode(digit, kernel, iterations = 3)

                digit = cv2.resize(digit, (28,28), interpolation = cv2.INTER_AREA)
#                 digit = cv2.erode(digit, kernel, iterations = 1)


                digits.append(digit)

                r2.append(r)


            except:
                pass

        prediction, ind = 0, 0
        for i in range(len(digits)):
            classifications = []
            prediction = model.predict(np.array(np.round(digits[i]/255)).reshape(1,28,28,1))
            ind = np.where(prediction > 0.8)
            for j in ind:
                if(len(j)):
                    classifications.append(j[0])


            if(len(classifications)):
                prediction = np.max(classifications)
                if(prediction != 10):
                    im_copy = cv2.rectangle(im_copy,(r2[i][0],r2[i][2]),(r2[i][1],r2[i][3]),(0,0,255),1)
                    cv2.putText(im_copy, str(prediction), (r2[i][0],r2[i][2]), font, 1, (175,255,155))



        cv2.imshow('Original ', im_copy)


        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()


    return digits




model = tf.keras.models.load_model('MNIST_Classifier-v3-ALL.model')   #Lower classifcation accuracy than v2 but much less noisy
                                                                      #Trained on an 11th 'no digit' class generated from bernoulli distribution
digits = show_webcam(model, mirror=False)
