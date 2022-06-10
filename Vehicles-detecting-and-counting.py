import cv2
import numpy as np
# from time import sleep

centered_coordinate = []
cars_counter = 0
# 	Vehicle Detecting and Counting
get_center = lambda x, y, w, h: (x + w//2, y + h//2)

path = r"D:\Projects\Computer Vision\Vehicle Detecting and Counting\video.mp4"
cap = cv2.VideoCapture(path)
ww = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"X264")
output_path = r"D:\Projects\Computer Vision\Vehicle Detecting and Counting\output.mp4"
out = cv2.VideoWriter(output_path, fourcc, 30, (ww, hh))

subtract = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    done , frame = cap.read()
    if done==True:
        # tempo = float(1/60)
        # sleep(tempo) 
        grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey,(3,3),5)
        subtracted = subtract.apply(blur)
        dilated = cv2.dilate(subtracted,np.ones((5,5)))
        # dilated = cv2.dilate(subtracted,None, iterations=5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx (dilated, cv2. MORPH_CLOSE , kernel)
        closed = cv2.morphologyEx (closed, cv2. MORPH_CLOSE , kernel)
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

        cv2.line(frame, (25, 550), (1200, 550), (255,127,0), 3)
        for(i,c) in enumerate(contours):
            (x,y,w,h) = cv2.boundingRect(c)
            if not ((w >= 80) and (h >= 80)):
                continue

            cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),2)
            center = get_center(x, y, w, h)
            centered_coordinate.append(center)
            cv2.circle(frame, center, 4, (0, 0,255), -1)

            for (x,y) in centered_coordinate:
                if y<(550 + 6) and y>(550 - 6):
                    cars_counter+=1
                    cv2.line(frame, (25, 550), (1200, 550), (0,127,255), 3)  
                    centered_coordinate.remove((x,y))
                    # print("No of detected cars : "+str(cars_counter))        
           
        cv2.putText(frame, "VEHICLE COUNT : "+str(cars_counter), (350, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)

        
        cv2.imshow("Video Original" , frame)
        # cv2.imshow("Detectar",dilated)
        
        out.write(frame)
        
        if cv2.waitKey(1) == ord('f'):
            break
    else: break


cv2.destroyAllWindows()
cap.release()
out.release()