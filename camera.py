import cv2
import time
import datetime


capture = cv2.VideoCapture(0)
face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
recording = False
frame_size = (int(capture.get(3)), int(capture.get(4)))
cc = cv2.VideoWriter_fourcc(*"mp4v")

recording_stopped = None
timer = False
after_detection_recording = 5


while True:
    _, frame = capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)
    bodies = face.detectMultiScale(gray, 1.3, 5)
    
     
    if len(faces) + len(bodies) > 0:
        if recording:
            timer = False
        else:
            recording = True
            curr_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{curr_time}.mp4", cc, 20, frame_size)
            print("Recording started")
    elif recording:
        if timer:
            if time.time() - recording_stopped >= after_detection_recording:
                recording = False
                timer = False
                out.release()
                print("Recording stopped")
        else:
            timer = True
            recording_stopped = time.time()    
    if recording:
        out.write(frame)
        
    
    
    cv2.imshow("camera", frame)
    if cv2.waitKey(1) == ord("q"): #hit "Q" key to quit the program
        break
 
out.release()   
capture.release()
cv2.destroyAllWindows()