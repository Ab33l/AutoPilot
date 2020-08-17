import cv2

# car Image File
img_file = "car_traffic.jpg"
video = cv2.VideoCapture("Tesla.mp4")
video1 = cv2.VideoCapture("Pedestrians.mp4")

# pre-Trained Car Classifier
car_tracker_classifier = "car_detector.xml"
pedestrian_tracker_classifier = "pedestrians_detector.xml"

#create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_classifier)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_classifier)

# create opencv image
img = cv2.imread(img_file)

#convert to greyscale(required for haas cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

while True:

    #reads the current frame
    (read_successful, frame) = video1.read()

    if read_successful:
        #must convert to greyscale
        greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and pedestrians
    cars = car_tracker.detectMultiScale(greyscale_frame)
    pedestrisns = pedestrian_tracker.detectMultiScale(greyscale_frame)

    # highlight cars with rectanges
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x + 1, y + 1), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # highlight pedestrians with rectanges
    for (x, y, w, h) in pedestrisns:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # display image with spotted faces
    cv2.imshow("Car Detector", frame)

    # prevent autoclose. Display image while waiting for key press
    key = cv2.waitKey(0)

    #if Q is pressed then quit window
    if key == 81 or key == 113:
        break

#release VideoCapture Object
video1.release()