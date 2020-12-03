from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QFile, QTextStream, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap, QColor
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import breeze_resources
import os.path

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):

        self.maskedFaces = 0
        self.unmaskedFaces = 0
        # capture from web cam
        cap = cv2.VideoCapture(0)
        '''while True:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)'''

        # load our serialized face detector model from disk
        print("[INFO] loading face detector model...")
        prototxt_path = "../data/deploy.prototxt"
        weights_path = "../data/res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxt_path, weights_path)

        # load the face mask detector model from disk
        print("[INFO] loading face mask detector model...")
        # mask_model_path = '../data/mask_detector.model'
        mask_model_path = 'D:/Temp/model/mask_detector.model'
        maskNet = load_model(mask_model_path)

        # initialize the video stream and allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            frame = vs.read()
            frame = imutils.resize(frame, width=400)

            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            locs, preds = detect_and_predict_mask(frame, faceNet, maskNet)

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                if label == "Mask": self.maskedFaces += 1
                else: self.unmaskedFaces +=1
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            self.change_pixmap_signal.emit(frame)

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(974, 657)
        font = QtGui.QFont()
        font.setFamily("Inter")
        font.setPointSize(10)
        MainWindow.setFont(font)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("")
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")

        '''self.vframe = QtWidgets.QFrame(self.centralwidget)
        self.vframe.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.vframe.setFrameShadow(QtWidgets.QFrame.Raised)
        self.vframe.setObjectName("frame")
        self.horizontalLayout.addWidget(self.vframe)'''

        self.vframe = QLabel()
        self.vframe.setObjectName("vframe")
        self.horizontalLayout.addWidget(self.vframe)

        width = 640
        height = 480
        grey = QPixmap(width, height)
        grey.fill(QColor('darkGray'))
        self.vframe.setPixmap(grey)

        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 80, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_6.addWidget(self.line)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)

        '''self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setMaximumSize(QtCore.QSize(70, 16777215))
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)'''

        spacerItem2 = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setMaximumSize(QtCore.QSize(70, 16777215))
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        spacerItem4 = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem4)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem5)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_4.addWidget(self.label_5)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setMaximumSize(QtCore.QSize(70, 16777215))
        self.label_6.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_4.addWidget(self.label_6)
        spacerItem6 = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem6)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem7)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_5.addWidget(self.label_7)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setMaximumSize(QtCore.QSize(70, 16777215))
        self.label_8.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_5.addWidget(self.label_8)
        spacerItem8 = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem8)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem9)
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_8.addWidget(self.label_9)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setMaximumSize(QtCore.QSize(70, 16777215))
        self.label_10.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_8.addWidget(self.label_10)
        spacerItem10 = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem10)
        self.verticalLayout_2.addLayout(self.horizontalLayout_8)
        spacerItem11 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.verticalLayout_2.addItem(spacerItem11)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        spacerItem12 = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem12)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setMaximumSize(QtCore.QSize(200, 16777215))
        self.pushButton_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout_3.addWidget(self.pushButton_2)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setMaximumSize(QtCore.QSize(200, 16777215))
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_3.addWidget(self.pushButton)
        self.horizontalLayout_7.addLayout(self.verticalLayout_3)
        spacerItem13 = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem13)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_6.addLayout(self.verticalLayout_2)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        spacerItem14 = QtWidgets.QSpacerItem(20, 80, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.verticalLayout.addItem(spacerItem14)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.pushButton.clicked.connect(self.resetCounters)
        self.pushButton_2.clicked.connect(self.savedata)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        #self.dir_path = ""

        now = datetime.now()
        self.startTime = now.strftime("%d/%m/%Y %H:%M:%S")

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Mask Detector"))
        MainWindow.setWindowIcon(QtGui.QIcon('icon.png'))
        #self.label.setText(_translate("MainWindow", "Faces in frame:"))
        #self.label_2.setText(_translate("MainWindow", "0"))
        self.label_3.setText(_translate("MainWindow", "Total frames with face:"))
        self.label_4.setText(_translate("MainWindow", "0"))
        self.label_5.setText(_translate("MainWindow", "Frames with mask:"))
        self.label_6.setText(_translate("MainWindow", "0"))
        self.label_7.setText(_translate("MainWindow", "Frames without mask:"))
        self.label_8.setText(_translate("MainWindow", "0"))
        self.label_9.setText(_translate("MainWindow", "Percent masked:"))
        self.label_10.setText(_translate("MainWindow", "0"))
        self.pushButton_2.setText(_translate("MainWindow", "Save data to file"))
        self.pushButton.setText(_translate("MainWindow", "Reset counters"))

    def savedata(self):
        '''options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.ShowDirsOnly
        dirPath = QFileDialog.getExistingDirectory(None, "Select folder", "C:/Users/The User/Desktop/", options=options)
        if dirPath:
            self.dir_path = dirPath'''

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(None, "Save file", "C:/Users/The User/Desktop/mask_data.txt",
                                                  "Text Files (*.txt)", options=options)

        '''completeName = os.path.join(dirPath, "mask_data.txt")
        file = open(completeName, "w")'''

        file = open(fileName, "w")

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        file.write("Mask detection data (" + self.startTime + " – " + dt_string + "):\n")
        file.write("Total frames with face: " + str(self.thread.maskedFaces + self.thread.unmaskedFaces) + "\n")
        file.write("Frames with mask: " + str(self.thread.maskedFaces) + "\n")
        file.write("Frames without mask: " + str(self.thread.unmaskedFaces) + "\n")
        if (self.thread.maskedFaces + self.thread.unmaskedFaces == 0):
            file.write("Percent masked: 0.00%\n")
        else: file.write("Percent masked: " + "{:.2f}".format((self.thread.maskedFaces / (self.thread.maskedFaces + self.thread.unmaskedFaces))*100) + "%\n")

        file.close()

    def resetCounters(self):
        self.thread.maskedFaces = 0
        self.thread.unmaskedFaces = 0
        now = datetime.now()
        self.startTime = now.strftime("%d/%m/%Y %H:%M:%S")

    #@pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.vframe.setPixmap(qt_img)
        self.label_4.setText(str(self.thread.maskedFaces + self.thread.unmaskedFaces)) #total faces deteced
        self.label_6.setText(str(self.thread.maskedFaces)) #masked detections
        self.label_8.setText(str(self.thread.unmaskedFaces)) #unmasked detections
        if (self.thread.maskedFaces + self.thread.unmaskedFaces == 0): #percent masked
            self.label_10.setText("0.00%")
        else: self.label_10.setText("{:.2f}".format((self.thread.maskedFaces / (self.thread.maskedFaces + self.thread.unmaskedFaces))*100)+"%")


    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


def detect_and_predict_mask(frame, faceNet, maskNet):
    threshold = 0.5
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > threshold:
            try:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startX, startY, endX, endY = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w - 1, endX), min(h - 1, endY)

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))
            except Exception as e:
                print(e)

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding locations

    return locs, preds


def runAnalysis():
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxt_path = "../data/deploy.prototxt"
    weights_path = "../data/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxt_path, weights_path)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    #mask_model_path = '../data/mask_detector.model'
    mask_model_path = 'D:/Temp/model/mask_detector.model'
    maskNet = load_model(mask_model_path)

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        locs, preds = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    file = QFile(":/dark.qss")
    file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(file)
    app.setStyleSheet(stream.readAll())
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

