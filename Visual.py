import PIL.Image
import tkinter as tk
from tkinter import *
from tkinter import messagebox, filedialog
import glob
from PIL import ImageTk, Image
import cv2
from Live_detect_face import FaceDetectionApp

########################################################################################################################
#Function for the first button command for Detect Face
def open_image():
    file_path=file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        display_image_on_canvas(file_path, canvas)

def detect_faces(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

def display_image_on_canvas(image_path, canvas):
    # Detect faces in the image
    image_with_faces = detect_faces(image_path)
    # Convert the image to RGB format
    image_with_faces_rgb = cv2.cvtColor(image_with_faces, cv2.COLOR_BGR2RGB)
    # Convert the image to PIL format
    image_pil = Image.fromarray(image_with_faces_rgb)
    # Convert the PIL image to Tkinter-compatible format
    image_tk = ImageTk.PhotoImage(image_pil)
    # Set the image on the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
    # Keep a reference to the image to prevent it from being garbage collected
    canvas.image = image_tk

###############################################################################################################################
############################################################################################################################
def live_capture():
    intance=Live_detect_face.FaceDetectionApp

#####################################################################################################################


def detect_object():
    thres = 0.5

    # img = cv2.imread('haitu.jpg')
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    success, img = cap.read()

    while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):  # flatten = so object
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 10, box[1] - 10),
                            cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

        cv2.imshow("Output", img)
        # Wait for 5 seconds (500 milliseconds)
        key = cv2.waitKey(5000)
        # If the user presses any key or the time elapses, break out of the loop
        if key == -1:
            cv2.destroyAllWindows()





##################################################################################################################



###################################################################################################################
# Create a Tkinter window
window = tk.Tk()
window.title("Visual")
window.iconbitmap("home.ico")
window.geometry("1200x720")

window.configure(bg="#a28fcc")
window.resizable(width=False, height=False)
###########################################################################################################
# Create a Tkinter canvas
canvas = tk.Canvas(window,width=800, height=600)
canvas.place(x=30, y=50)
canvas.pack()


########################################################################################################

#create a menubar
menubar = Menu(window, background='#ff8000', foreground='black', activebackground='white', activeforeground='black')
file = Menu(menubar, tearoff=1, background='#4680a7', foreground='black')
file.add_command(label="New")
file.add_command(label="Open")
file.add_command(label="Save")
file.add_command(label="Save as")
file.add_command(label="Detect")
file.add_separator()
file.add_command(label="Exit", command=window.quit)
menubar.add_cascade(label="File", menu=file)

edit = Menu(menubar, tearoff=0)
edit.add_command(label="Undo")
edit.add_separator()
edit.add_command(label="Cut")
edit.add_command(label="Copy")
edit.add_command(label="Paste")
menubar.add_cascade(label="Edit", menu=edit)

help = Menu(menubar, tearoff=0)
help.add_command(label="About")
menubar.add_cascade(label="Help", menu=help)

window.config(menu=menubar)





#############################################################################################
#create a button to open images
open_button=tk.Button(window, text="Detect Face",command=open_image, fg="white", bg="green", width=10, height=2, activebackground="Red", font=("Helvetica",15,'bold'))
open_button.place(x=1050, y=120)

#button to detect face
detect_face_button=tk.Button(window, text="Live Capture",command=lambda: live_capture ,fg="white", bg="green", width=10, height=2,activebackground="Red", font=("Helvetica",15,'bold'))
detect_face_button.place(x=1050, y=190)

#button for live detect face
live_capture_button=tk.Button(window, text="Detect_Object",command=detect_object, fg="white", bg="green", width=12, height=2,activebackground="Red", font=("Helvetica",15,'bold'))
live_capture_button.place(x=1040, y=260)








###########################################################


window.mainloop()
