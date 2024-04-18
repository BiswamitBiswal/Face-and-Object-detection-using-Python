import PIL.Image
import tkinter as tk
from tkinter import *
from tkinter import messagebox, filedialog
import glob
from PIL import ImageTk, Image
import cv2

def about_us():
    print("Hii")
    print("This project is developed by Biswamit Biswal and Rajeev Tirkey")
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
        cv2.putText(image, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
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
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize video capture object
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces and put a label
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Exit the loop if 'q' is pressed or the window is closed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Face Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release video capture object
    cap.release()
    cv2.destroyAllWindows()


#####################################################################################################################

def detect_object():
    thres = 0.5

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


        key=cv2.waitKey(1) & 0xFF
        # If the 'q' key is pressed, break from the loop and close the window
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


###################################################################################################################
# Create a Tkinter window
window = tk.Tk()
window.title("Dristi")
window.iconbitmap("home.ico")
window.geometry("1200x720")

window.configure(bg="#6c6457")
window.resizable(width=False, height=False)
###########################################################################################################
# Create a Tkinter canvas
canvas = tk.Canvas(window,width=800, height=600)
#canvas.configure(bg='gray')
canvas.place(x=30, y=50)
canvas.pack()


########################################################################################################

#create a menubar
menubar = Menu(window, background='#ff8000', foreground='black', activebackground='white', activeforeground='black')
file = Menu(menubar, tearoff=1, background='#4680a7', foreground='black')
file.add_command(label="New", command=open_image)
file.add_command(label="Open", command=open_image)
file.add_command(label="Save")
file.add_command(label="Save as")
file.add_command(label="Detect face", command=open_image)
file.add_command(label="Live Capture", command=live_capture)
file.add_command(label="Detect Object", command=detect_object)
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
help.add_command(label="About", command=about_us)
menubar.add_cascade(label="Help", menu=help)

window.config(menu=menubar)


#############################################################################################
#create a button to open images
open_button=tk.Button(window, text="Detect Face",command=open_image, fg="white", bg="green", width=10, height=2, activebackground="Red", font=("Helvetica",15,'bold'))
open_button.place(x=1050, y=120)

#button to detect face
detect_face_button=tk.Button(window, text="Live Capture",command=live_capture ,fg="white", bg="green", width=10, height=2,activebackground="Red", font=("Helvetica",15,'bold'))
detect_face_button.place(x=1050, y=190)

#button for live detect face
live_capture_button=tk.Button(window, text="Detect_Object",command=detect_object, fg="white", bg="green", width=12, height=2,activebackground="Red", font=("Helvetica",15,'bold'))
live_capture_button.place(x=1040, y=260)

###########################################################


window.mainloop()
