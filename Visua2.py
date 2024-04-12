import PIL.Image
import tkinter as tk
from tkinter import *
from tkinter import messagebox, filedialog
import glob
from PIL import ImageTk, Image
import cv2
#from Live_detect_face import FaceDetectionApp


def open_image():
    file_path=file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        display_image_on_canvas(file_path, canvas)




############################################################################################################################
def live_capture():


    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    class FaceDetectionApp:
        def __init__(self, window1, window_title):
            self.window1= window1
            self.window1.title(window_title)

            # Open the video capture device (default webcam)
            self.video_capture = cv2.VideoCapture(0)

            # Create a canvas to display the video stream
            self.canvas = tk.Canvas(window1, width=self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                    height=self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.canvas.pack()

            # Button to start/stop face detection
            self.btn_start_stop = tk.Button(window1, text="Start Detection", command=self.toggle_detection)
            self.btn_start_stop.pack(fill=tk.BOTH, expand=True)

            # Variable to track if face detection is active
            self.detecting = False

            # Start the update loop
            self.update()

            self.window1.protocol("WM_DELETE_WINDOW", self.on_close)

        def toggle_detection(self):
            self.detecting = not self.detecting
            if self.detecting:
                self.btn_start_stop.config(text="Stop Detection")
            else:
                self.btn_start_stop.config(text="Start Detection")

        def update(self):
            # Read frame from the video stream
            ret, frame = self.video_capture.read()

            if self.detecting:
                # Convert frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the frame
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # Draw rectangles around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Convert the frame to RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to a PIL Image
            img = Image.fromarray(rgb_frame)

            # Convert PIL Image to ImageTk format
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the canvas with the new image
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

            # Schedule the next update
            self.window1.after(10, self.update)

        def on_close(self):
            # Release the video capture object
            self.video_capture.release()
            # Close the window
            self.window1.destroy()

    # Create a Tkinter window
    window1 = tk.Tk()
    app = FaceDetectionApp(window1, "Face Detection")
    window1.mainloop()


########################################################################################################################

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
open_button=tk.Button(window, text="Open Image",command=open_image, fg="white", bg="green", width=10, height=2, activebackground="Red", font=("Helvetica",15,'bold'))
open_button.place(x=1050, y=120)

#button to detect face
detect_face_button=tk.Button(window, text="Detect Face",fg="white", bg="green", width=10, height=2,activebackground="Red", font=("Helvetica",15,'bold'))
detect_face_button.place(x=1050, y=190)

#button for live detect face
live_capture_button=tk.Button(window, text="Live Capture",command=live_capture, fg="white", bg="green", width=10, height=2,activebackground="Red", font=("Helvetica",15,'bold'))
live_capture_button.place(x=1050, y=260)


#button for object detect
object_button=tk.Button(window, text="Detect Object",fg="white", bg="green", width=13, height=2,activebackground="Red", font=("Helvetica",15,'bold'))
object_button.place(x=1030, y=330)





###########################################################

# Load and display the image with faces detected
#image_path = filedialog.askopenfilename(title="Select an Image",
    #                                filetypes=(("Image files", "*.jpg *.jpeg *.png"),), initialdir="C:/Users/Lenovo pc/PycharmProjects/Artificial/6 SEM PROJECT/people")




# Start the Tkinter event loop

window.mainloop()
