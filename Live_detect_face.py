import tkinter as tk
import cv2
from PIL import Image, ImageTk

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class FaceDetectionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Open the video capture device (default webcam)
        self.video_capture = cv2.VideoCapture(0)

        # Create a canvas to display the video stream
        self.canvas = tk.Canvas(window, width=self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Button to start/stop face detection
        self.btn_start_stop = tk.Button(window, text="Start Detection", command=self.toggle_detection)
        self.btn_start_stop.pack(fill=tk.BOTH, expand=True)

        # Variable to track if face detection is active
        self.detecting = False

        # Start the update loop
        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

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
        self.window.after(10, self.update)

    def on_close(self):
        # Release the video capture object
        self.video_capture.release()
        # Close the window
        self.window.destroy()

# Create a Tkinter window
window = tk.Tk()
app = FaceDetectionApp(window, "Face Detection")
window.mainloop()
