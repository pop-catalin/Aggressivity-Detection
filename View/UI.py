import tkinter as tk
from tkinter import filedialog as fd
import cv2
from PIL import ImageTk as itk
from PIL import Image
import os

class UI:
    def __init__(self, root, root_title, controller):
        self.controller = controller
        HEIGHT = 600
        WIDTH = 1000
        self.liveFlag = True
        self.root = root
        self.root.title(root_title)
        root.configure(bg='#696667')
        root.attributes("-fullscreen", True)
        self.canvas = tk.Canvas(root, bg='#696667', highlightthickness=0, 
                                height=HEIGHT, width=WIDTH)
        self.canvas.pack()

        self.buttonFrame = tk.Frame(root, bg='#964F4C', bd=10)
        self.buttonFrame.place(relx=0.75, rely=0.05, relwidth=0.22, relheight=0.4)

        self.addVideoButton = tk.Button(self.buttonFrame, text='Add Video', 
                                        font=('Courier', 16, 'bold'), bg='#567572',
                                        activebackground='#696667', 
                                        command=self.addVideoPress)
        self.addVideoButton.place(rely=0.05, relwidth=1, relheight=0.4)

        self.liveVideoButton = tk.Button(self.buttonFrame, text='Live Video', 
                                         font=('Courier', 16, 'bold'), bg='#567572',
                                         activebackground='#696667', command = self.liveVideoPress)
        self.liveVideoButton.place(rely=0.55, relwidth=0.5, relheight=0.4)
        
        self.stopLiveButton = tk.Button(self.buttonFrame, text = 'Stop Live',
                                        font=('Courier', 16, 'bold'), bg='#567572',
                                        activebackground='#696667', command = self.stopLive)
        self.stopLiveButton.place(rely=0.55, relx = 0.5, relwidth=0.5, relheight=0.4)

        self.videoFrame = tk.Frame(root, bg='#964F4C', bd=10)
        self.videoFrame.place(relx=0.02, rely=0.05, relwidth=0.7, relheight=0.6)

        self.label = tk.Label(self.videoFrame, bg='#964F4C')
        self.label.place(relwidth=1, relheight=1)
        self.img = itk.PhotoImage(Image.open(r'images\securitycamera.png').resize((300,200)))
        self.label.config(image=self.img)

        self.retrainFrame = tk.Frame(root, bg='#964F4C', bd=10)
        self.retrainFrame.place(relx=0.75, rely=0.55, relwidth=0.22, relheight=0.4)

        self.retrainButton = tk.Button(self.retrainFrame, text='Retrain Model', 
                                       font=('Courier', 16, 'bold'), bg='#567572',
                                       activebackground='#696667', 
                                       command=self.retrainModelPress)
        self.retrainButton.place(rely = 0.05, relwidth=1, relheight=0.4)
        
        self.quitButton = tk.Button(self.retrainFrame, text='Quit App',
                                    font=('Courier', 16, 'bold'), bg='#567572',
                                    activebackground='#696667',
                                    command=self.root.destroy)
        self.quitButton.place(rely=0.55, relwidth=1, relheight=0.4)

        self.messageFrame = tk.Frame(root, bg='#964F4C', bd=5)
        self.messageFrame.place(relx=0.1, rely=0.8, relwidth=0.5, relheight=0.1)

        self.messageLabel = tk.Label(self.messageFrame, font=('Courier', 28, 'bold'), 
                                     fg='#4c9658', bg='#696667')
        self.messageLabel.place(relwidth=1, relheight=1)

        self.root.mainloop()

    def addVideoPress(self):
        path = fd.askopenfilename()
        if path != '':
            self.messageLabel.configure(text='Video is loading...', fg='#964F4C')
            self.messageLabel.update_idletasks()
            
            dirname, filename = os.path.split(path)
            self.controller.setPath(dirname)
            frames = self.getFramesVideo(filename)
            try:
                if self.controller.predictVideo(frames) == True:
                    self.messageLabel.configure(text='Violence detected!', fg='red')
                else:
                    self.messageLabel.configure(text='Violence not detected!', fg='green')
            except ValueError as e:
                self.messageLabel.configure(text='Chosen file not valid!')
            except ZeroDivisionError as e:
                self.messageLabel.configure(text='Chosen file not valid')

    def retrainModelPress(self):
        path = fd.askdirectory()
        if path != '':
            self.controller.setPath(path)
            self.messageLabel.configure(text='Model is training...', fg='#964F4C')
            self.messageLabel.update_idletasks()
            self.controller.retrainModel()
            self.messageLabel.configure(text='Model is done training!', fg='#964F4C')
            self.messageLabel.update_idletasks()
     
    def liveVideoPress(self):
        self.liveFlag = True
        self.startLive()
           
    def startLive(self):
        if self.liveFlag == True:
            frames = self.getFramesWebcam()
            if self.controller.predictWebcamVideo(frames) == True:
                self.messageLabel.configure(text = 'Violence detected!', fg='red')
            else:
                self.messageLabel.configure(text = 'Violence not detected!', fg='green')
            self.root.after(1000, self.startLive)
    
    def stopLive(self):
        self.liveFlag = False
        self.label.config(image=self.img)
        self.cap.release()
    
    def showFrame(self, frame):
        resized = cv2.resize(frame, (self.label.winfo_width(), self.label.winfo_height()))
        img = itk.PhotoImage(image=Image.fromarray(resized))
        
        self.label.image = img
        self.label.config(image=img)
        self.label.update_idletasks()
    
    def getFramesWebcam(self):
        self.cap = cv2.VideoCapture(0)
        frames = []
        while len(frames) < 100:
            ret, frame = self.cap.read()
            if ret == False:
                break
            else:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.showFrame(image)
                frames.append(image)
        return frames
    
    def getFramesVideo(self, filename):
        frames = []
        self.cap = cv2.VideoCapture(self.controller.getPath() + '\\' + filename)
        nrOfFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while len(frames) < nrOfFrames:
            ret, frame = self.cap.read()
            if ret == False:
                break
            else:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.showFrame(image)
                frames.append(image)
        return frames
