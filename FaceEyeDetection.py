import cv2
import customtkinter
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

root = customtkinter.CTk()
root.geometry('175x425')
root.resizable(0,0)
root.title('Face & Iris Detection Module')

camFrame = customtkinter.CTkLabel(root) #To ingrain our webcam into a ctkinter window, we need to setup a label as a placeholder.
camFrame.place(x = 0, y = -30) 						
cap = cv2.VideoCapture(0)
destroyInterval = 0

def ShowCamFrame(): 
	global destroyInterval #We are using the variable initialized earlier at the global level 
	destroyInterval = destroyInterval + 1 #Increases by 1 every ~10ms
	root.geometry('900x425')
	ret, frame = cap.read() #Reads in the input from webcam capture 
	
	faces = cFace.detectMultiScale(frame,
								scaleFactor=1.3, #scaleFactor reduces image size by x amount (1.3 = 30%), higher reduction may increase performance
								minNeighbors=4, 
								minSize=(30, 30))
	for x,y,w,h in faces:
				frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(69, 0, 243),1) #Draws rectangle upon detetected faces
																			#Last 2 parameters hold values for color and stroke width
	numOfFacesText = "Found {0} face(s)!".format(len(faces))
	facelabel = CreateFaceLabel(numOfFacesText) #Not sure if this is the greates solution here, but my familiarity with tkinter is lacking
											#I don't want to continously initialize new labels, so we'll try and destroy the labels every
											#so often, based upon the destruction interval.

	#Using "retinas" and "eyes" as equivalencies, though of course they are not exactly synonymous.
	#Eye detection produces many more false positives in comparison to face detection. Lower false positives,
	#especially upon the nostrils, can be avoided by increasing the minSize value.
	retinas = cRetina.detectMultiScale(frame,
								scaleFactor=1.3, 
								minNeighbors=4, 
								minSize=(30, 30))
	for x,y,w,h in retinas:
				frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(63, 153, 243),1)
	
	numOfEyesText = "Found {0} eye(s)!".format(len(retinas))
	eyelabel = CreateEyesLabel(numOfEyesText) 

	cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

	cam = Image.fromarray(cv2image).resize((720, 490)) #This is the webcam displayed in GUI's resolution
	camtk = ImageTk.PhotoImage(image = cam) 
	camFrame.camtk = camtk
	camFrame.place(x=185) 
	camFrame.configure(image=camtk)
	
	if (((destroyInterval % 15) == 0) and (destroyInterval != 0)):
		DestroyFaceLabel(facelabel)
		DestroyEyeLabel(eyelabel)
		#Don't want to continously stack new labels, so destroy the labels every
		#so often, based upon the destruction interval.

	camFrame.after(25, ShowCamFrame) #Updates webcam frame every 10ms, increasing number may increase performance.

def CreateFaceLabel(numOfFacesText):
	numOfFaces_label = customtkinter.CTkLabel(master = root, text = numOfFacesText)
	numOfFaces_label.place(x=15, y=165)
	return numOfFaces_label

def CreateEyesLabel(numOfEyesText):
	numOfEyes_label = customtkinter.CTkLabel(master = root, text = numOfEyesText)
	numOfEyes_label.place(x=15, y=200)
	return numOfEyes_label

def DestroyFaceLabel(numOfFaces_label):
	numOfFaces_label.destroy()

def DestroyEyeLabel(numOfEyes_label):
	numOfEyes_label.destroy()

def CloseWebcams(): #This function was sort of buggy so I've removed it for now. Camera usage should be freed upon full program exit anyway.
	cv2.destroyAllWindows() 
	cap.release()

def GetFaceFile():
	return filedialog.askopenfilename(title = 'Select xml for face detection dataset')

def GetEyesFile():
	return filedialog.askopenfilename(title = 'Select xml for eyes detection dataset')

def CreateCC():
	global cFace, cRetina 
	#It doesn't really matter which order the user selects the files in, but the rectangle drawing settings were
	#designed specifically for face and eyes sizes.
	tk.messagebox.showinfo("Dataset Prompt","Select the XML file for face detection.")
	faceDataset = GetFaceFile()
	tk.messagebox.showinfo("Dataset Prompt","Select the XML file for eyes detection.")
	eyeDataset = GetEyesFile()
	cFace = cv2.CascadeClassifier(faceDataset) #Sets up the CascadeClassifier, based upon folder selection.
	cRetina = cv2.CascadeClassifier(eyeDataset)


enableCam_button = customtkinter.CTkButton(
                        text="Enable Webcam",
                        
                        text_font="none 10",
                        text_color="white",
						fg_color="#515663",
                        
                        width=150,
                        height=30,
						command = lambda: [ShowCamFrame()])

enableCam_button.place(x = 15, y = 80)

selectData_button = customtkinter.CTkButton(
                        text="Select dataset folder",
                        
                        text_font="none 10",
                        text_color="white",
                        
                        width=150,
                        height=30,
						command = lambda: [CreateCC()])

selectData_button.place(x = 15, y = 20)

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"

root.mainloop()