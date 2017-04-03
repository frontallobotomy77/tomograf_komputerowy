import numpy as np
import skimage
from numpy.core.umath import rad2deg
from skimage import data
from matplotlib import cm
import matplotlib.pyplot as plt
from tkinter import *
from PIL import Image, ImageTk
from dicom.dataset import Dataset, FileDataset
import datetime

slave = Tk()
master = Tk()
iteracyjny = 0

def get_values():
    values = np.arange(3)
    values[0] = stepSlider.get()
    values[1]=detectorsNumberSlider.get()
    if values[1]%2==0:
        values[1] += 1
    values[2]=detectorWidthSlider.get()
    return values

def update_image(array, label):
    tmpArray = array/max(array.flatten())
    tmpArray *= 255
    tmpImage = Image.fromarray(tmpArray)
    tmpPhoto = ImageTk.PhotoImage(tmpImage)
    label.configure(image=tmpPhoto)
    master.update()

def BresenhamAlgorithm(input, A, B, output=None, moreThanZeroValues=True, returnOrDraw=True, lineColor=0.5):
    if returnOrDraw and output is None: output = []
    if not returnOrDraw:
        if output is None: raise NameError("output must be given")
        outSizeX, outSizeY = len(output[0]), len(output)

    inputSizeX, inputSizeY,  = len(input[0]), len(input)
    X, Y = int(A[0]), int(A[1])
    X2, Y2 = int(B[0]), int(B[1])
    dx, dy = abs(X-X2), abs(Y-Y2)
    xAdd, yAdd = 1 if X<X2 else -1, 1 if Y<Y2 else -1

    def bresenhamLoop(X,Y,output):
        if returnOrDraw and X >= 0 and Y >= 0 and X < inputSizeX and Y < inputSizeY:
            color = input[inputSizeY - 1 - int(Y)][int(X)]
            if not moreThanZeroValues or color > 0: output.append(color)
        if not returnOrDraw and X>=0 and Y>=0 and Y<outSizeY and X < outSizeX:
            output[outSizeY - 1 - int(Y)][int(X)] += lineColor
        return X+xAdd,Y+yAdd,output

    if dx >= dy :
        yAdd = float(abs(Y-Y2))/abs(X-X2)*yAdd
        while X != X2: X,Y,output = bresenhamLoop(X,Y,output)
    else:
        xAdd = float(abs(X-X2))/abs(Y-Y2)*xAdd
        while Y != Y2: X,Y,output = bresenhamLoop(X,Y,output)
    return output

def radonCircleLoop(input,label, output, stepsArray, step, center, circleRadius, detectorsNumber, detectorsWidth, inverse=False):
    detectorDistance = (circleRadius * 2 * detectorsWidth / 180) / detectorsNumber

    for stepAngle in stepsArray:
        centralEmiterPos = (center[0] + circleRadius * np.sin(np.radians(stepAngle)),center[1] + np.cos(np.radians(stepAngle)) * circleRadius)
        centralReceiverPos = (center[0] - circleRadius * np.sin(np.radians(stepAngle)),center[1] - np.cos(np.radians(stepAngle)) * circleRadius)

        for currentDetector in range(0,detectorsNumber):
            distanceFromMainDetector = (currentDetector - (detectorsNumber / 2)) * detectorDistance

            cos = np.cos(np.radians(stepAngle))
            sin = np.sin(np.radians(stepAngle))
            emiterPos = centralEmiterPos[0] + distanceFromMainDetector * cos, centralEmiterPos[1] - distanceFromMainDetector * sin
            receiverPos = centralReceiverPos[0] + distanceFromMainDetector * cos, centralReceiverPos[1] - distanceFromMainDetector * sin
            if not inverse:
                points = BresenhamAlgorithm(input, emiterPos, receiverPos)
                if len(points) > 0: output[currentDetector][int(stepAngle/step)] = sum(points)  # Normalizacja
            else:
                color = input[currentDetector, int(stepAngle/step)]
                output = BresenhamAlgorithm(input, emiterPos, receiverPos, output, returnOrDraw=False, lineColor=color)
        if(iteracyjny==1):
            update_image(output, label)
    return output

def radonTransform(input,label, stepSize=1, stepsArray=None, output=None, detectorsNumber=100, detectorsWidth=140):
    if stepsArray is None: stepsArray = np.arange(0,180,stepSize)
    xSize = int(180/stepSize+1)

    if output is None: output = np.zeros((detectorsNumber,xSize))

    circleRadius = np.sqrt(np.power(len(input)/2,2)+np.power(len(input[0])/2,2) )
    center = (len(input[0])/2,len(input)/2)
    output = radonCircleLoop(input, label, output, stepsArray, stepSize, center, circleRadius,detectorsNumber,detectorsWidth)

    output /= max(output.flatten()) #Normalizacja
    return output

#Filtracja
def filterKernel(signal,detectorsNumber):
    out = np.zeros(detectorsNumber)
    middle = int(len(signal)/2)
    out[middle]=1
    for k in range(1,middle,2):
        out[middle - k] = out[middle + k] = (-4 / np.square(np.pi) / np.square(k))
    return out

def convolution1D(signal, mask):
    out = np.zeros(len(signal))
    maskSize, signalSize = len(mask),len(signal)
    for X in range(0,signalSize):
        for h in range(-int(maskSize/2),int(maskSize/2+1)):
            cX = X + h
            if cX < 0: cX += signalSize
            if cX >= signalSize: cX -= signalSize
            out[X] += signal[cX] * mask[h + int(maskSize / 2)]
    return out

def fourierLoop(sinogram,detectorsNumber):
    xSize, ySize = sinogram.shape
    output = np.zeros(shape=(xSize, ySize))
    for k in range(ySize):
        signal = sinogram[:, k]
        mask = filterKernel(signal, detectorsNumber=detectorsNumber)
        filteredSignal = convolution1D(signal,mask)
        output[:, k] = filteredSignal
    return output

def inverseRadonTransform(input, label, stepSize=1, stepsArray=None, detectorsWidth=140, output=None, outputWidth=None, outputHeight=None):
    if stepsArray is None: stepsArray = np.arange(0,181, stepSize)
    if output is None:
        if outputHeight is None: outputHeight = len(input)
        if outputWidth is None: outputWidth = outputHeight
        output = np.zeros((outputHeight,outputWidth))

    circleRadius = np.sqrt(np.power(outputWidth/2,2)+np.power(outputHeight/2,2) )
    center = (outputWidth/2,outputHeight/2)
    output = radonCircleLoop(input,label, output, stepsArray, stepSize, center,circleRadius,len(input),detectorsWidth,inverse=True)

    output -= min(output.flatten())
    output /= max(output.flatten())

    return output

def saveDicomFile(filename, patientName, patientId, gender, birthday, imageArray, transpose=False):

    meta = Dataset()
    SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    meta.MediaStorageSOPClassUID = SOPClassUID
    date=datetime.datetime.now().strftime('%Y%m%d')
    time=datetime.datetime.now().strftime('%H%M%S.%f')
    randomUId = SOPClassUID + "."+date+time
    meta.MediaStorageSOPInstanceUID = randomUId
    meta.ImplementationClassUID = randomUId+"."+"1"

    dataSet = FileDataset(filename, {}, file_meta=meta, preamble=b"\0"*128)
    dataSet.PatientName = patientName
    dataSet.PatientID=patientId
    dataSet.PatientBirthDate = birthday
    dataSet.PatientSex = gender
    dataSet.is_little_endian=True
    dataSet.is_implicit_VR=True
    dataSet.ContentDate = date
    dataSet.StudyDate = date
    dataSet.StudyTime = time
    dataSet.ContentTime=time
    dataSet.StudyInstanceUID = randomUId+"."+"2"
    dataSet.SeriesInstanceUID = randomUId+"."+"3"
    dataSet.SOPInstanceUID = randomUId+"."+"4"
    dataSet.SOPClassUID = "CT."+date+time

    dataSet.SamplesPerPixel = 1
    dataSet.PhotometricInterpretation = "MONOCHROME2"
    dataSet.PixelRepresentation = 0
    dataSet.HighBit = 15
    dataSet.BitsStored = 16
    dataSet.BitsAllocated = 16
    dataSet.SmallestImagePixelValue = b'\\x00\\x00'
    dataSet.LargestImagePixelValue = b'\\xff\\xff'
    dataSet.Rows = imageArray.shape[1]
    dataSet.Columns = imageArray.shape[0]
    if imageArray.dtype != np.uint16:
        imageArray = skimage.img_as_uint(imageArray)
        if transpose == True:
            dataSet.Rows = imageArray.shape[0]
            dataSet.Columns = imageArray.shape[1]
    dataSet.PixelData = imageArray.tostring()
    dataSet.save_as(filename)

def calculate():
    parameters = get_values()
    step = parameters[0]
    detectorsNumber = parameters[1]
    detectorWidth = parameters[2]

    show_slave()
    inData = data.imread("resources/in.png", as_grey=True)

    inputImage = Image.open('resources/in.png')
    resized = inputImage.resize((150, 150), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(resized)
    l5.configure(image=photo)
    master.update_idletasks()

    radonImage = radonTransform(inData, l10, stepSize=step, detectorsNumber=detectorsNumber, detectorsWidth=detectorWidth)

    radonImage*=255.0
    radon = Image.fromarray(radonImage)
    radonPhoto = ImageTk.PhotoImage(radon)
    l10.configure(image=radonPhoto)
    master.update_idletasks()

    fourier = fourierLoop(radonImage, detectorsNumber=detectorsNumber)

    filtered=Image.fromarray(fourier)
    filteredPhoto = ImageTk.PhotoImage(filtered)
    l11.configure(image = filteredPhoto)
    master.update()

    inverseRadonImage = inverseRadonTransform(radonImage, l12, stepSize=step, detectorsWidth=detectorWidth)

    inverseRadonImage*=255
    inv = Image.fromarray(inverseRadonImage)
    invPhoto = ImageTk.PhotoImage(inv)
    l12.configure(image = invPhoto)
    master.update()

    inverseFiltered = inverseRadonTransform(fourier, l13, stepSize=step, detectorsWidth=detectorWidth)
    saveDicomFile("out.dcm", "Andrzej", "1234", "male", "20070304", inverseFiltered)

    inverseFiltered*=255
    invFil = Image.fromarray(inverseFiltered)
    invFilPhoto = ImageTk.PhotoImage(invFil)
    l13.configure(image=invFilPhoto)
    master.update()
    input("Press any key to clear")

def show_slave():
    l4.grid(row=0, column=0)
    l5.grid(row=0, column=1)
    l6.grid(row=1, column=0)
    l10.grid(row=1, column=1)
    l7.grid(row=2, column=0)
    l11.grid(row=2, column=1)
    l8.grid(row=3, column=0)
    l12.grid(row=3, column=1)
    l9.grid(row=4, column=0)
    l13.grid(row=4, column=1)

def calculate_immediately():
    global iteracyjny
    iteracyjny = 0
    calculate()

def calculate_iterative():
    global iteracyjny
    iteracyjny = 1
    calculate()

l4 = Label(slave,justify=LEFT,padx = 10,text="Obraz wejściowy")
l5 = Label(slave)
l6 = Label(slave,justify=LEFT,padx = 10,text="Sinogram")
l10 = Label(slave)
l7 = Label(slave,justify=LEFT,padx = 10,text="Sinogram po filtracji")
l11= Label(slave)
l8 = Label(slave,justify=LEFT,padx = 10,text="Obraz wyjściowy")
l12= Label(slave)
l9 = Label(slave,justify=LEFT,padx = 10,text="Obraz wyjściowy po filtracji")
l13= Label(slave)

l1 = Label(master,justify=LEFT,padx = 10,text="Liczba detektorów (tylko nieparzyste)").pack()
detectorsNumberSlider = Scale(master, from_=101,to=251, orient=HORIZONTAL)
detectorsNumberSlider.pack()
l2 = Label(master,justify=LEFT,padx = 10,text="Szerokość kątowa detektorów").pack()
detectorWidthSlider = Scale(master, from_=90, to=150, orient=HORIZONTAL)
detectorWidthSlider.pack()
l3 = Label(master,justify=LEFT,padx = 10,text="Kąt obrotu tomografu").pack()
stepSlider = Scale(master, from_=1, to=5, orient=HORIZONTAL)
stepSlider.pack()
l14 = Label(master,justify=LEFT,padx=10,text="Wybierz sposób obliczeń :").pack()
calculateButton = Button(master, text="Wykonaj obliczenia natychmiastowo", command=calculate_immediately)
calculateButton.pack()
bbutton = Button(master, text="Wykonaj obliczenia iteracyjnie", command=calculate_iterative)
bbutton.pack()


def main():
    master.mainloop()

if __name__ == "__main__":
    main()
