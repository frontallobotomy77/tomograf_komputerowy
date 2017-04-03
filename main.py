import numpy as np
import skimage
from skimage import data
import matplotlib.pyplot as plt
from dicom.dataset import Dataset, FileDataset
import datetime, time

step = 2
detectorsNumber = 120
detectorWidth = 130


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

def radonCircleLoop(input, output, stepsArray, step, center, circleRadius, detectorsNumber, detectorsWidth, inverse=False):
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
    return output

def radonTransform(input, stepSize=1, stepsArray=None, detectorsNumber=100, detectorsWidth=140, output=None):
    if stepsArray is None: stepsArray = np.arange(0,180,stepSize)
    xSize = int(180/stepSize+1)

    if output is None: output = np.zeros((detectorsNumber,xSize))

    circleRadius = np.sqrt(np.power(len(input)/2,2)+np.power(len(input[0])/2,2) )
    center = (len(input[0])/2,len(input)/2)
    output = radonCircleLoop(input,output, stepsArray, stepSize, center, circleRadius,detectorsNumber,detectorsWidth)

    output /= max(output.flatten()) #Normalizacja
    return output

def inverseRadonTransform(input, stepSize=1, stepsArray=None, detectorsWidth=140, output=None, outputWidth=None, outputHeight=None):
    if stepsArray is None: stepsArray = np.arange(0,181, stepSize)
    if output is None:
        if outputHeight is None: outputHeight = len(input)
        if outputWidth is None: outputWidth = outputHeight
        output = np.zeros((outputHeight,outputWidth))

    circleRadius = np.sqrt(np.power(outputWidth/2,2)+np.power(outputHeight/2,2) )
    center = (outputWidth/2,outputHeight/2)
    output = radonCircleLoop(input,output, stepsArray, stepSize, center,circleRadius,len(input),detectorsWidth,inverse=True)

    output -= min(output.flatten())
    output /= max(output.flatten())

    return output

def saveDicomFile(filename, patientName, patientId, gender, birthday, imageArray, transpose=False):

    meta = Dataset()
    SOPClassUID = "1.2.840.10008.5.1.4.1.1.2" # sop class UID dla obrazow CT
    meta.MediaStorageSOPClassUID = SOPClassUID  # Wygenerowany unikalny UID
    date=datetime.datetime.now().strftime('%Y%m%d') # Obecny czas
    time=datetime.datetime.now().strftime('%H%M%S.%f') # Obecny czas
    randomUId = SOPClassUID + "."+date+time # Wygenerowany unikalny UID
    meta.MediaStorageSOPInstanceUID = randomUId # Wygenerowany unikalny UID
    meta.ImplementationClassUID = randomUId+"."+"1" # Wygenerowany unikalny UID

    dataSet = FileDataset(filename, {}, file_meta=meta, preamble=b"\0"*128) # Utworzenie obiektu DICOM
    dataSet.PatientName = patientName   # Imie pacjenta
    dataSet.PatientID=patientId # Id pacjenta
    dataSet.PatientBirthDate = birthday # Data urodzenia pacjenta
    dataSet.PatientSex = gender # Plec pacjenta
    dataSet.is_little_endian=True
    dataSet.is_implicit_VR=True
    dataSet.ContentDate = date  # Czas utworzenia pliku (YYYY:MM:DD)
    dataSet.StudyDate = date    # Czas ostatniego otworzenia obrazu (YYYY-MM-DD)
    dataSet.StudyTime = time    # Czas ostatniego otworzenia obrazu (HH:MM:SS)
    dataSet.ContentTime=time    # Czas utworzenia pliku (HH:MM:SS)
    dataSet.StudyInstanceUID = randomUId+"."+"2"   # Wygenerowany unikalny UID
    dataSet.SeriesInstanceUID = randomUId+"."+"3"   # Wygenerowany unikalny UID
    dataSet.SOPInstanceUID = randomUId+"."+"4"   # Wygenerowany unikalny UID
    dataSet.SOPClassUID = "CT."+date+time   # Wygenerowany unikalny UID

    dataSet.SamplesPerPixel = 1 # Liczba kanałów. 1 - dla skali szarosci
    dataSet.PhotometricInterpretation = "MONOCHROME2" # MONOCHROE - obraz jest w skali szarości, 2 - maksymalna wartosc wskazuje kolor bialy
    dataSet.PixelRepresentation = 0 # 0 - wartosci sa tylko dodatnie (unsigned) 1 - wartosci sa tez ujemne
    dataSet.HighBit = 15    # Najważniejszy bit w pliku z obrazem
    dataSet.BitsStored = 16 # Liczba bitow na jedna wartosc w obrazie
    dataSet.BitsAllocated = 16  # Liczba bitow na jedna wartosc ktora jest zaalokowana dla obrazu
    dataSet.SmallestImagePixelValue = b'\\x00\\x00' # Wskazanie minimalnej wartosci dla kanalu
    dataSet.LargestImagePixelValue = b'\\xff\\xff'  # Wskazanie maksymalnej wartosci dla kanalu
    dataSet.Rows = imageArray.shape[1]  # Liczba wierszy
    dataSet.Columns = imageArray.shape[0]   # Liczba kolumn
    if imageArray.dtype != np.uint16:   # Sprawdzenie czy wartosci sa w przedziale [0,255]
        imageArray = skimage.img_as_uint(imageArray)    # Zamiana na wartosci w przedziale [0,255]
        if transpose == True:   # Zamiana wierszy i kolumn (opcjonalne)
            dataSet.Rows = imageArray.shape[0]
            dataSet.Columns = imageArray.shape[1]
    dataSet.PixelData = imageArray.tostring()   # Zapisanie obrazu
    dataSet.save_as(filename)   # Zapisanie pliku na dysku


def main():
    inData = data.imread("resources/in.png", as_grey=True)

    plt.subplot(2, 3, 1)
    plt.title("Original image")
    plt.imshow(inData, cmap='gray')

    plt.subplot(2, 3, 2)
    plt.xlabel("Emiter/detector rotation")
    plt.ylabel("Number of receiver")
    plt.title("Radon transform image")

    radonImage = radonTransform(inData, stepSize=step, detectorsNumber=detectorsNumber, detectorsWidth=detectorWidth)
    plt.imshow(radonImage, cmap='gray', extent=[0,180,len(radonImage),0], interpolation=None)
    inverseRadonImage = inverseRadonTransform(radonImage, stepSize=step, detectorsWidth=detectorWidth)
    saveDicomFile("out.dcm", "Andrzej", "1234", "male", "20070304", inverseRadonImage)
    #write_dicom(inverseRadonImage,'pretty.dcm')
    plt.subplot(2, 3, 4)
    plt.title("Inverse Radon transform image")
    plt.imshow(inverseRadonImage, cmap='gray')

    plt.show()


if __name__ == "__main__":
    main()
