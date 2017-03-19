import numpy as np
from skimage import data
import matplotlib.pyplot as plt

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
    plt.subplot(2, 3, 4)
    plt.title("Inverse Radon transform image")
    plt.imshow(inverseRadonImage, cmap='gray')

    plt.show()


if __name__ == "__main__":
    main()
