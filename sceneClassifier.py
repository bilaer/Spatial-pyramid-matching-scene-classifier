from pythonCV.lm import makeLMfilters
from pythonCV.cv import image
from pythonCV.cv import Filter
import copy
import random
import os
import PIL
import numpy as np

class sceneRec(object):
    def __init__(self):
        # The structure of visualWordsDic is as such:
        # K = [id, [pos], {data1, data2,....}]
        self.visualWordsDic = dict()
        self.visualWordsColorTags = dict()
        self.filterBank = self.initalFilter()
        # the structure of this dictionary is:
        # {dirName:{image1, image2}, dirName:{image1, image2}...}
        self.trainingImageNameDic = dict()
        self.trainingImageHisto = dict()
        self.appFilter = Filter()
        self.trainingImagePath = "G:/SUN397"
        self.trainingImageListFile = "ClassName.txt"
        self.trainResultFile = "responses.txt"
        self.trainDirName = "trainData"
        self.testDirName = "testData"
        self.kResult = "kResult.txt"
        self.testResult = "testResult.txt"
        # Number of pixels sampling
        self.alpha = 120
        # Number of dictionary words
        self.k = 200

    def setTrainingFilePath(self, path):
        self.trainingFilePath = path

    def setAlpha(self, alpha):
        self.alpha = alpha

    def setK(self, k):
        self.k = k

    def initalFilter(self):
        return makeLMfilters()

    # Generate alpha number of random position for
    # collecting the filter responses
    def genRandomPos(self, ranNum, lox, hix, loy, hiy):
        result = []
        hasAdd = set()
        for i in range(ranNum):
            while True:
                x = random.randint(lox, hix)
                y = random.randint(loy, hiy)
                if (x, y) not in hasAdd:
                    result.append((x, y))
                    hasAdd.add((x, y))
                    break
        return result

    # Selectively choose some of the filters in the LM filter bank
    def selectFilter(self, filterBanks, removeList):
        result = []
        for i in range(len(filterBanks)):
            if i not in removeList:
                result.append(filterBanks[i])
        return result

    # Read the paths of all training images and add them into the set for training
    def readTrainingImagePath(self, numOfClass):
        print("read the training files")
        trainingImageDirs = []
        with open(self.trainingImagePath + "/" + self.trainingImageListFile) as f:
            for dir in f:
                trainingImageDirs.append(dir[:-1])

        # Open each one of the dir and get the path of training images
        print("Adding the path of training image...")
        for dir in trainingImageDirs:
            count = 1
            self.trainingImageNameDic[dir] = []
            for imageFile in os.listdir(self.trainingImagePath + dir):
                imagePath = self.trainingImagePath + dir + "/" + imageFile
                self.trainingImageNameDic[dir].append(imagePath)
            count = count + 1
            if count == numOfClass:
                break


    # Main function that is used to train and store the training data
    def filteringWithTrainingImage(self, numOfImagesPerClass, largestImageWidth, largestImageHeight):
        # Get all the training files
        self.readTrainingImagePath(numOfImagesPerClass)

        if not os.path.exists(self.trainDirName):
            os.makedirs(self.trainDirName)

        print("writing the response")
        # Open the training files and conv it with filters bank
        count = 1
        for dir in self.trainingImageNameDic:
            count = 1
            print("current file: %d" %count)
            for fileName in self.trainingImageNameDic[dir]:
                # Get the filter response
                if self.convWithFilterBank(fileName, fileName, largestImageWidth, largestImageHeight):
                    count = count + 1
                    if count == numOfImagesPerClass:
                        break

    # Helper function that conv image with filters in filter bank
    def convWithFilterBank(self, imageName, fileName, largestImageWidth, largestImageHeight):
        # Initialize
        try:
            print(fileName)
            trainImage = image(fileName)
        except IOError:
            print("cannot not open this image")
            return False
        else:
            channel = trainImage.channels

            if channel == None:
                return False

            rchannel, gchannel, bchannel = channel[0], channel[1], channel[2]

            # Restrict the size of image
            if rchannel.shape[0] > largestImageHeight or rchannel.shape[1] > largestImageWidth:
                return False

            # Create a new directory for holding the training images
            if not os.path.exists(self.trainDirName):
                os.makedirs(self.trainDirName)
            newDir = "trainData/" + imageName[imageName.index(":") + 2:imageName.index(".")]
            print(newDir)
            if not os.path.exists(newDir):
                os.makedirs(newDir)

            # Conv each filter
            count = 1
            for filter in self.filterBank:
                # Add padding
                print("start conv with %d filter" %count)
                r = int((filter.shape[0] - 1)/2)
                rcpad = self.appFilter.padding(rchannel, rchannel.shape[1], rchannel.shape[0], r)
                gcpad = self.appFilter.padding(gchannel, gchannel.shape[1], gchannel.shape[0], r)
                bcpad = self.appFilter.padding(bchannel, bchannel.shape[1], bchannel.shape[0], r)

                # Conv it with filter in the filter bank
                rcconv = self.appFilter.conv(rcpad, rcpad.shape[1], rcpad.shape[0], 1, filter, r)
                gcconv = self.appFilter.conv(gcpad, gcpad.shape[1], gcpad.shape[0], 1, filter, r)
                bcconv = self.appFilter.conv(bcpad, bcpad.shape[1], bcpad.shape[0], 1, filter, r)

                print("finished filter %d" % count)
                # Modify and return the results
                print("correcting the result")
                rcconv = self.appFilter.correctDifference(np.array(rcconv))
                gcconv = self.appFilter.correctDifference(np.array(gcconv))
                bcconv = self.appFilter.correctDifference(np.array(bcconv))

                result = np.full((rcconv.shape[0], rcconv.shape[1], 3), 0)
                for i in range(result.shape[0]):
                    for j in range(result.shape[1]):
                        result[i][j][0] = rcconv[i][j]
                        result[i][j][1] = gcconv[i][j]
                        result[i][j][2] = bcconv[i][j]
                fileName = newDir + ("/filter%d.jpg" %count)
                im = PIL.Image.fromarray(np.uint8(result), "RGB")
                im.save(fileName)
                count = count + 1
            return True

    # Randomly choose some sample points from every filtered training images
    def generateSamplePts(self, numOfPointsPerImage):

        # Helper function that recursively handle the training images in the folders
        def generateSamplePtsHelp(currentAddress, numOfPointsPerImage):
            dirOrFiles = os.listdir(currentAddress)
            for dirOrFile in dirOrFiles:
                newAddress = currentAddress + "/" + dirOrFile
                # This is the directory that hold filtered images of training the image, handle the image
                if os.path.isfile(newAddress):
                    # Handle the data
                    print("handling training file: %s ..." %currentAddress)
                    samples = []
                    for fileName in dirOrFiles:
                        trainImage = image(currentAddress + "/" + fileName)
                        temp = []
                        rchannel, gchannel, bchannel = trainImage.channels
                        ranPos = self.genRandomPos(numOfPointsPerImage, 0, rchannel.shape[1] - 1,
                                                   0, rchannel.shape[0] - 1)
                        for x, y in ranPos:
                            temp.append((rchannel[y][x], gchannel[y][x], bchannel[y][x]))
                        samples.append(temp)

                    # Parse the data
                    result = []
                    for i in range(len(samples[0])):
                        temp = []
                        for j in range(len(samples)):
                            temp = temp + [samples[j][i][0], samples[j][i][1], samples[j][i][2]]
                        result.append(temp)

                    # Write into the files
                    responseFile = open(self.trainResultFile, "a+")
                    for sample in result:
                        responseFile.write(str(sample) + "\n")
                    responseFile.close()

                else:
                    # This not a training image, goes to next directory
                    generateSamplePtsHelp(newAddress, numOfPointsPerImage)

        # Open the images
        if os.path.exists(self.trainDirName):
            dirs = os.listdir(self.trainDirName)
            for dir in dirs:
                generateSamplePtsHelp(self.trainDirName + "/" + dir, numOfPointsPerImage)

    # Given two vectors, return euclidean distance between two distances
    def calDistance(self, first, second):
        return np.linalg.norm(second - first)

    # Helper function that calculate the distance for k-means algorithm
    # and return the k with shortest distance
    def shortestK(self, sample):
        distances = dict()
        for k in self.visualWordsDic:
            # We want to minimize the square of distance
            distances[self.calDistance(self.visualWordsDic[k][1], sample)**2] = k
        return distances[sorted(distances.keys())[0]]

    # Calculating the averages of data in order to update the k
    def calNewDistance(self, data, k):
        # Ignore the empty centroids
        if len(self.visualWordsDic[k][2]) == 0:
            return self.visualWordsDic[k][1]
        else:
            avg = np.zeros(len(self.visualWordsDic[k][1]))
            for dataLine in range(len(self.visualWordsDic[k][2])):
                avg = avg + self.trainDataProcessing(data[dataLine])
            return (1/len(self.visualWordsDic[k][2]))*avg

    # Return true if centroids of k means remain stable, e.g, the value
    # of centroids didn't change after a iteration
    def areCentroidsStable(self, old, new):
        for i in range(len(old)):
            if (new[i]).all() != (old[i]).all():
                print("not equal")
                return False
        return True

    # Print out the rss of given the current centroids
    def printRSS(self, data):
        rss = 0
        for key in self.visualWordsDic:
            for line in self.visualWordsDic[key][2]:
                dataVector = self.trainDataProcessing(data[line])
                rss = self.calDistance(dataVector, self.getPosOfK(key))**2 + rss
        return rss


    # Initialize the dictionary to the data structure given below
    # {k:[id, initial position, {data1, data2...}, k2:...}
    def initVisaulWordDict(self, id, initialPos):
        self.visualWordsDic[id] = [id, np.array(initialPos), set()]

    # Assign a points to a given k in dictionary
    def assignSampleToK(self, k, sample):
        self.visualWordsDic[k][2].add((sample))

    # Get the current vector of k
    def getPosOfK(self, k):
        return self.visualWordsDic[k][1]

    # Update the vector of centroids
    def changePosOfk(self, k, newPos):
        self.visualWordsDic[k][1] = newPos

    # Clean the set of data assigned to k for next iteration
    def cleanAssignedPointsOfK(self, k):
        self.visualWordsDic[k][2] = set()

    # Filter more training files into the dataset
    def addMoreTrainingImage(self, imageClassPath, savePath, numOfImagesPerClass,
                             largestImageWidth, largestImageHeight):
        print("get the directory")
        imageFilesName = os.listdir(imageClassPath)
        trainImageDirName = os.listdir(savePath)
        print("start ending more training images")
        for imageName in imageFilesName:
            if imageName[:imageName.index(".")] not in trainImageDirName:
                filePath = imageClassPath + "/" + imageName
                if self.convWithFilterBank(filePath, filePath, largestImageWidth, largestImageHeight):
                    count = count + 1
                    if count == numOfImagesPerClass:
                        break

    # Algorithm that is used to classified the image responses
    def kmeans(self, k, dimension, iteration, outerKmean=False):
        with open(self.trainResultFile, "r") as f:
            data = f.readlines()
            # Picking up samples from data randomly as initialization
            print("randomly choose k centers...")
            if not outerKmean:
                hasAdd = set()
                for i in range(k):
                    while True:
                        initialK = []
                        for j in range(dimension):
                            initialK.append(random.randint(0, 255))
                        if tuple(initialK) not in hasAdd:
                            self.initVisaulWordDict(i, np.array(initialK))
                            hasAdd.add(tuple(initialK))
                            break

            for _ in range(iteration):
                # Clean all the samples assigned to this k before assigning new data into the dictionary
                for key in self.visualWordsDic:
                    self.cleanAssignedPointsOfK(key)

                print("assign values to its closest centers...")
                # Open the stored responses and assign every values to its closest k
                for i in range(len(data)):
                    print("data left: %d" %(len(data) - i - 1))
                    dataVector = self.trainDataProcessing(data[i])
                    # Get the min k:
                    minK = self.shortestK(dataVector)
                    # Assign the values to the nearest k
                    self.assignSampleToK(minK, i)

                # Update the k values
                print("update the centers...")
                old, new = [0]*k, [0]*k
                count = -1
                for key in self.visualWordsDic:
                    count = count + 1
                    print("updating the distance of the %dth key" %count)
                    newDistance = self.calNewDistance(data, key)
                    # Get the data for calculating the distances changes
                    old[key] = copy.deepcopy(self.getPosOfK(key))
                    new[key] = newDistance
                    # Update the delta list, check if the centroids have becomes stable
                    self.changePosOfk(key, newDistance)

                # Check if k remain stable and quit if the centroids are remains stable
                rss = self.printRSS(data)
                with open("kmeanResult.txt", "a+") as f:
                    f.write("kwords: %d, it: %d rss: %f \n" %(k, _, rss))
                #if self.areCentroidsStable(old, new):
                #    print("centers are stable right now")
                #    break

        # Save the k-mean result
        print("writing the result into the file")
        with open(self.kResult, "w") as f:
            for key in self.visualWordsDic:
                # Ignore the empty centroids
                if len(self.visualWordsDic[key][2]) != 0:
                    result = str(key) + ":" + str(self.visualWordsDic[key][1]) + "\n"
                    f.write(result)
                else:
                    print("centroid %d has no points assigned " %int(key))

    # Processing the data trained for k-mean clustering
    # Give a list that has string like list in python, convert
    # it into the a list and then return as a numpy array
    def trainDataProcessing(self, str):
        result, i = [], 1
        while i < len(str):
            end = i
            while True:
                if str[end] == "," or str[end] == "]":
                    result.append(float(str[i:end]))
                    i = end + 2
                    break
                else:
                    end = end + 1
        return np.array(result)

    def visualWordsDataProcessing(self, str):
        result, i, end = [], 0, 0
        while not str[i].isdigit():
            i = i + 1
        while i < len(str) and end < len(str):
            end = i
            if str[end] == "]":
                break
            while end < len(str):
                if (str[end].isdigit() and (str[end + 1] == " " or str[end + 1] == "]")) \
                    or (str[end] == "." and (str[end + 1] == " " or str[end + 1] == "]")):
                    result.append(float(str[i:end + 1]))
                    i = end + 1
                    break
                elif str[end] == " ":
                    i = i + 1
                    break
                else:
                    end = end + 1
        return np.array(result)

    # Read the vectors of visual words that previously trained to the file
    # and parse it into ndarray
    def readVisaulWordsData(self):
        count = 0
        with open(self.kResult, "r") as f:
            data = f.readlines()
            for i in range(len(data)):
                if ":" in data[i]:
                    pos = data[i][data[i].index(":") + 1:-1]
                    j = i + 1
                    while "]" not in data[j]:
                        pos = pos + data[j][:-1]
                        j = j + 1
                    pos = pos + data[j][:data[j].index("]") + 1]
                    pos = self.visualWordsDataProcessing(pos)
                    self.visualWordsDic[count] = [count, pos, set()]
                    count = count + 1

    # Randomly assign some colors to the dictionary keys
    def assignColorTagsToK(self):
        hasTag = set()
        with open("colorTag.txt", "a+") as f:
            for key in self.visualWordsDic:
                while True:
                    colorTag = []
                    for i in range(3):
                        colorTag.append(random.randint(0, 255))
                    if tuple(colorTag) not in hasTag:
                        self.visualWordsColorTags[key] = colorTag
                        #result = str(key) + ":" + str(colorTag) + "\n"
                        #f.write(result)
                        hasTag.add(tuple(colorTag))
                        break

    def visualWord(self, filterImageArrayList, currentAddress, computeVisualWord=True):
        # Assign k to every pixels in the training images and save as a image
        print("assign pixels to its closest color tag...")
        if computeVisualWord:
            resultImageR = np.zeros(filterImageArrayList[0].channels[0].shape)
            resultImageG = np.zeros(filterImageArrayList[0].channels[0].shape)
            resultImageB = np.zeros(filterImageArrayList[0].channels[0].shape)
        histo = np.zeros(filterImageArrayList[0].channels[0].shape)
        for i in range(filterImageArrayList[0].channels[0].shape[0]):
            for j in range(filterImageArrayList[1].channels[0].shape[1]):
                # The data that has the same data alignment as the training data
                sample = []
                for k in range(len(filterImageArrayList)):
                    sample = sample + [filterImageArrayList[k].channels[0][i][j],
                                       filterImageArrayList[k].channels[1][i][j],
                                       filterImageArrayList[k].channels[2][i][j]]
                # print("closest k is: %d" %self.shortestK(sample))
                # color = self.visualWordsColorTags[self.shortestK(np.array(sample))]
                shortestK = self.shortestK(np.array(sample))
                histo[i][j] = shortestK
                if computeVisualWord:
                    color = self.visualWordsColorTags[shortestK]
                    resultImageR[i][j] = color[0]
                    resultImageG[i][j] = color[1]
                    resultImageB[i][j] = color[2]
            print("current row: %d" % i)

        if computeVisualWord:
            print("saving current visual word image...")
            result = np.zeros((resultImageR.shape[0], resultImageR.shape[1], 3))
            result[:, :, 0] = resultImageR
            result[:, :, 1] = resultImageG
            result[:, :, 2] = resultImageB
            im = PIL.Image.fromarray(np.uint8(result), "RGB")
            im.save(currentAddress + "/" + "visualWord.jpg")

        print("saving histo data")
        address = currentAddress + "/" + "histo.txt"
        np.savetxt(address, histo.astype(int), fmt="%s", delimiter=" ", newline="\r\n")

    # Compute the images bases upon the visual words obtained from clustering.
    def computeVisualWordsOfImage(self, start=None):
        # Helper function that recursively handle the training images in the folders
        def computeVisualWordsHelp(currentAddress):
            dirOrFiles = os.listdir(currentAddress)
            for dirOrFile in dirOrFiles:
                newAddress = currentAddress + "/" + dirOrFile
                # This is the directory that hold filtered images of training the image, handle the image
                if os.path.isfile(newAddress):
                    # Handle the data
                    print("handling training file: %s ..." % currentAddress)
                    filterImageArrayList = []
                    # Get all the data from the filter images
                    for filterImage in dirOrFiles:
                        if "filter" not in filterImage:
                            filterImageArrayList.append(image(currentAddress + "/" + filterImage))
                    # Compute the visual words and histo and save the files
                    self.visualWord(filterImageArrayList, currentAddress)
                    return
                else:
                    # This not a training image, goes to next directory
                    computeVisualWordsHelp(newAddress)

        # Open the images
        dirs = os.listdir(self.trainDirName)
        for dir in dirs:
            computeVisualWordsHelp(self.trainDirName + "/" + dir)

    # Normalize the histogram for the convenience of computation
    def normalizeHisto(self, histo):
        total = np.sum(histo)
        result = np.float32(histo)*(1/total)
        return result

    # Pass the image and the boundaries of areas needed for calculations
    def calHisto(self, trainImage, lowRow, highRow, lowCol, highCol):
        Histo = [0]*len(self.visualWordsDic)
        for i in range(lowRow, highRow):
            for j in range(lowCol, highCol):
                Histo[trainImage[i][j]] = Histo[trainImage[i][j]] + 1
        return Histo

    # Use the separate the image into several cells and calculate the histograms
    # I use 3 levels model here
    def pyramidMatching(self, trainImage):
        # Calculate the level 0 and normalize it
        height, width = trainImage.shape[0], trainImage.shape[1]
        level0Histo = self.calHisto(trainImage, 0, height, 0, width)
        level0Histo = self.normalizeHisto(level0Histo)

        # Calculate the level 1 histogram and normalize it
        level1Histo = []
        for i in range(0, height, height//2 + 1):
            for j in range(0, width, width//2 + 1):
                level1Histo = level1Histo + self.calHisto(trainImage, i, min(i + height//2, height),
                                            j, min(j + width//2, width))
        level1Histo = self.normalizeHisto(level1Histo)

        # Calculate the level 2 histogram and normalize it
        level2Histo = []
        for i in range(0, height, height//4 + 1):
            for j in range(0, width, width//4 + 1):
                level2Histo = level2Histo + self.calHisto(trainImage, i, min(i + height//4, height),
                                             j, min(j + width//4, width))
        level2Histo = self.normalizeHisto(level2Histo)

        # Integrate three level histo into one vector
        result = np.append(np.append(0.25*np.array(level0Histo), 0.25*np.array(level1Histo)), 0.5*np.array(level2Histo))
        # Check if the level has the correct length
        assert(level0Histo.shape == (188,))
        assert(level1Histo.shape == (752,))
        assert(level2Histo.shape == (3008,))
        assert(result.shape == (3948,))
        return result

    # Preprocess the histogram data into the a list
    def processHistoData(self, file):
        result = []
        for data in file:
            temp = data.split()
            temp = [int(temp[i]) for i in range(len(temp))]
            result.append(temp)
        return np.array(result)

    # Recursively travel through all the training images in the file and calculate the
    # histogram of those images
    def processTrainHist(self):
        # Helper function that recursively handle the training images in the folders
        def computeHisto(currentAddress):
            dirOrFiles = os.listdir(currentAddress)
            for dirOrFile in dirOrFiles:
                newAddress = currentAddress + "/" + dirOrFile
                # This is the directory that hold filtered images of training the image, handle the image
                if os.path.isfile(newAddress):
                    print("current image: %s" %currentAddress)
                    histoFile = open(currentAddress + "/" + "histo.txt", "r")
                    histoRaw = self.processHistoData(histoFile)
                    pyramid = self.pyramidMatching(histoRaw)
                    histoFile.close()
                    return pyramid
                else:
                    # Assign the label of this image
                    result = computeHisto(newAddress)
                    if result is not None:
                        print(os.path.basename(currentAddress))
                        className = os.path.basename(currentAddress)
                        if className not in self.trainingImageHisto:
                            self.trainingImageHisto[className] = [result]
                        else:
                            self.trainingImageHisto[className].append(result)

        # Open the training images directory
        dirs = os.listdir(self.trainDirName)
        for dir in dirs:
            computeHisto(self.trainDirName + "/" + dir)

    # Compare the test image with training image using sum(min(x,y))
    def compareImage(self, pyramid):
        assert(len(self.trainingImageHisto) != 0)
        trainingImageDict = dict()
        for key in self.trainingImageHisto:
            for trainPyramid in self.trainingImageHisto[key]:
                sum = 0
                for i in range(len(trainPyramid)):
                    sum = sum + min(trainPyramid[i], pyramid[i])
                trainingImageDict[sum] = key
        return trainingImageDict[sorted(trainingImageDict.keys())[-1]]

    # Testing the overall accuracy given a set of image
    def test(self):
        error = [0]
        imageNum = [0]
        def testHelp(currentAddress):
            dirOrFiles = os.listdir(currentAddress)
            for dirOrFile in dirOrFiles:
                newAddress = currentAddress + "/" + dirOrFile
                # This is the directory that hold filtered images of training the image, handle the image
                if os.path.isfile(newAddress):
                    print("current error: %d" %error[0])
                    print("current image: %d" %imageNum[0])
                    # Handle the data
                    print("handling training file: %s ..." % currentAddress)
                    filterImageArrayList = []
                    # Get all the data from the filter images
                    for filterImage in dirOrFiles:
                        if filterImage != "visualWords.jpg":
                            filterImageArrayList.append(image(currentAddress + "/" + filterImage))
                    # Compute the visual words and histo and save the files
                    self.visualWord(filterImageArrayList, currentAddress, True)
                    # Get the pyramid and compare
                    file = open(currentAddress + "/" + "histo.txt", "r")
                    histoRaw = self.processHistoData(file)
                    pyramid = self.pyramidMatching(histoRaw)
                    predictLabel = self.compareImage(pyramid)
                    file.close()
                    imageNum[0] = imageNum[0] + 1
                    return predictLabel
                else:
                    predictLabel = testHelp(newAddress)
                    if predictLabel is not None:
                        trueLabel = os.path.basename(currentAddress)
                        result = "predict: %s, true: %s" % (predictLabel, trueLabel) + "\n"
                        print(result)
                        if trueLabel != predictLabel:
                            error[0] = error[0] + 1
                        file = open(self.testResult, "a+")
                        file.write(result)
                        file.close()

        dirs = os.listdir(self.testDirName)
        for dir in dirs:
            testHelp(self.testDirName + "/" + dir)

        # Calculate the general accuracy of prediction
        result = "general accuracy: %f" %(1 - error[0]/imageNum[0])
        print(result)
        file = open(self.testResult, "a+")
        file.write(result)
        file.close()

#test = sceneRec()
#testFilter = Filter()
#test.readVisaulWordsData()
#test.kmeans(150, 144, 2, True)
#test.assignColorTagsToK()
#test.computeVisualWordsOfImage()







