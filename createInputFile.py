from PIL import Image
import numpy
import random


def randomFileTrain():
    trainData = open("train.csv","w")

    all_data = []

    for i in range(0,500):
        nume = "ped/"
        if i<10:
            nume += "img_" + "0000" + str(i)+".pgm"
        if i>=10 and i<100:
            nume += "img_" + "000" + str(i)+".pgm"
        if i>=100:
            nume += "img_" + "00" + str(i)+".pgm"
    
        img = Image.open(nume)
        arr = numpy.array(img)  

        A = numpy.ravel(arr)
        A = numpy.append(A,1)

        all_data.append(A)

    for i in range(0,500):
        nume = "nonped/"
        if i<10:
            nume += "img_" + "0000" + str(i)+".pgm"
        if i>=10 and i<100:
            nume += "img_" + "000" + str(i)+".pgm"
        if i>=100:
            nume += "img_" + "00" + str(i)+".pgm"
    
        img = Image.open(nume)
        arr = numpy.array(img)

        A = numpy.ravel(arr)
        A = numpy.append(A,0)

        all_data.append(A)

    random.shuffle(all_data)

    for a in all_data:
        for i in range(len(a)):
            trainData.write(str(a[i]))
            if i != (len(a)-1):
                trainData.write(',')
        trainData.write("\n")

    trainData.close()


#randomFile()

def randomFileTest():
    test = open("test.csv","w")

    all_data = []

    for i in range(0,299):
        nume = "pedTest/"
        nume += "img_" + "00" + str(i+294)+".pgm"
    
        img = Image.open(nume)
        arr = numpy.array(img)  

        A = numpy.ravel(arr)
        A = numpy.append(A,1)

        all_data.append(A)

    for i in range(0,299):
        nume = "nonpedTest/"
        nume += "img_" + "0" + str(i+2001)+".pgm"
    
        img = Image.open(nume)
        arr = numpy.array(img)

        A = numpy.ravel(arr)
        A = numpy.append(A,0)

        all_data.append(A)

    random.shuffle(all_data)

    for a in all_data:
        for i in range(len(a)):
            test.write(str(a[i]))
            if i != (len(a)-1):
                test.write(',')
        test.write("\n")

    test.close()


randomFileTest()

trainFile = open("test.csv",'r')

trainList = trainFile.readlines()

all_values = trainList[0].split(',')

print(all_values[648])