import numpy as np
import cv2
import readingCsv
import img_manipulator
import csv

left_imgs = []
right_imgs = []
center_imgs = []
steering_angles= []
def read_csv():
    i = 0
    with open('/home/shiva/data/driving_log.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')

        for row in csvreader:
            if i == 0:
                print(row)
            else:
                center_imgs.append(row[0])
                left_imgs.append(row[1])
                right_imgs.append(row[2])
                steering_angles.append(row[3])
            i+=1

def main():
    read_csv()
    for i in range (0,5):
         src = cv2.imread()
    return left_imgs, right_imgs, center_imgs, steering_angles



if __name__ == '__main__':
    main()