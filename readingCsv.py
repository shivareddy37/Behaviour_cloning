import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2


left_imgs = []
right_imgs = []
center_imgs = []
steering_angles= []




def reading_csv():
    # loading data from diffrent csv's where _f means clockwise driving
    # _b means anticloclwise driving and t1 and t2 signifies track 1 and 2 respectively
    # _r means recovery lap

    with open('/home/shiva/data_p3/forward_t1.csv','r') as track1_f:
        csvreader = csv.reader(track1_f, delimiter=',')
        for row in csvreader:
            center_imgs.append(row[0])
            left_imgs.append(row[1])
            right_imgs.append(row[2])
            steering_angles.append(row[3])

    with open ('/home/shiva/data_p3/backward_t1.csv','r') as track1_b:
        csvreader = csv.reader(track1_b, delimiter=',')
        for row in csvreader:
            center_imgs.append(row[0])
            left_imgs.append(row[1])
            right_imgs.append(row[2])
            steering_angles.append(row[3])

    with open ('/home/shiva/data_p3/forward_t2.csv','r') as track2_f:
        csvreader = csv.reader(track2_f, delimiter=',')
        for row in csvreader:
            center_imgs.append(row[0])
            left_imgs.append(row[1])
            right_imgs.append(row[2])
            steering_angles.append(row[3])

    with open ('/home/shiva/data_p3/backward_t2.csv','r') as track2_b:
        csvreader = csv.reader(track2_b, delimiter=',')
        for row in csvreader:
            center_imgs.append(row[0])
            left_imgs.append(row[1])
            right_imgs.append(row[2])
            steering_angles.append(row[3])


    with open('/home/shiva/data_p3/recovery_t1.csv', 'r') as track1_r:
        csvreader = csv.reader(track1_r, delimiter=',')
        for row in csvreader:
            center_imgs.append(row[0])
            left_imgs.append(row[1])
            right_imgs.append(row[2])
            steering_angles.append(row[3])

    with open('/home/shiva/data_p3/recovery_t2.csv', 'r') as track2_r:
        csvreader = csv.reader(track2_r, delimiter=',')
        for row in csvreader:
            center_imgs.append(row[0])
            left_imgs.append(row[1])
            right_imgs.append(row[2])
            steering_angles.append(row[3])

def main():
    reading_csv()
    return left_imgs, right_imgs, center_imgs, steering_angles



if __name__ == '__main__':
    main()

