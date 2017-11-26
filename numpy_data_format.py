import numpy as np
import pandas

data = np.loadtxt('model backup/trial 3/episodesAndRewards_1.txt')
data2 = np.loadtxt('model backup/trial 3/episodesAndRewards_2.txt')
data3 = np.loadtxt('model backup/trial 3/episodesAndRewards_3.txt')
#data4 = np.loadtxt('model backup/episodesAndRewards_4.txt')
#data5 = np.loadtxt('model backup/episodesAndRewards_5.txt')


test = data2.item((0,1))

frame1 = pandas.DataFrame(data)
frame2 = pandas.DataFrame(data2)
frame3 = pandas.DataFrame(data3)
#frame4 = pandas.DataFrame(data4)
#frame5 = pandas.DataFrame(data5)

frames = [frame1, frame2, frame3]#, frame4, frame5]

hours_36 = pandas.concat(frames)
writer = pandas.ExcelWriter('model backup/trial 3/36_hour_test.xlsx')
hours_36.to_excel(writer, 'trial 3')
writer.save()

#print(hours_36)
