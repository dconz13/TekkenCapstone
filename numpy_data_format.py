import numpy as np
import pandas

data = np.loadtxt('model backup/episodesAndRewards_1.txt')
data2 = np.loadtxt('model backup/episodesAndRewards_2.txt')
data3 = np.loadtxt('model backup/episodesAndRewards_3.txt')


test = data2.item((0,1))

frame1 = pandas.DataFrame(data)
frame2 = pandas.DataFrame(data2)
frame3 = pandas.DataFrame(data3)

frames = [frame1, frame2, frame3]

hours_36 = pandas.concat(frames)
writer = pandas.ExcelWriter('36_hour_test.xlsx')
hours_36.to_excel(writer, 'Sheet1')
writer.save()

#print(hours_36)
