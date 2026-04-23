import os
l=os.listdir('/data1/wcx/fivek/fivek_png/gt')
l.sort()
for idx,i in enumerate(l):
    if idx<4500:
        with open('mit5k_train.txt','a') as f:
            f.write(i)
            f.write('\n')
    else:
        with open('mit5k_test.txt','a') as f:
            f.writelines(i)
            f.write('\n')