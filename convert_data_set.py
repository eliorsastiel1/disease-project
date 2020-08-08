import csv
import pandas as pd
text_file = open("symptoms.txt", "r")
lines = text_file.readlines()[0].split(',')
lines.insert(0,'disease')
print(lines)

empty_arr = [0]*len(lines)

with open('converted_data.csv', mode='w') as converted_data:
    writer = csv.writer(converted_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(lines)
    original_data_set=pd.read_csv("original_data_set.csv")
    for index, row in original_data_set.iterrows():
        arr = empty_arr[:]
        for i,r in enumerate(row):
            if type(r) is str:
                if(i==0):
                    arr[0]=r
                else:
                    index=lines.index(r.strip())
                    arr[index]=1
            if(i==17):
                writer.writerow(arr)

