import csv
import shutil
import pandas as pd
import os

with open("label.csv", "r") as handler:
 df = csv.reader(handler, delimiter=',')
 for row in df:
   heights = [row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11]]
    
   max_height = max(heights)
   index = heights.index(max_height)
    
   f = row[0]
   l = index
   shutil.move(f'E:\FERPlus\data\FER2013Train/{f}', f'E:\FERPlus\data\FER2013Train/{l}/{f}')
    