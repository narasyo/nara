# -*- coding: utf-8 -*-
import math
import numpy as np
import os

def distance(lat1,lon1,lat2,lon2) :

    pole_radius = 6356752.314245                  # 極半径[m]
    equator_radius = 6378137.0                    # 赤道半径[m]

    # 緯度経度をラジアンに変換
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    lat_difference = lat1_rad - lat2_rad       # 緯度差
    lon_difference = lon1_rad - lon2_rad       # 経度差
    lat_average = (lat1_rad + lat2_rad) / 2    # 平均緯度

    e2 = (math.pow(equator_radius, 2) - math.pow(pole_radius, 2)) \
            / math.pow(equator_radius, 2)  # 第一離心率^2

    w = math.sqrt(1- e2 * math.pow(math.sin(lat_average), 2))

    m = equator_radius * (1 - e2) / math.pow(w, 3) # 子午線曲率半径 

    n = equator_radius / w                         # 卯酉線曲半径

    distance = math.sqrt(math.pow(m * lat_difference, 2) \
                   + math.pow(n * lon_difference * math.cos(lat_average), 2)) # 距離計測

    return(distance / 1000)

os.chdir("./data/furukawa_data")
file_input = "lantitude_longitude_20130804.txt"
f_number,lantitude,longitude = np.loadtxt(file_input,dtype='str',unpack=True)
lantitude1 = [float(i) for i in lantitude]
longitude1 = [float(i) for i in longitude]
number = np.arange(1,29,1)

for n,f,lat,lon in zip(number,f_number,lantitude1,longitude1) :
    lantitude2 = lantitude1[n:]
    longitude2 = longitude1[n:]
    f_number2 = f_number[n:]
    for f2,lat2,lon2 in zip(f_number2,lantitude2,longitude2) :
        d = distance(lat,lon,lat2,lon2)
        f = str(f)
        f2 = str(f2)
        d = str(d)
        with open('distance_20130804.txt', 'a', encoding='utf-8', newline='\n') as file_output:
            file_output.write(f+' '+f2+' '+d+'\n')



