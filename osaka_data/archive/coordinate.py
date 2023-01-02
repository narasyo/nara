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

def coordinate(lat0,lon0,lat1,lon1) :

    lat_dis = distance(lat0,0,lat1,0)
    lon_dis = distance(0,lon0,0,lon1)

    lat_dif = lat1 - lat0
    lon_dif = lon1 - lon0

    if lat_dif >= 0 :
        y = lat_dis
    elif lat_dif < 0 :
        y = -lat_dis

    if lon_dif >= 0 :
        x = lon_dis
    elif lon_dif < 0 :
        x = -lon_dis

    return x,y

OSA = {"lat":34.6985,"lon":135.4758}
OSB = {"lat":34.6934,"lon":135.4832}
OSC = {"lat":34.6943,"lon":135.4622}
FKS = {"lat":34.69,"lon":135.473}
O = {"lat":34.6934,"lon":135.473}

OSA_x,OSA_y = coordinate(O["lat"],O["lon"],OSA["lat"],OSA["lon"])
OSB_x,OSB_y = coordinate(O["lat"],O["lon"],OSB["lat"],OSB["lon"])
OSC_x,OSC_y = coordinate(O["lat"],O["lon"],OSC["lat"],OSC["lon"])
FKS_x,FKS_y = coordinate(O["lat"],O["lon"],FKS["lat"],FKS["lon"])

coo_dict = {"OSA_x":OSA_x,"OSA_y":OSA_y,"OSB_x":OSB_x,"OSB_y":OSB_y,"OSC_x":OSC_x,"OSC_y":OSC_y,"FKS_x":FKS_x,"FKS_y":FKS_y}

os.chdir("./osaka_data/archive")
np.save("coordinate.npy",coo_dict)


