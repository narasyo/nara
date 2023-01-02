import numpy as np 
import plot
import math
import matplotlib.pyplot as plt

x = np.arange(0,1,0.01)
y1 = np.sin(x)
y2 = np.sin(2*x)
y3 = np.sin(3*x)
y4 = np.sin(4*x)

# y = [[y1,y2],[y3,y4]]
# row = 2
# column = 2
# figsize = (10,10)
# fontsize = 10
# labelsize = 10
# linewidth = 1.0
# color = [["red","blue"],["yellow","green"]]
# xlim = [[[0,1],[0,1]],[[0,1],[0,1]]]
# ylim = [[[-1,1],[-1,1]],[[-1,1],[-1,1]]]
# xlabel =[["x","x"],["x","x"]]
# ylabel =[["sinx","sin2x"],["sin3x","sin4x"]]
# title = [["y=sinx","y=sin2x"],["y=sin3x","y=sin4x"]]

# graph = plot.plot(x,y,row=row,column=column,figsize=figsize,
#                   fontsize=fontsize,labelsize=labelsize,linewidth=linewidth,
#                   color=color,xlim=xlim,ylim=ylim,
#                   params_length=1.0,xlabel=xlabel,ylabel=ylabel,title=title)
# graph.plot_graph_matrix()

# plt.show()

# y = [y1,y2,y3,y4]
# row = 1
# column = 4
# figsize = (10,10)
# fontsize = 10
# labelsize = 10
# linewidth = 1.0
# color = ["red","blue","yellow","green"]
# xlim = [[0,1],[0,1],[0,1],[0,1]]
# ylim = [[-1,1],[-1,1],[-1,1],[-1,1]]
# xlabel =["x","x","x","x"]
# ylabel =["sinx","sin2x","sin3x","sin4x"]
# title = ["y=sinx","y=sin2x","y=sin3x","y=sin4x"]

# graph = plot.plot(x,y,row=row,column=column,figsize=figsize,
#                   fontsize=fontsize,labelsize=labelsize,linewidth=linewidth,
#                   color=color,xlim=xlim,ylim=ylim,
#                   params_length=1.0,xlabel=xlabel,ylabel=ylabel,title=title)
# graph.plot_graph_column()

# plt.show()

y =y1
figsize = (10,10)
fontsize = 10
labelsize = 10
linewidth = 1.0
color = "red"
xlim = [0,1]
ylim = [-1,1]
xlabel = "x"
ylabel ="sinx"
title = "y=sinx"

graph = plot.plot(x,y,row="",column="",figsize=figsize,
                  fontsize=fontsize,labelsize=labelsize,linewidth=linewidth,
                  color=color,xlim=xlim,ylim=ylim,
                  params_length=1.0,xlabel=xlabel,ylabel=ylabel,title=title)
graph.plot_scatter()

plt.show()
