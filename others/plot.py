import matplotlib.pyplot as plt

#################################
##          plot    class      ##
#################################
class plot:

    def __init__(self,x,y,row,column,
                figsize,fontsize,labelsize,
                linewidth,color,xlim,ylim,
                params_length,xlabel,ylabel,title,name):
        self.x = x
        self.y = y
        self.row = row
        self.column = column
        self.figsize = figsize
        self.fontsize = fontsize
        self.labelsize = labelsize
        self.linewidth = linewidth
        self.color = color
        self.xlim = xlim
        self.ylim = ylim
        self.params_length = params_length  
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.name = name

#-----------------------------------------------#

    def unify_matrix(x,row,column) :

        x_row = [x for i in range(row)]
        x_column = [x_row for i in range(column)]
        return x_column
        
    def unify_vector(x,number) :

        x_vector = [x for i in range(number)]
        return x_vector
           
        
#######################
# figのデフォルト設定 #
#######################

    figure_facecolor = "white"
    figure_edgecolor = "black"
    figure_linewidth = 0

########################
# axesのデフォルト設定 #
########################

    axes_facecolor = "whitesmoke"
    grid = True
    marker = "."
    marker_size = 500
    marker_color = "red"

#-----------------------------------------------#


    def plot_graph_matrix(self,unify) :

        if unify == True : 
            x = plot.unify_matrix(self.x,self.row,self.column)
            fontsize = plot.unify_matrix(self.fontsize,self.row,self.column)
            labelsize = plot.unify_matrix(self.labelsize,self.row,self.column)
            linewidth = plot.unify_matrix(self.linewidth,self.row,self.column)
            color = plot.unify_matrix(self.color,self.row,self.column)
            xlim = plot.unify_matrix(self.xlim,self.row,self.column)
            ylim = plot.unify_matrix(self.ylim,self.row,self.column)
            params_length = plot.unify_matrix(self.params_length,self.row,self.column)
            xlabel = plot.unify_matrix(self.xlabel,self.row,self.column)
            ylabel = plot.unify_matrix(self.ylabel,self.row,self.column)

            fig, axes = plt.subplots(self.row,self.column,
                                        figsize=self.figsize,facecolor=plot.figure_facecolor,
                                        linewidth=plot.figure_linewidth,edgecolor=plot.figure_edgecolor,
                                        subplot_kw=dict(facecolor=plot.axes_facecolor))
            for i in range(self.row) :
                for j in range(self.column) :
                    axes[i][j].plot(x[i][j],self.y[i][j],color=color[i][j],linewidth=linewidth[i][j])
                    axes[i][j].set_xlim(xlim[i][j])
                    axes[i][j].set_ylim(ylim[i][j])
                    axes[i][j].set_xlabel(f"{xlabel[i][j]}",fontsize=fontsize[i][j])
                    axes[i][j].set_ylabel(f"{ylabel[i][j]}",fontsize=fontsize[i][j])
                    axes[i][j].set_title(f"{self.title[i][j]}",fontsize=fontsize[i][j])
                    axes[i][j].tick_params(direction="inout",labelsize=labelsize[i][j],length=params_length[i][j])
                    axes[i][j].grid(plot.grid)
                    
            fig.tight_layout()
            fig.savefig(f"{self.name}.png")
            fig.savefig(f"{self.name}.eps")

        elif unify == False : 
            x = plot.unify_matrix(self.x,self.row,self.column)

            fig, axes = plt.subplots(self.row,self.column,
                                        figsize=self.figsize,facecolor=plot.figure_facecolor,
                                        linewidth=plot.figure_linewidth,edgecolor=plot.figure_edgecolor,
                                        subplot_kw=dict(facecolor=plot.axes_facecolor))
            for i in range(self.row) :
                for j in range(self.column) :
                    axes[i][j].plot(x[i][j],self.y[i][j],color=self.color[i][j],linewidth=self.linewidth)
                    axes[i][j].set_xlim(self.xlim[i][j])
                    axes[i][j].set_ylim(self.ylim[i][j])
                    axes[i][j].set_xlabel(f"{self.xlabel[i][j]}",fontsize=self.fontsize)
                    axes[i][j].set_ylabel(f"{self.ylabel[i][j]}",fontsize=self.fontsize)
                    axes[i][j].set_title(f"{self.title[i][j]}",fontsize=self.fontsize)
                    axes[i][j].tick_params(direction="inout",labelsize=self.fontsize,length=self.params_length)
                    axes[i][j].grid(plot.grid)
                    
            fig.tight_layout()
            fig.savefig(f"{self.name}.png")
            fig.savefig(f"{self.name}.eps")

    def plot_graph_row(self,unify) :

        if unify == True : 

            x = plot.unify_vector(self.x,self.row)
            fontsize = plot.unify_vector(self.fontsize,self.row)
            labelsize = plot.unify_vector(self.labelsize,self.row)
            linewidth = plot.unify_vector(self.linewidth,self.row)
            color = plot.unify_vector(self.color,self.row)
            xlim = plot.unify_vector(self.xlim,self.row)
            ylim = plot.unify_vector(self.ylim,self.row)
            params_length = plot.unify_vector(self.params_length,self.row)
            xlabel = plot.unify_vector(self.xlabel,self.row)
            ylabel = plot.unify_vector(self.ylabel,self.row)

            fig, axes = plt.subplots(self.row,1,
                                        figsize=self.figsize,facecolor=plot.figure_facecolor,
                                        linewidth=plot.figure_linewidth,edgecolor=plot.figure_edgecolor,
                                        subplot_kw=dict(facecolor=plot.axes_facecolor))
            for i in range(self.row) :
                axes[i].plot(x[i],self.y[i],color=color[i],linewidth=linewidth[i])
                axes[i].set_xlim(xlim[i])
                axes[i].set_ylim(ylim[i])
                axes[i].set_xlabel(f"{xlabel[i]}",fontsize=fontsize[i])
                axes[i].set_ylabel(f"{ylabel[i]}",fontsize=fontsize[i])
                axes[i].set_title(f"{self.title[i]}",fontsize=fontsize[i])
                axes[i].tick_params(direction="inout",labelsize=labelsize[i],length=params_length[i])
                axes[i].grid(plot.grid)
                    
            fig.tight_layout()
            fig.savefig(f"{self.name}.png")
            fig.savefig(f"{self.name}.eps")

        if unify == False :

            x = plot.unify_vector(self.x,self.row)
            fig, axes = plt.subplots(self.row,1,
                                    figsize=self.figsize,facecolor=plot.figure_facecolor,
                                    linewidth=plot.figure_linewidth,edgecolor=plot.figure_edgecolor,
                                    subplot_kw=dict(facecolor=plot.axes_facecolor))
            for i in range(self.row) :
                axes[i].plot(x[i],self.y[i],color=self.color[i],linewidth=self.linewidth)
                axes[i].set_xlim(self.xlim[i])
                axes[i].set_ylim(self.ylim[i])
                axes[i].set_xlabel(f"{self.xlabel[i]}",fontsize=self.fontsize)
                axes[i].set_ylabel(f"{self.ylabel[i]}",fontsize=self.fontsize)
                axes[i].set_title(f"{self.title[i]}",fontsize=self.fontsize)
                axes[i].tick_params(direction="inout",labelsize=self.fontsize,length=self.params_length)
                axes[i].grid(plot.grid)
                    
            fig.tight_layout()
            fig.savefig(f"{self.name}.png")
            fig.savefig(f"{self.name}.eps")

    def plot_graph_column(self) :

        x = plot.unify_vector(self.x,self.column)
        fig, axes = plt.subplots(1,self.column,
                                 figsize=self.figsize,facecolor=plot.figure_facecolor,
                                 linewidth=plot.figure_linewidth,edgecolor=plot.figure_edgecolor,
                                 subplot_kw=dict(facecolor=plot.axes_facecolor))
        for i in range(self.column) :
            axes[i].plot(x[i],self.y[i],color=self.color[i],linewidth=self.linewidth)
            axes[i].set_xlim(self.xlim[i])
            axes[i].set_ylim(self.ylim[i])
            axes[i].set_xlabel(f"{self.xlabel[i]}",fontsize=self.fontsize)
            axes[i].set_ylabel(f"{self.ylabel[i]}",fontsize=self.fontsize)
            axes[i].set_title(f"{self.title[i]}",fontsize=self.fontsize)
            axes[i].tick_params(direction="inout",labelsize=self.fontsize,length=self.params_length)
            axes[i].grid(plot.grid)
                
        fig.tight_layout()
        fig.savefig(f"{self.name}.png")
        fig.savefig(f"{self.name}.eps")

    def plot_graph(self) :

        fig, axes = plt.subplots(figsize=self.figsize,facecolor=plot.figure_facecolor,
                                 linewidth=plot.figure_linewidth,edgecolor=plot.figure_edgecolor,
                                 subplot_kw=dict(facecolor=plot.axes_facecolor))
        axes.plot(self.x,self.y1,color=self.color,linewidth=self.linewidth)
        axes.set_xlim(self.xlim)
        axes.set_ylim(self.ylim)
        axes.set_xlabel(f"{self.xlabel}",fontsize=self.fontsize)
        axes.set_ylabel(f"{self.ylabel}",fontsize=self.fontsize)
        axes.set_title(f"{self.title}",fontsize=self.fontsize)
        axes.tick_params(direction="inout",labelsize=self.fontsize,length=self.params_length)
        axes.grid(plot.grid)

        fig.tight_layout()
        fig.savefig(f"{self.name}.png")
        fig.savefig(f"{self.name}.eps")


    def plot_scatter(self) :

        fig, axes = plt.subplots(figsize=self.figsize,facecolor=plot.figure_facecolor,
                                 linewidth=plot.figure_linewidth,edgecolor=plot.figure_edgecolor,
                                 subplot_kw=dict(facecolor=plot.axes_facecolor))
        axes.scatter(self.x,self.y,c=plot.marker_color,s=plot.marker_size,marker=plot.marker)
        axes.set_xlim(self.xlim)
        axes.set_ylim(self.ylim)
        axes.set_xlabel(f"{self.xlabel}",fontsize=self.fontsize)
        axes.set_ylabel(f"{self.ylabel}",fontsize=self.fontsize)
        axes.set_title(f"{self.title}",fontsize=self.fontsize)
        axes.tick_params(direction="inout",labelsize=self.fontsize,length=self.params_length)
        axes.grid(plot.grid)

        fig.tight_layout()
        fig.savefig(f"{self.name}.png")
        fig.savefig(f"{self.name}.eps")