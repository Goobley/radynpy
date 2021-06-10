import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np 

def rmovie_basicvar(cdf, 
                      var = 'tg1',
                      Mm = False,
                      km = False,
                      savefig = False,
                      figname = 'radynvar.html',
                      color = 'steelblue'):

    '''
    A function to produce an animated figure of RADYN variables.
    This version is pre-constructed and lets you just input the 
    variable you want to plot. Other variables (such as populations)
    will require more input, and are separate functions.

    Turns the output into a pandas dataframe, which is then passed to 
    plotly express to create the animated figure

    Parameters
    __________
    cdf : The radyn cdf object
    var : str
          The variable to plot (default = 'tg1')
    Mm :  Boolean
          Plot height in Mm (default = False)
    km :  Boolean
          Plot height in km (default = False)
    savefig : Boolean
              Save the figure (html file)
    figname : str
              Filename, if saving the output


    NOTES :
             So far, allowed variables are 
                 tg1 - temperature
                 ne1 - electron density
                 bheat1 - beam heating rate
                 d1 - mass density
                 vz1 - velocity
                 np - proton density

    Graham Kerr, March 2021
    '''

    ########################################################################
    # Some preliminary set up
    ########################################################################

    if Mm == True:
        xtitle = 'Height [Mm]'
        height = cdf.z1/1e8
    elif km == True:
        xtitle = 'Height [km]'
        height = cdf.z1/1e5
    else:
        xtitle = 'Height [cm]'
        height = cdf.z1

    if var == 'tg1':
        rvar = cdf.tg1
        ytitle = 'Temperature [K]'
        ylog = True
        xlog = False
    elif var == 'ne1':
        rvar = cdf.ne1
        ytitle = 'Electron Density [cm<sup>-3</sup>]'
        ylog = True
        xlog = False
    elif var == 'bheat1':
        rvar = cdf.bheat1
        ytitle = 'Q<sub>beam</sub> [erg cm<sup>-3</sup> s<sup>-1</sup>]'
        ylog = False
        xlog = False
    elif var == 'd1':
        rvar = cdf.d1
        ytitle = 'Mass Density [g cm<sup>-3</sup>]'
        ylog = True
        xlog = False
    elif var == 'vz1':
        rvar = cdf.vz1/1e5
        ytitle = 'Velocity [km s<sup>-1</sup>]'
        ylog = False
        xlog = False
    elif var == 'np':
        rvar = cdf.n1[:,:,5,0]
        ytitle = 'Proton Density [cm<sup>-3</sup>]'
        ylog = True
        xlog = False

   

    template = dict(
		    layout = go.Layout(font = dict(family = "Rockwell", size = 16),
		                       title_font = dict(family = "Rockwell", size = 20), 
		                       plot_bgcolor = 'white',
		                       paper_bgcolor = 'white',
		                       xaxis = dict(
		                       	    showexponent = 'all',
					                exponentformat = 'e',
		                            tickangle = 0,
		                            linewidth = 3,
		                            showgrid = True,
		                        ),
		                        yaxis = dict(
                              showexponent = 'all',
					          exponentformat = 'e',
		                            linewidth = 3,
		                            showgrid = True,
		                            anchor = 'free',
		                            position = 0,
		                            domain = [0.0,1]
		                        ),
		                        coloraxis_colorbar = dict(
		                            thickness = 15,
		                            tickformat = '0.2f',
		                            ticks = 'outside',
		                            titleside = 'right'
		                        )
		                        ))

    ########################################################################
    #  Build the dataframe
    ########################################################################

    col1 = ytitle
    col2 = xtitle
    time = 'Time [s]'
    timeind = 'Time index'
    df_list = []
    for i in range(len(cdf.time)):
        data = {col1:rvar[i,:],
                col2:height[i,:],
                time: cdf.time[i],
                timeind: i
                }
        df_list.append(pd.DataFrame(data))
    
        df = pd.concat(df_list)

    ########################################################################
    #  Plot the variable
    ########################################################################


    h1 = 700
    w1 = 700

    fig1 = px.line(df,
               x = df.columns[1], y = df.columns[0],
#                animation_group = 'Time [s]',
               animation_frame = 'Time [s]',
               log_x = xlog,
               log_y = ylog,
               template = template,
               color_discrete_sequence = [color])

    fig1.show()

    if savefig == True:
        fig1.write_html(figname)


    return df




def rmovie(var1, var2, 
                      time = [-10.0],
                      savefig = False,
                      figname = 'radynvar.html',
                      xtitle = 'Var 1', 
                      ytitle = 'Var 2',
                      title = ' ',
                      color = 'steelblue',
                      xlog = False, ylog = False):

    '''
    A function to produce an animated figure of RADYN variables.
    This version is 'dumb' and just plots col1 vs col2 without any 
    axes labels, unless passed through the function fall.

    Variables must be input as [time, dim1]

    Turns the output into a pandas dataframe, which is then passed to 
    plotly express to create the animated figure

    Parameters
    __________
    var1 : float
           The variable to plot on the x-axis [time, dim1]
    var2 : float
           The variable to plot on the y-axis [time, dim1]
    xtitle : str
             The xaxis label (default "Var 1")
    ytitle : str
             The xaxis label (default "Var 2")
    title : str
             A plot title (default " ")
    savefig : Boolean
              Save the figure (html file)
    figname : str
              Filename, if saving the output
    xlog : boolean
           Default is false. Set to True to have log x-axis
    ylog : boolean
           Default is false. Set to True to have log y-axis

    NOTES :
            

    Graham Kerr, March 2021
    '''

    ########################################################################
    # Some preliminary set up
    ########################################################################

    if time[0] == -10:
        time = np.arange(0,var1.shape[0])
        col3 = 'Time [index]'
    else:
    	col3 = 'Time [s]'
    
    template = dict(
		    layout = go.Layout(font = dict(family = "Rockwell", size = 16),
		                       title_font = dict(family = "Rockwell", size = 20), 
		                       plot_bgcolor = 'white',
		                       paper_bgcolor = 'white',
		                       xaxis = dict(
		                       	    showexponent = 'all',
					                exponentformat = 'e',
		                            tickangle = 0,
		                            linewidth = 3,
		                            showgrid = True,
		                        ),
		                        yaxis = dict(
                              showexponent = 'all',
					          exponentformat = 'e',
		                            linewidth = 3,
		                            showgrid = True,
		                            anchor = 'free',
		                            position = 0,
		                            domain = [0.0,1]
		                        ),
		                        coloraxis_colorbar = dict(
		                            thickness = 15,
		                            tickformat = '0.2f',
		                            ticks = 'outside',
		                            titleside = 'right'
		                        )
		                        ))

    ########################################################################
    #  Build the dataframe
    ########################################################################

    col1 = xtitle
    col2 = ytitle
    df_list = []
    for i in range(len(time)):
        data = {col1:var1[i,:],
                col2:var2[i,:],
                col3: time[i],
                }
        df_list.append(pd.DataFrame(data))
    
        df = pd.concat(df_list)


    ########################################################################
    #  Plot the variable
    ########################################################################


    h1 = 700
    w1 = 700

    fig1 = px.line(df,
               x = df.columns[0], y = df.columns[1],
#                animation_group = 'Time [s]',
               animation_frame = df.columns[2],
               log_x = xlog,
               log_y = ylog,
               title = title,
               color_discrete_sequence = [color],
               template = template)

    fig1.show()

    if savefig == True:
        fig1.write_html(figname)


    return df