
import warnings 
warnings.filterwarnings('ignore')


import os
import numpy as np
import pandas as pd
import re
import string
from collections import Counter
import time


import missingno as msno
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
from pywaffle import Waffle






'''load the dataset'''
dat = pd.read_csv("healthcare-dataset-stroke-data.csv")




'''Plot the percentage graph'''
x = pd.DataFrame(dat.groupby(['stroke'])['stroke'].count())


#plot the subplot and add legends to the plot
fig, ax = plt.subplots(figsize = (6,6), dpi = 70)
ax.barh([1], x.stroke[1], height = 0.7, color = '#343bfe')
plt.text(-1150,-0.08, 'Healthy',{'font': 'Serif','weight':'bold','Size': '12','style':'normal', 'color':'#e6a129'})
plt.text(5000,-0.08, '95%',{'font':'Serif','weight':'bold' ,'size':'16','color':'#e6a129'})
ax.barh([0], x.stroke[0], height = 0.7, color = '#e6a129')
plt.text(-1000,1, 'Stroke', {'font': 'Serif','weight':'bold','Size': '12','style':'normal', 'color':'#343bfe'})
plt.text(300,1, '5%',{'font':'Serif', 'weight':'bold','size':'16','color':'#343bfe'})

#fill out the graphs with the following chosed color
fig.patch.set_facecolor('#f6f5f5')
ax.set_facecolor('#f6f5f5')


#Add legend to the bar plot
plt.text(-1150,1.77, 'Percentage of People Having Strokes and without strokes' ,{'font': 'Serif', 'Size': '18','weight':'bold', 'color':'black'})

plt.text(4650,0.8, 'Stroke ', {'font': 'Serif','weight':'bold','Size': '12','weight':'bold','style':'normal', 'color':'#343bfe'})

plt.text(5650,0.8, '|', {'color':'black' , 'size':'12', 'weight': 'bold'})

plt.text(5750,0.8, 'Healthy', {'font': 'Serif','weight':'bold', 'Size': '12','style':'normal', 'weight':'bold','color':'#e6a129'})

plt.text(-1150,1.5, 'We can see that it is a significantly unbalanced distribution,\nand clearly we see that 5 percent of people are likely to get \nheart strokes.', 
        {'font':'Serif', 'size':'12.5','color': 'black'})


#Use the plt function to set x-axis and y-axis and save the image. 
plt.tight_layout()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig("Percentage plot")








#Create numerical variables

dat['bmi_cat'] = pd.cut(dat['bmi'], bins = [0, 19, 25,30,10000], labels = ['Underweight', 'Ideal', 'Overweight', 'Obesity'])
dat['age_cat'] = pd.cut(dat['age'], bins = [0,13,18, 45,60,200], labels = ['Children', 'Teens', 'Adults','Mid Adults','Elderly'])
dat['glucose_cat'] = pd.cut(dat['avg_glucose_level'], bins = [0,90,160,230,500], labels = ['Low', 'Normal', 'High', 'Very High'])





'''plot Heart stroke and age graph'''
fig = plt.figure(figsize = (24,10), dpi = 60)

gt = fig.add_gridspec(10,24)
gt.update(wspace = 1, hspace = 0.05)


ax1 = fig.add_subplot(gt[1:10,13:]) 
ax2 = fig.add_subplot(gt[1:4,0:8]) 
ax3 = fig.add_subplot(gt[6:9, 0:8]) 


# set up axes list
axes = [ ax1,ax2, ax3]


# setting of axes
for ax in axes:
    ax.axes.get_yaxis().set_visible(False)
    ax.set_facecolor('#f6f5f5')
    
    for loc in ['left', 'right', 'top', 'bottom']:
        ax.spines[loc].set_visible(False)

fig.patch.set_facecolor('#f6f5f5')
        
ax1.axes.get_xaxis().set_visible(False)
ax1.axes.get_yaxis().set_visible(True)



stroke_age = dat[dat['stroke'] == 1].age_cat.value_counts()
healthy_age = dat[dat['stroke'] == 0].age_cat.value_counts()

ax1.hlines(y = ['Children', 'Teens', 'Adults', 'Mid Adults', 'Elderly'], xmin = [644,270,1691,1129,1127], 
          xmax = [1,1,11,59,177], color = 'grey',**{'linewidth':0.5})


sns.scatterplot(y = stroke_age.index, x = stroke_age.values, s = stroke_age.values*2, color = '#343bfe', ax= ax1, alpha = 1)
sns.scatterplot(y = healthy_age.index, x = healthy_age.values, s = healthy_age.values*2, color = '#e6a129', ax= ax1, alpha = 1)

ax1.axes.get_xaxis().set_visible(False)
ax1.set_xlim(xmin = -500, xmax = 2250)
ax1.set_ylim(ymin = -1,ymax = 5)

ax1.set_yticklabels( labels = ['Children', 'Teens', 'Adults', 'Mid Adults', 'Elderly'],fontdict = {'font':'Serif', 'fontsize':16,'fontweight':'bold', 'color':'black'})

ax1.text(-950,5.8, 'How Age Impact on Heart Strokes' ,{'font': 'Serif', 'Size': '25','weight':'bold', 'color':'black'},alpha = 0.9)
ax1.text(1000,4.8, 'Stroke ', {'font': 'Serif','weight':'bold','Size': '16','weight':'bold','style':'normal', 'color':'#343bfe'})
ax1.text(1300,4.8, '|', {'color':'black' , 'size':'16', 'weight': 'bold'})
ax1.text(1350,4.8, 'Healthy', {'font': 'Serif','weight':'bold', 'Size': '16','style':'normal', 'weight':'bold','color':'#e6a129'})
ax1.text(-950,5., 'Age have significant association with stokes, older people have larger probability of getting strokes \nmid age adults are the second higest', 
        {'font':'Serif', 'size':'16','color': 'black'})

ax1.text(stroke_age.values[0] + 30,4.05, stroke_age.values[0], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#343bfe'})
ax1.text(healthy_age.values[2] - 300,4.05, healthy_age.values[2], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#e6a129'})

ax1.text(stroke_age.values[1] + 30,3.05, stroke_age.values[1], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#343bfe'})
ax1.text(healthy_age.values[1] - 300,3.05, healthy_age.values[1], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#e6a129'})


# plot the distribution plots 

sns.kdeplot(data = dat, x = 'age', ax = ax2, shade = True, color = '#c76a48', alpha = 1, )
ax2.set_xlabel('Age of a person', fontdict = {'font':'Serif', 'color': 'black', 'size': 16,'weight':'bold' })
ax2.text(-17,0.025,'Age Distribution', {'font':'Serif', 'color': 'black','weight':'bold','size':24}, alpha = 0.9)
ax2.text(-17,0.021, 'From this graph we have adult population is the median group.', 
        {'font':'Serif', 'size':'16','color': 'black'})
ax2.text(80,0.019, 'Total',{'font':'Serif', 'size':'14','color': '#c76a48','weight':'bold'})
ax2.text(92,0.019, '=',{'font':'Serif', 'size':'14','color': 'black','weight':'bold'})
ax2.text(97,0.019, 'Stroke',{'font':'Serif', 'size':'14','color': '#343bfe','weight':'bold'})
ax2.text(113,0.019, '+',{'font':'Serif', 'size':'14','color': 'black','weight':'bold'})
ax2.text(117,0.019, 'Healthy',{'font':'Serif', 'size':'14','color': '#e6a129','weight':'bold'})


# plot the distribution plots and add legend and comments to the graph


sns.kdeplot(data = dat[dat['stroke'] == 0], x = 'age',ax = ax3, shade = True,  alpha = 1, color = '#e6a129' )
sns.kdeplot(data = dat[dat['stroke'] == 1], x = 'age',ax = ax3, shade = True,  alpha = 0.8, color = '#343bfe')

ax3.set_xlabel('Age of a person', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})

ax3.text(-17,0.0525,'Stroke-Age Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':24}, alpha= 0.9)
ax3.text(-17,0.043,'From the Distribution plot it is clear that old people are \nhaving larger number of strokes than young people.', {'font':'Serif', 'color': 'black', 'size':14})
ax3.text(100,0.043, 'Stroke ', {'font': 'Serif','weight':'bold','Size': '16','weight':'bold','style':'normal', 'color':'#343bfe'})
ax3.text(117,0.043, '|', {'color':'black' , 'size':'16', 'weight': 'bold'})
ax3.text(120,0.043, 'Healthy', {'font': 'Serif','weight':'bold', 'Size': '16','style':'normal', 'weight':'bold','color':'#e6a129'})

fig.text(0.25,0.05,'Relationship between Heart Strokes and Age',{'font':'Serif', 'weight':'bold','color': 'black', 'size':30})
plt.tight_layout()
plt.savefig("Heart stroke and age")









'''Plot the Heart stroke and glucose graph'''
fig = plt.figure(figsize = (24,10), dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)


ax2 = fig.add_subplot(gs[0:3,0:10]) 
ax3 = fig.add_subplot(gs[5:10, 0:10]) 
ax1 = fig.add_subplot(gs[0:,13:]) 

# setting up axes list
axes = [ ax1,ax2, ax3]


# setting of axes
for ax in axes:
    ax.axes.get_yaxis().set_visible(False)
    ax.set_facecolor('#f6f5f5')
    
    for loc in ['left', 'right', 'top', 'bottom']:
        ax.spines[loc].set_visible(False)

fig.patch.set_facecolor('#f6f5f5')
        
ax1.axes.get_xaxis().set_visible(False)
ax1.axes.get_yaxis().set_visible(True)


#plot of stoke and healthy people

stroke_glu = dat[dat['stroke'] == 1].glucose_cat.value_counts()
healthy_glu = dat[dat['stroke'] == 0].glucose_cat.value_counts()

ax1.hlines(y = ['Low', 'Normal', 'High', 'Very High'], xmin = [2316,1966,478,101], 
          xmax = [89,71,71,18], color = 'grey',**{'linewidth':0.5})


sns.scatterplot(y = stroke_glu.index, x = stroke_glu.values, s = stroke_glu.values, color = '#343bfe', ax= ax1, alpha = 1)
sns.scatterplot(y = healthy_glu.index, x = healthy_glu.values, s = healthy_glu.values, color = '#e6a129', ax= ax1, alpha = 1)

ax1.axes.get_xaxis().set_visible(False)
ax1.set_xlim(xmin = -500, xmax = 3000)
ax1.set_ylim(ymin = -1.5,ymax = 4.5)

ax1.set_yticklabels( labels = ['Low', 'Normal', 'High', 'Very High'],fontdict = {'font':'Serif', 'fontsize':16,'fontweight':'bold', 'color':'black'})

ax1.text(-1000,4.3, 'How Glucose level Impact on Heart Strokes' ,{'font': 'Serif', 'Size': '25','weight':'bold', 'color':'black'})
ax1.text(1700,3.5, 'Stroke ', {'font': 'Serif','weight':'bold','Size': '16','weight':'bold','style':'normal', 'color':'#343bfe'})
ax1.text(2050,3.5, '|', {'color':'black' , 'size':'16', 'weight': 'bold'})
ax1.text(2075,3.5, 'Healthy', {'font': 'Serif','weight':'bold', 'Size': '16','style':'normal', 'weight':'bold','color':'#e6a129'})
ax1.text(-1000,3.8, 'Glucose level does not have significant association with strokes.', 
        {'font':'Serif', 'size':'16','color': 'black'})


ax1.text(stroke_glu.values[0] + 30,0.05, stroke_glu.values[0], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#343bfe'})
ax1.text(healthy_glu.values[0] + -355,0.05, healthy_glu.values[0], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#e6a129'})

ax1.text(stroke_glu.values[2] + 30,1.05, stroke_glu.values[2], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#343bfe'})
ax1.text(healthy_glu.values[2] + 1170,1.05, healthy_glu.values[2], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#e6a129'})

ax1.text(stroke_glu.values[1] + 30,2.05, stroke_glu.values[1], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#343bfe'})
ax1.text(healthy_glu.values[1] - 1450,2.05, healthy_glu.values[1], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#e6a129'})



# plotting distribution plots

sns.kdeplot(data = dat, x = 'avg_glucose_level', ax = ax2, shade = True, color = '#c76a48', alpha = 1, )
ax2.set_xlabel('Average Glucose Level', fontdict = {'font':'Serif', 'color': 'black', 'size': 16,'weight':'bold' })
ax2.text(25,0.023,'Glucose Distribution', {'font':'Serif', 'color': 'black','weight':'bold','size':20})
ax2.text(25,0.019, 'From the distribution plot, we see most people have similar glocose level.', 
        {'font':'Serif', 'size':'16','color': 'black'})
ax2.text(210,0.017, 'Total',{'font':'Serif', 'size':'14','color': '#c76a48','weight':'bold'})
ax2.text(240,0.017, '=',{'font':'Serif', 'size':'14','color': 'black','weight':'bold'})
ax2.text(250,0.017, 'Stroke',{'font':'Serif', 'size':'14','color': '#343bfe','weight':'bold'})
ax2.text(280,0.017, '+',{'font':'Serif', 'size':'14','color': 'black','weight':'bold'})
ax2.text(290,0.017, 'Healthy',{'font':'Serif', 'size':'14','color': '#e6a129','weight':'bold'})


# distribution plots adding comments and legends


sns.kdeplot(data = dat[dat['stroke'] == 0], x = 'avg_glucose_level',ax = ax3, shade = True,  alpha = 1, color = '#e6a129' )
sns.kdeplot(data = dat[dat['stroke'] == 1], x = 'avg_glucose_level',ax = ax3, shade = True,  alpha = 0.8, color = '#343bfe')

ax3.set_xlabel('Average Glucose Level', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})

ax3.text(-17,0.0195,'Stroke-Glucose Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':20})
ax3.text(-17,0.0176,'It is hard to determine whether glucose level effect \npeople of having strokes.', {'font':'Serif', 'color': 'black', 'size':14})
ax3.text(240,0.0174, 'Stroke ', {'font': 'Serif','weight':'bold','Size': '16','weight':'bold','style':'normal', 'color':'#343bfe'})
ax3.text(290,0.0174, '|', {'color':'black' , 'size':'16', 'weight': 'bold'})
ax3.text(300,0.0174, 'Healthy', {'font': 'Serif','weight':'bold', 'Size': '16','style':'normal', 'weight':'bold','color':'#e6a129'})


fig.text(0.2,0.03,'Assocaition between Heart Strokes and Glucose',{'font':'Serif', 'weight':'bold','color': 'black', 'size':25})
plt.tight_layout()
plt.savefig("Heart stroke and glutose")






'''plot the Heart stoke and weight graph'''
fig = plt.figure(figsize = (24,10),dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)


ax2 = fig.add_subplot(gs[1:4,0:8]) 
ax3 = fig.add_subplot(gs[6:9, 0:8]) 
ax1 = fig.add_subplot(gs[2:9,13:]) 

# setting up axes list
axes = [ax1,ax2, ax3]


# setting of axes
for ax in axes:
    ax.axes.get_yaxis().set_visible(False)
    ax.set_facecolor('#f6f5f5')
    
    for loc in ['left', 'right', 'top', 'bottom']:
        ax.spines[loc].set_visible(False)

fig.patch.set_facecolor('#f6f5f5')
        
ax1.axes.get_xaxis().set_visible(False)
ax1.axes.get_yaxis().set_visible(True)
ax1.set_xlim(xmin = -250,xmax = 2000)
ax1.set_ylim(ymin = -1,ymax =3.5)


# plot of stoke people and healthy people

stroke_bmi = dat[dat['stroke'] == 1].bmi_cat.value_counts()
healthy_bmi = dat[dat['stroke'] == 0].bmi_cat.value_counts()

ax1.hlines(y = ['Obesity', 'Overweight', 'Ideal', 'Underweight'], xmin = [96,115,37,1], 
          xmax = [1797,1495,1159,410], color = 'grey',**{'linewidth':0.5})


sns.scatterplot(y = stroke_bmi.index, x = stroke_bmi.values, s = stroke_bmi.values*2, color = '#343bfe', ax= ax1, alpha = 1)
sns.scatterplot(y = healthy_bmi.index, x = healthy_bmi.values, s = healthy_bmi.values*2, color = '#e6a129', ax= ax1, alpha = 1)

ax1.set_yticklabels( labels = ['Obesity', 'Overweight', 'Ideal', 'Underweight'],fontdict = {'font':'Serif', 'fontsize':16,'fontweight':'bold', 'color':'black'})


ax1.text(-750,-1.5, 'How BMI Impact on Heart Strokes' ,{'font': 'Serif', 'Size': '25','weight':'bold', 'color':'black'})
ax1.text(1000,-1., 'Stroke ', {'font': 'Serif','weight':'bold','Size': '16','weight':'bold','style':'normal', 'color':'#343bfe'})
ax1.text(1250,-1, '|', {'color':'black' , 'size':'16', 'weight': 'bold'})
ax1.text(1300,-1, 'Healthy', {'font': 'Serif','weight':'bold', 'Size': '16','style':'normal', 'weight':'bold','color':'#e6a129'})
ax1.text(-750,-0.8, 'People with obesity and overweight people are more likely to getting heart strokes', 
        {'font':'Serif', 'size':'16','color': 'black'})



ax1.text(stroke_bmi.values[0] + 20 , 0.98, stroke_bmi.values[0], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#343bfe'})
ax1.text(healthy_bmi.values[1] - 275 ,0.98, healthy_bmi.values[1], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#e6a129'})

ax1.text(stroke_bmi.values[1] + 30,0, stroke_bmi.values[1], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#343bfe'})
ax1.text(healthy_bmi.values[0] - 300,0, healthy_bmi.values[0], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#e6a129'})


#plot distribution plots 

sns.kdeplot(data = dat, x = 'bmi', ax = ax2, shade = True, color = '#c76a48', alpha = 1, )
ax2.set_xlabel('Body mass index of a person', fontdict = {'font':'Serif', 'color': 'black', 'size': 16,'weight':'bold' })
ax2.text(-17,0.085,'BMI Distribution', {'font':'Serif', 'color': 'black','weight':'bold','size':24})
ax2.text(-17,0.075, 'Most people have relatively similar BMI, the distribution follows a normal distribution', 
        {'font':'Serif', 'size':'16','color': 'black'})
ax2.text(80,0.06, 'Total',{'font':'Serif', 'size':'14','color': '#c76a48','weight':'bold'})
ax2.text(92,0.06, '=',{'font':'Serif', 'size':'14','color': 'black','weight':'bold'})
ax2.text(97,0.06, 'Stroke',{'font':'Serif', 'size':'14','color': '#343bfe','weight':'bold'})
ax2.text(113,0.06, '+',{'font':'Serif', 'size':'14','color': 'black','weight':'bold'})
ax2.text(117,0.06, 'Healthy',{'font':'Serif', 'size':'14','color': '#e6a129','weight':'bold'})


# distribution plots adding the legends and comments to the graph and save the graph


sns.kdeplot(data = dat[dat['stroke'] == 0], x = 'bmi',ax = ax3, shade = True,  alpha = 1, color = '#e6a129' )
sns.kdeplot(data = dat[dat['stroke'] == 1], x = 'bmi',ax = ax3, shade = True,  alpha = 0.8, color = '#343bfe')

ax3.set_xlabel('Body mass index of a person', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})

ax3.text(-15,0.10,'Stroke-BMI Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':24})
ax3.text(-15,0.9,'We see that Higher BMI has higher probability of getting stroke.', {'font':'Serif', 'color': 'black', 'size':16})
ax3.text(80,0.08, 'Stroke ', {'font': 'Serif','weight':'bold','Size': '16','weight':'bold','style':'normal', 'color':'#343bfe'})
ax3.text(95,0.08, '|', {'color':'black' , 'size':'16', 'weight': 'bold'})
ax3.text(97,0.08, 'Healthy', {'font': 'Serif','weight':'bold', 'Size': '16','style':'normal', 'weight':'bold','color':'#e6a129'})

fig.text(0.25,0.925,'Association between Heart Strokes and Weight',{'font':'Serif', 'weight':'bold','color': 'black', 'size':35})

plt.savefig("Heart stroke and weight")






'''Gender risk plot for heart stroke'''
stroke_gen = dat[dat['stroke'] == 1]['gender'].value_counts()
healthy_gen = dat[dat['stroke'] == 0]['gender'].value_counts()

#get the count of each gender from the dataset
female = dat['gender'].value_counts().values[0]
male =  dat['gender'].value_counts().values[1]

#find the number of stroke and healthy people from both genders
stroke_female = int(round (stroke_gen.values[0] / female * 100, 0))
stroke_male = int(round( stroke_gen.values[1] / male *100, 0))
healthy_female = int(round(healthy_gen.values[0] / female * 100, 0))
healthy_male = int(round(healthy_gen.values[1] / male *100, 0))

female_per = int(round(female/(female+male) * 100, 0))
male_per = int(round(male/(female+male)* 100, 0))


fig = plt.figure(FigureClass = Waffle, 
                 constrained_layout = True,
                 figsize = (7,7),
                 facecolor = '#f6f5f5',dpi = 100,
                 
                 plots = {'121':
                          {     
                           'rows':7,
                           'columns': 7,
                           'values' : [healthy_male,stroke_male],
                            'colors' : ['#e6a129','#343bfe'],
                              'vertical' : True,
                              'interval_ratio_y': 0.1,
                              'interval_ratio_x': 0.1,
                              'icons' : 'male',
                              'icon_legend': False,
                              'icon_size':20,
                              'plot_anchor':'C',
                              'alpha':0.1
                          },
                          '122' :
                          { 
                            'rows': 7,
                            'columns':7,
                            'values':[healthy_female,stroke_female],         
                              'colors' : ['#e6a129','#343bfe'],
                              'vertical': True,
                              'interval_ratio_y': 0.1,
                              'interval_ratio_x': 0.1,
                              'icons' : 'female',
                              'icon_legend' :False,
                              'icon_size':20,
                              'plot_anchor':'C',
                              'alpha':0.1
                                                      
                           }
                         },
                   
)

#Generate the graph for people with stroke and without stroke in both genders and save the graph
fig.text(0., 0.9, 'Gender and Heart Stroke ', {'font':'Serif', 'size':20, 'color':'black', 'weight':'bold'})
fig.text(0., 0.83, 'Risk rate of getting heart stroke in both gender are same,\nmen and women have the equal probability of getting heart stroke', {'font':'Serif', 'size':13, 'color':'black', 'weight':'normal'}, alpha = 0.7)
fig.text(0.24, 0.15, 'ooo', {'font':'Serif', 'size':16,'weight':'bold' ,'color':'#f6f5f5'})
fig.text(0.65, 0.15, 'ooo', {'font':'Serif', 'size':16,'weight':'bold', 'color':'#f6f5f5'})
fig.text(0.23, 0.2, '{}%'.format(healthy_male), {'font':'Serif', 'size':20,'weight':'bold' ,'color':'#e6a129'},alpha = 0.7,)
fig.text(0.65, 0.2, '{}%'.format(healthy_female), {'font':'Serif', 'size':20,'weight':'bold', 'color':'#e6a129'}, alpha = 0.7)
fig.text(0.21, 0.78, 'Male ({}%)'.format(male_per), {'font':'Serif', 'size':14,'weight':'bold' ,'color':'black'},alpha = 0.3,)
fig.text(0.61, 0.78, 'Female({}%)'.format(female_per), {'font':'Serif', 'size':14,'weight':'bold', 'color':'black'}, alpha = 0.3)


fig.text(0.7,0.73, 'Stroke ', {'font': 'Serif','weight':'bold','Size': '13','weight':'bold','style':'normal', 'color':'#343bfe'})
fig.text(0.82,0.73, '|', {'color':'black' , 'size':'13', 'weight': 'bold'})
fig.text(0.835,0.73, 'No Stroke', {'font': 'Serif','weight':'bold', 'Size': '13','style':'normal', 'weight':'bold','color':'#e6a129'},alpha = 1)

plt.tight_layout()
plt.savefig("Gender Risk")





'''Generate Hypertention risk plot for heart stroke'''
stroke_hyper = dat[dat['stroke'] == 1]['hypertension'].value_counts()
healthy_hyper = dat[dat['stroke'] == 0]['hypertension'].value_counts()

#Get the numbers of people who is with hypertension
n = dat['hypertension'].value_counts().values[0]
y =  dat['hypertension'].value_counts().values[1]


stroke_no = int(round (stroke_hyper.values[0] / n * 100, 0))
stroke_yes = int(round( stroke_hyper.values[1] / y *100, 0))
healthy_no = int(round(healthy_hyper.values[0] / n * 100, 0))
healthy_yes = int(round(healthy_hyper.values[1] / y *100, 0))

no_per = int(round(n/(n+y) * 100, 0))
yes_per = int(round(y/(n+y)* 100, 0))


fig = plt.figure(FigureClass = Waffle, 
                 constrained_layout = True,
                 figsize = (7,7),
                 facecolor = '#f6f5f5',dpi = 100,
                 
                 plots = {'121':
                          {     
                           'rows':7,
                           'columns': 7,
                           'values' : [stroke_yes,healthy_yes],
                            'colors' : ['#343bfe','#e6a129'],
                              'vertical' : True,
                              'interval_ratio_x': 0.005,
                              'interval_ratio_y': 0.005,
                              'icons' : 'heartbeat',
                              'icon_legend': False,
                              'icon_size':20,
                              'plot_anchor':'C',
                              'alpha':1,
                              'starting_location': 'NE'
                          },
                          '122' :
                          { 
                            'rows': 7,
                            'columns':7,
                            'values':[stroke_no,healthy_no],         
                              'colors' : ['#343bfe','#e6a129'],
                              'vertical': True,
                              'interval_ratio_x': 0.005,
                              'interval_ratio_y':0.005,
                              'icons' : 'heartbeat',
                              'icon_legend' :False,
                              'icon_size':20,
                              'plot_anchor':'C',
                              'alpha':1,
                              'starting_location': 'NE'
                                                      
                           }
                         },
                   
)

#Generate the graph for people with stroke and without stroke in both hypertention status and save the graph
fig.text(0., 0.85, 'Hypertension Risk and Heart Stroke', {'font':'Serif', 'size':20, 'color':'black', 'weight':'bold'})
fig.text(0.05, 0.73, 'People who have hypertention are more likely to get heart stroke, \nIt is about 3 times larger that people with \nhypertention having strokes than people \nwithout hypertention.', {'font':'Serif', 'size':13, 'color':'black', 'weight':'normal'},alpha = 0.8)
fig.text(0.24, 0.22, 'ooo', {'font':'Serif', 'size':16,'weight':'bold' ,'color':'#f6f5f5'})
fig.text(0.65, 0.22, 'ooo', {'font':'Serif', 'size':16,'weight':'bold', 'color':'#f6f5f5'})
fig.text(0.23, 0.28, '{}%'.format(healthy_yes), {'font':'Serif', 'size':20,'weight':'bold' ,'color':'#e6a129'},alpha = 1,)
fig.text(0.63, 0.28, '{}%'.format(healthy_no), {'font':'Serif', 'size':20,'weight':'bold', 'color':'#e6a129'}, alpha = 1)
fig.text(0.1, 0.68, 'Have Hypertension ({}%)'.format(yes_per), {'font':'Serif', 'size':14,'weight':'bold' ,'color':'black'},alpha = 0.7,)
fig.text(0.55, 0.68, "Don't have Hypertension({}%)".format(no_per), {'font':'Serif', 'size':14,'weight':'bold', 'color':'black'}, alpha = 0.7)

# Generate the legend for the graph
fig.text(0.60,0.75, 'Stroke ', {'font': 'Serif','weight':'bold','Size': '15','weight':'bold','style':'normal', 'color':'#343bfe'})
fig.text(0.72,0.75, '|', {'color':'black' , 'size':'15', 'weight': 'bold'})
fig.text(0.74,0.75, 'No Stroke', {'font': 'Serif','weight':'bold', 'Size': '15','style':'normal', 'weight':'bold','color':'#e6a129'},alpha = 1)


plt.savefig("Hypertention Risk")




'''Heart disease and stroke'''
stroke_hyper = dat[dat['stroke'] == 1]['heart_disease'].value_counts()
healthy_hyper = dat[dat['stroke'] == 0]['heart_disease'].value_counts()

#get the numbers of people with heart disease
no = dat['heart_disease'].value_counts().values[0]
yes =  dat['heart_disease'].value_counts().values[1]

#find the number of stroke and healthy people from both people with heart disease and without heart disease
stroke_no = int(round (stroke_hyper.values[0] / no * 100, 0))
stroke_yes = int(round( stroke_hyper.values[1] / yes *100, 0))
healthy_no = int(round(healthy_hyper.values[0] / no * 100, 0))
healthy_yes = int(round(healthy_hyper.values[1] / yes *100, 0))

no_per = int(round(no/(no+yes) * 100, 0))
yes_per = int(round(yes/(no+yes)* 100, 0))


fig = plt.figure(FigureClass = Waffle, 
                 constrained_layout = True,
                 figsize = (7,7),
                 facecolor = '#f6f5f5',dpi = 100,
                 
                 plots = {'121':
                          {     
                           'rows':7,
                           'columns': 7,
                           'values' : [stroke_yes,healthy_yes],
                            'colors' : ['#343bfe','#e6a129'],
                              'vertical' : True,
                              'interval_ratio_x': 0.005,
                              'interval_ratio_y': 0.005,
                              'icons' : 'heart',
                              'icon_legend': False,
                              'icon_size':20,
                              'plot_anchor':'C',
                              'alpha':0.8,
                              'starting_location': 'NE'
                          },
                          
                          '122' :
                          { 
                            'rows': 7,
                            'columns':7,
                            'values':[stroke_no,healthy_no],         
                              'colors' : ['#343bfe','#e6a129'],
                              'vertical': True,
                              'interval_ratio_x': 0.005,
                              'interval_ratio_y':0.005,
                              'icons' : 'heart',
                              'icon_legend' :False,
                              'icon_size':20,
                              'plot_anchor':'C',
                              'alpha':0.8,
                              'starting_location': 'NE'
                                                      
                           }
                         },
                   
)

#Generate the graph for people with stroke and without stroke in both heart disease status and save the graph
fig.text(0., 0.85, 'Heart disease and Heart Stroke', {'font':'Serif', 'size':20, 'color':'black', 'weight':'bold'})
fig.text(0., 0.79, 'From the graph we can see that the association between Heart disease \nand Heart Stroke is significant.', {'font':'Serif', 'size':13, 'color':'black', 'weight':'normal'}, alpha = 0.8)
fig.text(0.24, 0.22, 'ooo', {'font':'Serif', 'size':16,'weight':'bold' ,'color':'#f6f5f5'})
fig.text(0.65, 0.22, 'ooo', {'font':'Serif', 'size':16,'weight':'bold', 'color':'#f6f5f5'})
fig.text(0.25, 0.27, '{}%'.format(healthy_yes), {'font':'Serif', 'size':20,'weight':'bold' ,'color':'#2c003e'},alpha = 1,)
fig.text(0.65, 0.27, '{}%'.format(healthy_no), {'font':'Serif', 'size':20,'weight':'bold', 'color':'#2c003e'}, alpha = 1)
fig.text(0.12, 0.68, 'UnHealthy Heart ({}%)'.format(yes_per), {'font':'Serif', 'size':16,'weight':'bold' ,'color':'black'},alpha = 0.5,)
fig.text(0.55, 0.68, "Healthy Heart({}%)".format(no_per), {'font':'Serif', 'size':16,'weight':'bold', 'color':'black'}, alpha = 0.5)

# Generate the legend for the graph
fig.text(0.6,0.75, 'Stroke ', {'font': 'Serif','weight':'bold','Size': '16','weight':'bold','style':'normal', 'color':'#343bfe'})
fig.text(0.72,0.75, '|', {'color':'black' , 'size':'16', 'weight': 'bold'})
fig.text(0.74,0.75, 'No Stroke', {'font': 'Serif','weight':'bold', 'Size': '16','style':'normal', 'weight':'bold','color':'#e6a129'},alpha = 1)


plt.savefig("Heart disease risk")