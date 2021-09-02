from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
import tkinter.messagebox
import matplotlib
matplotlib.use('Agg')
from PIL import ImageTk,Image 
import pandas as pd
import joblib
Model1 = joblib.load('Model1.pkl')


def Predict(ThePatient):
    # ThePatient = [gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smorking_status]
    
    # create an empty dataframe
    ds = pd.DataFrame(columns=['Age', 'Male', 'HT_Yes', 'HD_Yes', 'EM_Yes', 'Never_worked', 'Private', 'Self-employed', 'children', 'BMI', 'Avg. Glucose Level',"RT_Urban","SS_formerly smoked","SS_never smoked","SS_smokes"])
    
    # fill in the dataframe by ThePatient
    ds['Age'] = [ThePatient[1]]

    if ThePatient[0] == 'Male':
        ds['Male'] = 1
    else:
        ds['Male'] = 0

    ds['HT_Yes'] = [ThePatient[2]]

    ds['HD_Yes'] = [ThePatient[3]]

    ds['EM_Yes'] = [ThePatient[4]]
    ds["EM_Yes"].replace(["No","Yes"], [0,1], inplace=True)

    if ThePatient[5] == 'Never_worked':
        ds['Never_worked'] = 1
    else:
        ds['Never_worked'] = 0
    if ThePatient[5] == 'Private':
        ds['Private'] = 1
    else:
        ds['Private'] = 0
    if ThePatient[5] == 'Self-employed':
        ds['Self-employed'] = 1
    else:
        ds['Self-employed'] = 0
    if ThePatient[5] == 'children':
        ds['children'] = 1
    else:
        ds['children'] = 0

    ds['BMI'] = [ThePatient[8]]

    ds['Avg. Glucose Level'] = [ThePatient[7]]
    
    if ThePatient[6] == 'Urban':
        ds['RT_Urban'] = 1
    else:
        ds['RT_Urban'] = 0

    if ThePatient[9] == 'formerly smoked':
        ds['SS_formerly smoked'] = 1
    else:
        ds['SS_formerly smoked'] = 0
    if ThePatient[9] == 'never smoked':
        ds['SS_never smoked'] = 1
    else:
        ds['SS_never smoked'] = 0
    if ThePatient[9] == 'smokes':
        ds['SS_smokes'] = 1
    else:
        ds['SS_smokes'] = 0


    #Model predict
    y = Model1.predict(ds)

    #Output
    if y == [1]:
        return "It is highly possible that the patient is stroke"
    else:
        return "It is highly possible that the patient is not stroke"



#==========================================================

def initial_window():
    """the initial introduction frame"""
    enter = tk.Tk()
    enter.title('Stroke Prediction Tools')
    enter.geometry('1000x1000')
    c = tk.Canvas(enter, width = 1000, height = 1000, bg="white")
    c.pack()
    c.create_text(500,20, text = "Introduction")
    c.create_text(500,120, text="""

    This is a stroke prediction tool, it needs you [bmi, gluco_level, gender, hypertension_status, 
    
    heartdisease_status, marry_status, work_status, resid_status, smoke_status, and age]
    
    We will not store your personal information, and we only use your personal information to do stroke prediction.

    The accurate rate of this prediction is around 75%, we are not garanted the prediction is true.
    
    If you would like to quit, please click the upper close button. 
    
    If you would like to use this program, please clict continue button.""")
    

    btn = tk.Button(enter,text='Continue',fg="black",width=7,compound='center',\
                      bg = "white",command = lambda :original_window(enter))
    btn.pack()
    btn.place(x=500,y=350)

    img1 = Image.open("Duke_Chapel.png")
    realized_img = img1.resize((610,400))
    img = ImageTk.PhotoImage(realized_img)  
    c.create_image(520, 400, anchor='n', image=img)
    enter.mainloop()

def original_window(enter):
    """build the main frame"""
    enter_w = tk.Toplevel(enter)
    enter_w.title('Stroke Prediction Tools')
    enter_w.geometry('1000x1000')

    global n1,n2, n3, n4, n5, n6, n7, n8, n9, n10
    
    #gender
    lab_3 = tk.Label(enter_w,width=7,text='gender')
    lab_3.place(x=300,y=15)
    lab_3.pack()
    n3 = tk.StringVar()
    r11 = tk.Radiobutton(enter_w, text='Male', variable=n3, value='Male')
    r11.place(x=200,y=50)
    r11.pack()
    r12 = tk.Radiobutton(enter_w, text='Female', variable=n3, value='Female')
    r12.place(x=200,y=50)
    r12.pack()
    
    #hypertension
    lab_4 = tk.Label(enter_w,width=12,text='Hypertension')
    lab_4.pack()
    n4 = tkinter.StringVar()
    hyper = {"Positive":1, "Negative":0}
    optionMenu1 = tkinter.OptionMenu(enter_w, n4, *hyper.keys())
    optionMenu1.pack(pady=2)

    #heartdisease
    lab_5 = tk.Label(enter_w,width=12,text='HeartDisease')
    lab_5.pack()
    n5 = tkinter.StringVar()
    heart = {"Positive":1, "Negative":0}
    optionMenu2 = tkinter.OptionMenu(enter_w, n5, *heart.keys())
    optionMenu2.pack(pady=2)

    #ever_married
    lab_6 = tk.Label(enter_w,width=12,text='Married_Status')
    lab_6.pack()
    n6 = tkinter.StringVar()
    m = {"Married":"Yes", "NonMarried":"No"}
    optionMenu3 = tkinter.OptionMenu(enter_w, n6, *m.keys())
    optionMenu3.pack(pady=2)

    #Work Status
    lab_7 = tk.Label(enter_w,width=12,text='Work_Type')
    lab_7.pack()
    n7 = tkinter.StringVar()
    w = {"Children":"children", "Goverment_Job":"Govt_job", "Never_Worked":"Never_worked", \
        "Private":"Private", "Self_Employed":"Self-employed"}
    optionMenu4 = tkinter.OptionMenu(enter_w, n7, *w.keys())
    optionMenu4.pack(pady=2)

    #Residence
    lab_8 = tk.Label(enter_w,width=14,text='Residence Type')
    lab_8.pack()
    n8 = tkinter.StringVar()
    re = {"Rural":"Rural", "Urban":"Urban"}
    optionMenu5 = tkinter.OptionMenu(enter_w, n8, *re.keys())
    optionMenu5.pack(pady=2)

    #Smoking Status
    lab_9 = tk.Label(enter_w,width=14,text='Smoking Status')
    lab_9.pack()
    n9 = tkinter.StringVar()
    smoke = {"Smokes":"smokes", "Formely_Smoked":"formerly smoked", \
        "Never_Smoked":"never smoked", "Unknown":"Unknown"}
    optionMenu6 = tkinter.OptionMenu(enter_w, n9, *smoke.keys())
    optionMenu6.pack(pady=2)

    lab_1 = tk.Label(enter_w,width=7,text='BMI',compound='center')
    lab_1.place(x=305,y=380)
   
    lab_2 = tk.Label(enter_w,width=12,text="""AvgGlucose""",compound='center')
    lab_2.place(x=295,y=405)

    lab_10 = tk.Label(enter_w,width=7,text="""Age""",compound='center')
    lab_10.place(x=305,y=430)

   #bmi input frame
    n1 = tk.StringVar()
    entry = tk.Entry(enter_w,textvariable=n1, highlightcolor='red', highlightthickness=1)
    entry.pack()
    
    
   #glu input frame
    n2 = tk.StringVar()
    entry_1 = tk.Entry(enter_w,textvariable=n2, highlightcolor='red', highlightthickness=1)
    entry_1.pack()

   #age input frame
    n10 = tk.StringVar()
    entry_2 = tk.Entry(enter_w,textvariable=n10, highlightcolor='red', highlightthickness=1)
    entry_2.pack()

   #duke picture
    canvas = tk.Canvas(enter_w, width = 1000, height = 1000)  
    canvas.pack()
    img1 = Image.open("duke.png")
    realized_img = img1.resize((310,200))
    img = ImageTk.PhotoImage(realized_img)  
    canvas.create_image(200, 50, anchor='n', image=img)

    #creator's name
    img2 = Image.open("created.png")
    realized_img1 = img2.resize((250,100))
    img_c = ImageTk.PhotoImage(realized_img1)  
    canvas.create_image(850, 200, anchor='n', image=img_c)


    # check the input value is within the valid range
    def check(enter_w):
        if (float(entry_2.get())<0.0):
            tk.messagebox.showerror('*_*','Age value should be larger than 0')
        elif (float(entry_2.get()) > 100.0):
            tk.messagebox.showerror('*_*','Age value should be less than 100')
        elif (float(entry_1.get())>300.0):
            tk.messagebox.showerror('*_*','Glucose_value is too large')
        elif (float(entry_1.get())<20.0):
            tk.messagebox.showerror('*_*','Glucose_value is too small')
        elif (float(entry.get())<=0.0):
            tk.messagebox.showerror('*_*','BMI should be larger than 0')
        elif (float(entry.get())>150.0):
            tk.messagebox.showerror('*_*','BMI is too large')
        else:
            tk.messagebox.showinfo('^_^','correct input value')
            new_window(enter_w, float(entry.get()), float(entry_1.get()), n3.get(), hyper[n4.get()],\
                heart[n5.get()], m[n6.get()], w[n7.get()], re[n8.get()], smoke[n9.get()], float(entry_2.get()))

    #calculate button ===> start check function
    btn = tk.Button(enter_w,text='calculate',fg="black",width=7,compound='center',\
                      bg = "white",command = lambda :check(enter_w))
    btn.pack()
    btn.place(x=470,y=570)
    enter_w.mainloop()

#final result window:
def new_window(enter_w, bmi, gluco, gender, hyper, heart, marr, work, resid, smoke, age):
    """final output window"""
    window_one = tk.Toplevel(enter_w)
    window_one.geometry('500x500')
    window_one.title('output_prediction')
    patient = []
    patient = [gender, age, hyper, heart, marr, work, resid, gluco, bmi, smoke]
    #print(patient)

    #ThePatient = [gender, age, hypertension, heart_disease, ever_married, 
    # work_type, Residence_type, avg_glucose_level, bmi, smorking_status]
    a = Predict(patient)
    
    Lab = tk.Label(window_one,text=f'''The prediction result is :
    
    {a}

    Prediction result is for reference only

    Return to main window could start a new test''', compound = tk.CENTER)
    Lab.place(x=100,y=50)
    Lab.pack()
initial_window()



