from tkinter import *
import numpy as np
from numpy import linalg as LA
import math

n = 1
loe1 = []
loe2 = []
loe3 = []
loe4 = []
loe5 = []
loe6 = []
myfont = 'Times 30'

def click_button():
    global n, loe1, loe2, loe3, loe4, loe5, loe6
    n = int(e1.get())
    window2 = Tk()
    window2.title("Ввод матриц")
    l4 = Label(window2, text = 'Матрицы взаимных связей:', font = myfont)
    l4.grid(row = 0, column = 0, columnspan = 3 * n + 3)
    l5 = Label(window2, text='Матрицы направленных связей:', font = myfont)
    l5.grid(row = n + 1, column = 0, columnspan = 3 * n + 3)
    l61 = Label(window2, text = 'C:', font = myfont)
    l61.grid(row = 1, column = 0)
    l62 = Label(window2, text='B:', font = myfont)
    l62.grid(row=1, column=n + 1)
    l63 = Label(window2, text = 'M:', font = myfont)
    l63.grid(row = 1, column = 2*n + 2)
    l64 = Label(window2, text='C:', font = myfont)
    l64.grid(row=n + 2, column=0)
    l65 = Label(window2, text='B:', font = myfont)
    l65.grid(row=n + 2, column=n + 1)
    l66 = Label(window2, text='M:', font = myfont)
    l66.grid(row=n + 2, column=2*n + 2)
    for i in range(n):
        loe1.append([])
        loe2.append([])
        loe3.append([])
        loe4.append([])
        loe5.append([])
        loe6.append([])
        for j in range(n):
            loe1[i].append(Entry(window2, width=5, font = myfont))
            loe1[i][j].insert(END, 0)
            loe2[i].append(Entry(window2, width=5, font = myfont))
            loe2[i][j].insert(END, 0)
            loe3[i].append(Entry(window2, width=5, font = myfont))
            loe3[i][j].insert(END, 0)
            loe4[i].append(Entry(window2, width=5, font = myfont))
            loe4[i][j].insert(END, 0)
            loe5[i].append(Entry(window2, width=5, font = myfont))
            loe5[i][j].insert(END, 0)
            loe6[i].append(Entry(window2, width=5, font = myfont))
            loe6[i][j].insert(END, 0)
            loe1[i][j].grid(row = i + 1, column = j + 1)
            loe2[i][j].grid(row = i + 1, column = n + j + 2)
            loe3[i][j].grid(row = i + 1, column = 2 * n + j + 3)
            loe4[i][j].grid(row = n + i + 2, column = j + 1)
            loe5[i][j].grid(row = n + i + 2, column = n + j + 2)
            loe6[i][j].grid(row = n + i + 2, column = 2 * n + j + 3)
    b2 = Button(window2, text='OK', command = click_button2, font = myfont)
    b2.grid(row = 2 * n + 2, column = 0, columnspan = 3 * n + 3)

def click_button2():
    global n, loe1, loe2, loe3, loe4, loe5, loe6
    t = []
    t.append(loe1[0][0].get())
    t.append(loe2[0][0].get())
    t.append(loe3[0][0].get())
    t.append(loe4[0][0].get())
    t.append(loe5[0][0].get())
    t.append(loe6[0][0].get())
    lon = []
    for i in range(6):
        lon.append([])
    for i in range(n):
        for k in range(6):
            lon[k].append([])
        for j in range(n):
            lon[0][i].append(float(loe1[i][j].get())) #C1
            lon[1][i].append(float(loe2[i][j].get())) #B1
            lon[2][i].append(float(loe3[i][j].get())) #M1
            lon[3][i].append(float(loe4[i][j].get())) #C2
            lon[4][i].append(float(loe5[i][j].get())) #B2
            lon[5][i].append(float(loe6[i][j].get())) #M2
    MC = np.dot(LA.inv(lon[2]), lon[0]) #M^(-1)*C
    w, v = LA.eig(MC)
    w1 = np.sqrt(w)
    l2 = Label(window, text = 'Собственные числа: ' + str(w1), font = myfont)
    l2.pack()
    l3 = Label(window, text = 'Cобственные частоты: ' + str(w1 / (2 * math.pi)), font = myfont)
    l3.pack()
    l4 = Label(window, text='Собственные векторы:\n ' + str(v), font = myfont)
    l4.pack()

window = Tk()
window.state('zoomed')
window.title("Поиск наиболее значимых причин самовозбуждения колебаний механических систем")
window.geometry('1600x900')
l1 = Label(text = 'Введите размерность системы', font = myfont)
l1.place(rely=.1/2, anchor="c")
e1 = Entry(window, width=10, font = myfont)
e1.place(rely=.1, anchor="c")
l1.pack()
e1.pack()
b1 = Button(text = 'OK', command = click_button, font = myfont)
b1.pack()
window.mainloop()
