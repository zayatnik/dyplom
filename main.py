from tkinter import *
from tkinter.messagebox import *
import numpy as np
from numpy import linalg as LA
import math

n = 1
myfont = 'Courier 30'

def sortbyw(w, v):
    L = len(w)
    for i in range(L):
        for j in range(L - 1):
            if w[j] > w[j + 1]:
                w[j], w[j + 1] = w[j + 1], w[j]
                for k in range(L):
                    v[k][j], v[k][j + 1] = v[k][j + 1], v[k][j]
    return w, v

def click_button():
    global n
    n = int(e1.get())
    loe1 = []
    loe2 = []
    loe3 = []
    loe4 = []
    loe5 = []
    loe6 = []
    window2 = Tk()
    window2['bg'] = 'aquamarine'
    window2.title("Ввод матриц")
    l4 = Label(window2, text = 'Матрицы взаимных связей:', font = myfont)
    l4['bg'] = 'aquamarine'
    l4.grid(row = 0, column = 0, columnspan = 3 * n + 3)
    l5 = Label(window2, text='Матрицы направленных связей:', font = myfont)
    l5['bg'] = 'aquamarine'
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
            loe1[i].append(Entry(window2, width = 5, font = myfont, justify = CENTER))
            loe1[i][j].insert(END, 0)
            loe2[i].append(Entry(window2, width = 5, font = myfont, justify = CENTER))
            loe2[i][j].insert(END, 0)
            loe3[i].append(Entry(window2, width = 5, font = myfont, justify = CENTER))
            loe3[i][j].insert(END, 0)
            loe4[i].append(Entry(window2, width = 5, font = myfont, justify = CENTER))
            loe4[i][j].insert(END, 0)
            loe5[i].append(Entry(window2, width = 5, font = myfont, justify = CENTER))
            loe5[i][j].insert(END, 0)
            loe6[i].append(Entry(window2, width = 5, font = myfont, justify = CENTER))
            loe6[i][j].insert(END, 0)
            loe1[i][j].grid(row = i + 1, column = j + 1)
            loe2[i][j].grid(row = i + 1, column = n + j + 2)
            loe3[i][j].grid(row = i + 1, column = 2 * n + j + 3)
            loe4[i][j].grid(row = n + i + 2, column = j + 1)
            loe5[i][j].grid(row = n + i + 2, column = n + j + 2)
            loe6[i][j].grid(row = n + i + 2, column = 2 * n + j + 3)
    b2 = Button(window2, text = 'OK', command = lambda: [click_button2(loe1, loe2, loe3, loe4, loe5, loe6, n), window2.destroy()], font = myfont)
    b2['bg'] = 'aquamarine'
    b2.grid(row = 2 * n + 2, column = 0, columnspan = 3 * n + 3)

def click_button2(loe1, loe2, loe3, loe4, loe5, loe6, n):
    global l2, l3, l4, l2i, l3i, l4i
    l2.destroy()
    for i in l2i:
        i.destroy()
    for i in l3i:
        i.destroy()
    l3.destroy()
    l4.destroy()
    for i in l4i:
        for j in i:
            j.destroy()
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
    er1 = False
    for i in range(n):
        for j in range(n):
            if lon[0][i][j] != 0 and i != j or lon[1][i][j] != 0 and i != j or lon[2][i][j] != 0 and i != j:
                er1 = True
                break
    if er1:
        showerror('Ошибка!', 'Матрицы взаимных связей должны быть диагональными!')
    else:
        MC = np.dot(LA.inv(lon[2]), lon[0]) #M^(-1)*C
        w, v = LA.eig(MC)
        w, v = sortbyw(w, v)
        w1 = np.sqrt(w)
        l2 = Label(window, text = 'Собственные числа:', font = myfont)
        l2.place(relx = 0.5, rely = 0.4, anchor = 'c')
        l2i = list()
        for i in range(n):
            l2i.append(Label(window, text = str(round(w1[i], 5)), font = myfont))
        for i in range(n):
            l2i[i].place(relx = 1.0 / float(n + 1) * (i + 1), rely = 0.45, anchor = 'c')
        l3 = Label(window, text = 'Cобственные частоты:', font = myfont)
        l3.place(relx = 0.5, rely = 0.55, anchor = 'c')
        l3i = list()
        for i in range(n):
            l3i.append(Label(window, text = str(round(w1[i] / (2 * math.pi), 5)), font = myfont))
        for i in range(n):
            l3i[i].place(relx = 1.0 / float(n + 1) * (i + 1), rely = 0.6, anchor='c')
        l4 = Label(window, text = 'Собственные векторы:', font = myfont)
        l4.place(relx = 0.5, rely = 0.7, anchor = 'c')
        l4i = list()
        for i in range(n):
            l4ij = list()
            for j in range (n):
                l4ij.append(Label(window, text = str(round(v[i][j], 5)), font = myfont))
            l4i.append(l4ij)
        for i in range(n):
            for j in range(n):
                l4i[i][j].place(relx = 1.0 / float(n + 1) * (j + 1), rely = 0.75 + i * 0.05, anchor = 'c')

window = Tk()
window['bg'] = 'aquamarine'
window.state('zoomed')
window.title("Поиск наиболее значимых причин самовозбуждения колебаний механических систем")
window.geometry('1600x900')
l2 = Label(window)
l2i = list()
l3 = Label(window)
l3i = list()
l4 = Label(window)
l4i = list()
l1 = Label(text = 'Введите размерность системы', font = myfont)
l1['bg'] = 'aquamarine'
e1 = Entry(window, width=10, font = myfont, justify = CENTER)
e1['bg'] = 'light cyan'
l1.place(relx = 0.5, rely = 0.1, anchor = 'c')
e1.place(relx = 0.5, rely = 0.2, anchor = 'c')
b1 = Button(text = 'OK', command = click_button, font = myfont)
b1['bg'] = 'aquamarine'
b1.place(relx = 0.5, rely = 0.3, anchor = 'c')
window.mainloop()
