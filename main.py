from tkinter import *
from tkinter.messagebox import *
import numpy as np
from numpy import linalg as LA
import math
import networkx as nx
import matplotlib.pyplot as plt


n = 1
myfont = 'Courier 30'
colors = ['0', 'g', 'b']
arrowstyles = ['-', '<-']


def firsts(path):
    res = []
    for i in path:
        res.append(i[0])
    return res


def dfs(graph, start, end):
    fringe = [(start, [])]
    while fringe:
        state, path = fringe.pop()
        if path and state == end:
            yield path
            continue
        for next_state in graph[state]:
            if next_state[0] in firsts(path):
                continue
            fringe.append((next_state[0], path+[next_state]))


def sortbyw(w, v):
    L = len(w)
    for i in range(L):
        for j in range(L - 1):
            if w[j] > w[j + 1]:
                w[j], w[j + 1] = w[j + 1], w[j]
                for k in range(L):
                    v[k][j], v[k][j + 1] = v[k][j + 1], v[k][j]
    return w, v


class Cycle:
    def __init__(self, cycle):
        self.cycle = cycle

    def __str__(self):
        res = str(self.cycle[0])
        for i in self.cycle[1:]:
            res += f' -->{(i[1])}--> {i[0]}'
        return res


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
    er1 = er2 = er3 = er4 = False
    for i in range(n):
        for j in range(n):
            if lon[0][i][j] != lon[0][j][i] or lon[1][i][j] != lon[1][j][i] or lon[2][i][j] != lon[2][j][i]:
                er1 = True
                break
            if lon[3][i][j] != 0 and i == j or lon[4][i][j] != 0 and i == j or lon[5][i][j] != 0 and i == j:
                er2 = True
                break
            if lon[3][i][j] == 0 and lon[3][j][i] == 0 and i != j or lon[4][i][j] == 0 and lon[4][j][i] == 0 and i !=j or lon[5][i][j] == 0 and lon[5][j][i] == 0 and i !=j:
                er3 = True
                break
            if lon[3][i][j] != 0 and lon[3][j][i] != 0 and i != j or lon[4][i][j] != 0 and lon[4][j][i] != 0 and i !=j or lon[5][i][j] != 0 and lon[5][j][i] != 0 and i !=j:
                er3 = True
                break
            if lon[0][i][j] == 0 and i == j or lon[1][i][j] == 0 and i == j or lon[2][i][j] == 0 and i == j:
                er4 = True
                break
    if er1:
        showerror('Ошибка!', 'Матрицы взаимных связей должны быть симметричными!')
    elif er2:
        showerror('Ошибка!', 'Матрицы направленных связей должны иметь нулевые диагонали!')
    elif er3:
        showerror('Ошибка!', 'Если один элемент матрицы напрвленных связей равен нулю, то другой должен быть ненулевым, и наоборот!')
    elif er4:
        showerror('Ошибка!', 'Диагонали матриц взаимных связей не должны содержать нулей!')
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
            l3i.append(Label(window, text = str(round(w1[i] / (2 * math.pi), 5)) + ' Гц', font = myfont))
        for i in range(n):
            l3i[i].place(relx = 1.0 / float(n + 1) * (i + 1), rely = 0.6, anchor='c')
        l4 = Label(window, text = 'Собственные векторы:', font = myfont)
        l4.place(relx = 0.5, rely = 0.7, anchor = 'c')
        l4i = list()
        for i in range(n):
            l4ij = list()
            for j in range(n):
                l4ij.append(Label(window, text = str(round(v[i][j], 5)), font = myfont))
            l4i.append(l4ij)
        for i in range(n):
            for j in range(n):
                l4i[i][j].place(relx = 1.0 / float(n + 1) * (j + 1), rely = 0.75 + i * 0.05, anchor = 'c')

        graph = []
        for i in range(n):
            graph.append([])
        for i in range(3):
            for j in range(n):
                for k in range(j):
                    if lon[i][j][k] != 0:
                        graph[j].append((k, i))
                        graph[k].append((j, i))
        for i in range(3, 6):
            for j in range(n):
                for k in range(n):
                    if lon[i][j][k] > 0:
                        graph[j].append((k, i))
        cycles = [[node] + path for node in range(n) for path in dfs(graph, node, node)]
        good_cycles = sorted(cycles)
        very_good_cycles = []
        for i in good_cycles:
            very_good_cycles.append(Cycle(i))

        types = {}
        for i, val in enumerate(graph):
            for j in val:
                if not (i, j[0]) in types:
                    types[(i, j[0])] = []
                types[(i, j[0])].append(j[1])
        for i in types:
            for j in types:
                if (j[1], j[0]) == (i[0], i[1]):
                    for k in types[i]:
                        if k < 3 and k in types[j]:
                            types[j].remove(k)

        G = nx.DiGraph()
        for i in types.keys():
            G.add_edges_from([i])
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_color='r', node_size=5000, alpha=1)
        nx.draw_networkx_labels(G, pos)
        ax = plt.gca()
        for e in G.edges:
            for i in types[e]:
                ax.annotate("",
                            xy=pos[e[0]], xycoords='data',
                            xytext=pos[e[1]], textcoords='data',
                            arrowprops=dict(arrowstyle=arrowstyles[i // 3], color=colors[i % 3],
                                            shrinkA=40, shrinkB=40,
                                            patchA=None, patchB=None,
                                            connectionstyle="arc3,rad=rrr".replace('rrr', str(0.1 * i), ), linewidth=4
                                            ),
                            )
        xmin, xmax, ymin, ymax = 0, 0, 0, 0
        for i in pos.values():
            if i[0] < xmin:
                xmin = i[0]
            if i[0] > xmax:
                xmax = i[0]
            if i[1] < ymin:
                ymin = i[1]
            if i[1] > ymax:
                ymax = i[1]
        plt.xlim([xmin - 0.5, xmax + 0.5])
        plt.ylim([ymin - 0.5, ymax + 0.5])
        plt.axis('off')
        plt.show()

        for i in very_good_cycles:
            flag = False
            for j in i.cycle[1:]:
                if j[1] > 2:
                    flag = True
                    break
            if not flag:
                very_good_cycles.remove(i)

        for i in very_good_cycles:
            print(i)
            for j in range(n):
                print(f'Работа по {j}-й форме:')
                sum = 0
                i_ind = i.cycle[0]
                for k in i.cycle[1:]:
                    j_ind = k[0]
                    type = k[1]
                    if type > 2:
                        sum += lon[type][i_ind][j_ind] * v[i_ind][j] * v[j_ind][j]
                    sum *= w[j] ** (type - 2)
                print(sum)




window = Tk()
window['bg'] = 'aquamarine'
window.state('zoomed')
window.title("Поиск наиболее значимых причин самовозбуждения колебаний механических систем")
window.geometry('1600x900')
l2 = l3 = l4 = Label(window)
l2i = l3i = l4i = list()
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
