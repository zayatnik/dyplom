from tkinter import *
from tkinter.messagebox import *
import numpy as np
from numpy import linalg as la
import math
import networkx as nx
import matplotlib.pyplot as plt


n = 1
my_font, my_font2 = 'Courier 30', 'Courier 15'
colors, arrows_colors = ['0', 'g', 'b'], ['black', 'green', 'blue']
arrows_styles1, arrows_styles2 = ['-', '<-'], ['---', '-->']


def search_cycles(graph: iter, start: int, end: int):
    fringe = [(start, [])]
    while fringe:
        state, path = fringe.pop()
        if path and state == end:
            yield path
            continue
        for next_state in graph[state]:
            if next_state[0] in [i[0] for i in path]:
                continue
            fringe.append((next_state[0], path + [next_state]))


def sort_by_w(w, v) -> tuple:
    length = len(w)
    for i in range(length):
        for j in range(length - 1):
            if w[j] > w[j + 1]:
                w[j], w[j + 1] = w[j + 1], w[j]
                for k in range(length):
                    v[k][j], v[k][j + 1] = v[k][j + 1], v[k][j]
    return w, v


def get_graph_and_cycles_from_lon(lon: list) -> tuple:
    global n
    graph = [[] for _ in range(n)]
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
    cycles = [[node] + path for node in range(n) for path in search_cycles(graph, node, node)]
    good_cycles = sorted(cycles)
    very_good_cycles = [Cycle(graph) for graph in good_cycles]
    return graph, very_good_cycles


def get_types_from_graph(graph: list) -> dict:
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
    return types


def clean_cycles(cycles: list) -> list:
    i = 0
    while i < len(cycles):
        flag = False
        for j in cycles[i].cycle[1:]:
            if j[1] > 2:
                flag = True
                i += 1
                break
        if not flag:
            cycles.remove(cycles[i])
    return cycles


def total_work(cycles: list, lon: list, v: list, w1: list) -> list:
    global n
    work = [0 for _ in range(n)]
    for i in cycles:
        for j in range(n):
            sum, i_ind = 0, i.cycle[0]
            for k in i.cycle[1:]:
                j_ind = k[0]
                arrow_type = k[1]
                if arrow_type > 2:
                    sum += lon[arrow_type][i_ind][j_ind] * v[i_ind][j] * v[j_ind][j] * (w1[j] / 2 / math.pi) **\
                           (arrow_type - 2)
                i_ind = j_ind
            work[j] += sum
    return work


def unstable_forms(works: list) -> list:
    res = []
    for i, val in enumerate(works):
        if val > 0:
            res.append(i)
    if len(res) > 2:
        res.remove(works.index(min(works)))
    elif len(res) == 0:
        res.append(works.index(max(works)))
    return res


class Cycle:
    def __init__(self, cycle):
        self.cycle = cycle

    def __str__(self):
        res = str(self.cycle[0])
        for i in self.cycle[1:]:
            res += f'{arrows_styles2[i[1] // 3]}{i[0]}'
        return res


def click_button1():
    global n
    n = int(e1.get())
    loe1, loe2, loe3 = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    loe4, loe5, loe6 = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    window2 = Tk()
    window2['bg'] = 'aquamarine'
    window2.title('Ввод матриц')

    l4 = Label(window2, text='Матрицы взаимных связей:', font=my_font, bg='aquamarine')
    l4.grid(row=0, column=0, columnspan=3 * n + 3)
    l5 = Label(window2, text='Матрицы направленных связей:', font=my_font, bg='aquamarine')
    l5.grid(row=n + 1, column=0, columnspan=3 * n + 3)
    l61, l62 = Label(window2, text='C:', font=my_font), Label(window2, text='B:', font=my_font)
    l61.grid(row=1, column=0)
    l62.grid(row=1, column=n + 1)
    l63, l64 = Label(window2, text='M:', font=my_font), Label(window2, text='C:', font=my_font)
    l63.grid(row=1, column=2 * n + 2)
    l64.grid(row=n + 2, column=0)
    l65, l66 = Label(window2, text='B:', font=my_font), Label(window2, text='M:', font=my_font)
    l65.grid(row=n + 2, column=n + 1)
    l66.grid(row=n + 2, column=2*n + 2)

    for i in range(n):
        for j in range(n):
            loe1[i].append(Entry(window2, width=5, font=my_font, justify=CENTER))
            loe2[i].append(Entry(window2, width=5, font=my_font, justify=CENTER))
            loe3[i].append(Entry(window2, width=5, font=my_font, justify=CENTER))
            loe4[i].append(Entry(window2, width=5, font=my_font, justify=CENTER))
            loe5[i].append(Entry(window2, width=5, font=my_font, justify=CENTER))
            loe6[i].append(Entry(window2, width=5, font=my_font, justify=CENTER))
            if i == j:
                loe1[i][j].insert(END, 1)
                loe2[i][j].insert(END, 1)
                loe3[i][j].insert(END, 1)
            else:
                loe1[i][j].insert(END, 0)
                loe2[i][j].insert(END, 0)
                loe3[i][j].insert(END, 0)
            loe4[i][j].insert(END, 0)
            loe5[i][j].insert(END, 0)
            loe6[i][j].insert(END, 0)
            loe1[i][j].grid(row=i + 1, column=j + 1)
            loe2[i][j].grid(row=i + 1, column=n + j + 2)
            loe3[i][j].grid(row=i + 1, column=2 * n + j + 3)
            loe4[i][j].grid(row=n + i + 2, column=j + 1)
            loe5[i][j].grid(row=n + i + 2, column=n + j + 2)
            loe6[i][j].grid(row=n + i + 2, column=2 * n + j + 3)

    b2 = Button(window2, text='OK', command=lambda: [click_button2(loe1, loe2, loe3, loe4, loe5, loe6),
                                                     window2.destroy()], font=my_font, bg='aquamarine')
    b2.grid(row=2 * n + 2, column=0, columnspan=3 * n + 3)


def click_button2(loe1: list, loe2: list, loe3: list, loe4: list, loe5: list, loe6: list):
    global l2, l3, l4, l2i, l3i, l4i, n
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

    lon = [[[] for __ in range(n)] for _ in range(6)]
    for i in range(n):
        for j in range(n):
            lon[0][i].append(float(loe1[i][j].get()))  # C1
            lon[1][i].append(float(loe2[i][j].get()))  # B1
            lon[2][i].append(float(loe3[i][j].get()))  # M1
            lon[3][i].append(float(loe4[i][j].get()))  # C2
            lon[4][i].append(float(loe5[i][j].get()))  # B2
            lon[5][i].append(float(loe6[i][j].get()))  # M2

    for i in range(n):
        for j in range(n):
            if lon[0][i][j] != lon[0][j][i] or lon[1][i][j] != lon[1][j][i] or lon[2][i][j] != lon[2][j][i]:
                showerror('Ошибка!', 'Матрицы взаимных связей должны быть симметричными!')
                return None
            if i == j and (lon[3][i][j] != 0 or lon[4][i][j] != 0 or lon[5][i][j] != 0):
                showerror('Ошибка!', 'Матрицы направленных связей должны иметь нулевые диагонали!')
                return None
            if i != j and (lon[3][i][j] != 0 and lon[3][j][i] != 0 or lon[4][i][j] != 0 and lon[4][j][i] != 0 or
                           lon[5][i][j] != 0 and lon[5][j][i] != 0):
                showerror('Ошибка!', 'Хотя бы один из двух противоположных элементов в матрицах направленных связей '
                                     'должен быть нулевым!')
                return None
            if i == j and (lon[0][i][j] == 0 or lon[1][i][j] == 0 or lon[2][i][j] == 0):
                showerror('Ошибка!', 'Диагонали матриц взаимных связей не должны содержать нулей!')
                return None

    try:
        mc = np.dot(la.inv(lon[2]), lon[0])  # M⁻¹C
    except la.LinAlgError:
        showerror('Ошибка!', 'Вырожденная матрица М!')
        return None
    w, v = la.eig(mc)  # СВ, СЧ
    w, v = sort_by_w(w, v)
    for i in w:
        if i < 0:
            showerror('Ошибка!', 'Некорректные матрицы взаимных связей: квадрат собственного числа матрицы M⁻¹C '
                                 'отрицательный!')
            return None

        if isinstance(i, np.complex128):
            showerror('Ошибка!', 'Некорректные матрицы взаимных связей: квадрат собственного числа матрицы M⁻¹C '
                                 'комплексный!')
            return None
    w1 = np.sqrt(w)

    l2 = Label(window, text='Собственные числа:', font=my_font)
    l2.place(relx=0.5, rely=0.4, anchor='center')
    l2i, l3i, l4i = [], [], []
    for i in range(n):
        l2i.append(Label(window, text=str(round(w1[i], 5)), font=my_font))
        l2i[i].place(relx=1.0 / float(n + 1) * (i + 1), rely=0.45, anchor='center')
        l3i.append(Label(window, text=str(round(w1[i] / (2 * math.pi), 5)) + ' Гц', font=my_font))
        l3i[i].place(relx=1.0 / float(n + 1) * (i + 1), rely=0.6, anchor='center')
        l4ij = []
        for j in range(n):
            l4ij.append(Label(window, text=str(round(v[i][j], 5)), font=my_font))
        l4i.append(l4ij)
        for j in range(n):
            l4i[i][j].place(relx=1.0 / float(n + 1) * (j + 1), rely=0.75 + i * 0.05, anchor='center')
    l3 = Label(window, text='Cобственные частоты:', font=my_font)
    l3.place(relx=0.5, rely=0.55, anchor='center')
    l4 = Label(window, text='Собственные векторы:', font=my_font)
    l4.place(relx=0.5, rely=0.7, anchor='center')

    graph, cycles = get_graph_and_cycles_from_lon(lon)
    types = get_types_from_graph(graph)
    types_for_picture = {}
    for key, value in types.items():
        types_for_picture[(key[0] + 1, key[1] + 1)] = value

    g = nx.DiGraph()
    for i in types_for_picture.keys():
        g.add_edges_from([i])
    pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, pos, node_color='r', node_size=5000, alpha=1)
    nx.draw_networkx_labels(g, pos)
    ax = plt.gca()
    for e in g.edges:
        for i in types_for_picture[e]:
            ax.annotate('', xy=pos[e[0]], xycoords='data', xytext=pos[e[1]], textcoords='data',
                        arrowprops=dict(arrowstyle=arrows_styles1[i // 3], color=colors[i % 3], shrinkA=40,
                                        shrinkB=40, patchA=None, patchB=None,
                                        connectionstyle="arc3,rad=rrr".replace('rrr', str(0.1 * i), ), linewidth=4))
    x_min, x_max, y_min, y_max = 0, 0, 0, 0
    for i in pos.values():
        if i[0] < x_min:
            x_min = i[0]
        if i[0] > x_max:
            x_max = i[0]
        if i[1] < y_min:
            y_min = i[1]
        if i[1] > y_max:
            y_max = i[1]
    x, y = [1000], [1000]
    plt.plot(x, y, 'black', label='связи по координате (С)')
    plt.plot(x, y, 'g', label='связи по скорости (B)')
    plt.plot(x, y, 'b', label='связи по ускорению (M)')
    plt.legend()
    plt.xlim([x_min - 0.5, x_max + 0.5])
    plt.ylim([y_min - 0.5, y_max + 0.5])
    plt.axis('off')
    plt.savefig('graph.png')
    plt.close()

    cycles = clean_cycles(cycles)
    work = total_work(cycles, lon, v, w1)

    font_for_cycles = my_font if len(cycles) < 16 else my_font2
    window3 = Tk()
    window3.title('Работа циклов')
    window3['bg'] = 'aquamarine'
    label1 = Label(window3, text='цикл', font=font_for_cycles, bg='aquamarine')
    label1.grid(row=0, column=0, columnspan=n * n + n - 1)
    for i in range(n):
        label2 = Label(window3, text=f' работа по \n {i + 1}-й форме ', font=font_for_cycles, bg='aquamarine')
        label2.grid(row=0, column=i + n * n + n)
    row, new_row, delta = 1, 0, n * (n + 2)
    for i in cycles:
        label3 = Label(window3, text=str(i.cycle[0] + 1), font=font_for_cycles, bg='aquamarine')
        if row > 27:
            label3.grid(row=new_row, column=delta)
        else:
            label3.grid(row=row, column=0)
        column = 1
        for j in i.cycle[1:]:
            label31 = Label(window3, text=arrows_styles2[j[1] // 3], font=font_for_cycles, fg=arrows_colors[j[1] % 3],
                            bg='aquamarine')
            if row > 27:
                label31.grid(row=new_row, column=delta + column)
            else:
                label31.grid(row=row, column=column)
            column += 1
            label31 = Label(window3, text=str(j[0] + 1), font=font_for_cycles, bg='aquamarine')
            if row > 27:
                label31.grid(row=new_row, column=delta + column)
            else:
                label31.grid(row=row, column=column)
            column += 1
        for j in range(n):
            sum, i_ind = 0, i.cycle[0]
            for k in i.cycle[1:]:
                j_ind = k[0]
                arrow_type = k[1]
                if arrow_type > 2:
                    sum += lon[arrow_type][i_ind][j_ind] * v[i_ind][j] * v[j_ind][j] * (w1[j] / 2 / math.pi) **\
                           (arrow_type - 2)
                i_ind = j_ind
            if sum:
                label4 = Label(window3, text=f' {str(round(sum, 2))} - {str(round(sum / work[j] * 100, 2))}% ',
                               font=font_for_cycles, bg='aquamarine')
            else:
                label4 = Label(window3, text=f' {str(round(sum, 2))} - 0.0% ', font=font_for_cycles, bg='aquamarine')
            if row > 27:
                label4.grid(row=new_row, column=delta + j + n * n + n)
            else:
                label4.grid(row=row, column=j + n * n + n)
        row += 1
        if row > 27:
            new_row += 1
            label1 = Label(window3, text='цикл', font=font_for_cycles, bg='aquamarine')
            label1.grid(row=0, column=delta, columnspan=n * n + n - 1)
            for k in range(n):
                label2 = Label(window3, text=f' работа по \n {k + 1}-й форме ', font=font_for_cycles, bg='aquamarine')
                label2.grid(row=0, column=k + n * n + n + delta)
    label5 = Label(window3, text='суммарная работа', font=font_for_cycles, bg='aquamarine')
    if new_row:
        label5.grid(row=new_row, column=delta, columnspan=n * n + n - 1)
    else:
        label5.grid(row=row, column=0, columnspan=n * n + n - 1)
    for i in range(n):
        label51 = Label(window3, text=str(round(work[i], 5)), font=font_for_cycles, bg='aquamarine')
        if new_row:
            label51.grid(row=new_row, column=delta + n * n + n + i)
        else:
            label51.grid(row=row, column=n * n + n + i)

    window4 = Tk()
    window4.title('Потенциально неустойчивые формы и наиболее значимые связи')
    window4['bg'] = 'aquamarine'
    forms, forms_text = unstable_forms(work), ''
    if len(forms) > 1:
        label6 = Label(window4, text=f'Потенциально неустойчивыми являются формы {forms[0] + 1} и {forms[1] + 1}',
                       font=my_font, bg='aquamarine')
    else:
        label6 = Label(window4, text=f'Потенциально неустойчивой является форма {forms[0] + 1}', font=my_font,
                       bg='aquamarine')
    label6.grid(row=0, column=0)
    row = 1
    for i in forms:
        label61 = Label(window4, text=f'наиболее значимые циклы по {i + 1}-й форме:', font=my_font, bg='aquamarine')
        label61.grid(row=row, column=0)


window = Tk()
window['bg'] = 'aquamarine'
window.state('zoomed')
window.title("Поиск наиболее значимых причин самовозбуждения колебаний механических систем")
window.geometry('1600x900')
l2 = l3 = l4 = Label(window)
l2i = l3i = l4i = list()
l1 = Label(text='Введите размерность системы', font=my_font, bg='aquamarine')
e1 = Entry(window, width=10, font=my_font, justify=CENTER, bg='light cyan')
l1.place(relx=0.5, rely=0.1, anchor='center')
e1.place(relx=0.5, rely=0.2, anchor='center')
b1 = Button(text='OK', command=click_button1, font=my_font, bg='aquamarine')
b1.place(relx=0.5, rely=0.3, anchor='center')
window.mainloop()
