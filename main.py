from tkinter import *
from tkinter.messagebox import *
from tkinter import ttk
from PIL import ImageTk, Image
import numpy as np
from numpy import linalg as la
import math
import networkx as nx
import matplotlib.pyplot as plt
from os import remove

n = 1
my_font, my_font2, my_font3 = 'Courier 30', 'Courier 15', 'Courier 7'
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
                if lon[i][j][k] != 0:
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
    work = [[0, 0] for _ in range(n)]
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
            work[j][0] += sum
            work[j][1] += abs(sum)
    return work


def unstable_forms(works: list) -> list:
    res = []
    for i, val in enumerate(works):
        if val[0] > 0:
            res.append(i)
    if len(res) > 2:
        res.remove(works.index(min(works)))
    elif len(res) == 0:
        res.append(works.index(max(works)))
    return res


def the_most_important_cycles_by_form(cycles: list, work_for_cycles: list, form: int) -> list:
    res = [[], []]
    for i, cycle in enumerate(cycles):
        ind = 0
        for j, w in res[1]:
            if work_for_cycles[i][form][1] < w:
                ind += 1
            if work_for_cycles[i][form][1] >= w or ind > 2:
                break
        if ind < 3:
            res = [res[0][:ind] + [cycle] + res[0][ind:2], res[1][:ind] + [work_for_cycles[i][form]] + res[1][ind:2]]
    return res


def clean_the_most_important_connections(the_most_important_connections: list) -> list:
    res = [[], []]
    for i in range(len(the_most_important_connections[0])):
        index = 0
        for j in res[1]:
            if the_most_important_connections[1][i] < j:
                index += 1
            if the_most_important_connections[1][i] >= j or index > 2:
                break
        if index < 3 and the_most_important_connections[1][i] > 0:
            res = [res[0][:index] + [the_most_important_connections[0][i]] + res[0][index:2], res[1][:index] +
                   [the_most_important_connections[1][i]] + res[1][index:2]]
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

    font_for_entry = my_font if n < 4 else my_font2
    l4 = Label(window2, text='Матрицы взаимных связей:', font=font_for_entry, bg='aquamarine')
    l4.grid(row=0, column=0, columnspan=3 * n + 3)
    l5 = Label(window2, text='Матрицы направленных связей:', font=font_for_entry, bg='aquamarine')
    l5.grid(row=n + 1, column=0, columnspan=3 * n + 3)
    l61, l62 = Label(window2, text='C:', font=font_for_entry), Label(window2, text='B:', font=font_for_entry)
    l61.grid(row=1, column=0)
    l62.grid(row=1, column=n + 1)
    l63, l64 = Label(window2, text='M:', font=font_for_entry), Label(window2, text='C:', font=font_for_entry)
    l63.grid(row=1, column=2 * n + 2)
    l64.grid(row=n + 2, column=0)
    l65, l66 = Label(window2, text='B:', font=font_for_entry), Label(window2, text='M:', font=font_for_entry)
    l65.grid(row=n + 2, column=n + 1)
    l66.grid(row=n + 2, column=2*n + 2)

    for i in range(n):
        for j in range(n):
            loe1[i].append(Entry(window2, width=6, font=font_for_entry, justify=CENTER))
            loe2[i].append(Entry(window2, width=6, font=font_for_entry, justify=CENTER))
            loe3[i].append(Entry(window2, width=6, font=font_for_entry, justify=CENTER))
            loe4[i].append(Entry(window2, width=6, font=font_for_entry, justify=CENTER))
            loe5[i].append(Entry(window2, width=6, font=font_for_entry, justify=CENTER))
            loe6[i].append(Entry(window2, width=6, font=font_for_entry, justify=CENTER))
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
    b2 = Button(window2, text='OK', command=lambda: click_button2(loe1, loe2, loe3, loe4, loe5, loe6, window2),
                font=font_for_entry, bg='aquamarine')
    b2.grid(row=2 * n + 2, column=0, columnspan=3 * n + 3)


def click_button2(loe1: list, loe2: list, loe3: list, loe4: list, loe5: list, loe6: list, window2):
    global l2, l3, l4, l2i, l3i, l4i, la4, la5, la61, la62, la63, la64, la65, la66, la71, la72, la73, la74, la75, la76
    global n, labels2
    for i in l2i:
        i.destroy()
    for i in l3i:
        i.destroy()
    for i in l4i:
        for j in i:
            j.destroy()
    for i in [l2, l3, l4, la4, la5, la61, la62, la63, la64, la65, la66]:
        i.destroy()
    for i in [la71, la72, la73, la74, la75, la76]:
        for j in i:
            for k in j:
                k.destroy()
    for i in labels2:
        i.destroy()

    lon = [[[] for __ in range(n)] for _ in range(6)]
    for i in range(n):
        for j in range(n):
            lon[0][i].append(float(loe1[i][j].get()))  # C1
            lon[1][i].append(float(loe2[i][j].get()))  # B1
            lon[2][i].append(float(loe3[i][j].get()))  # M1
            lon[3][i].append(float(loe4[i][j].get()))  # C2
            lon[4][i].append(float(loe5[i][j].get()))  # B2
            lon[5][i].append(float(loe6[i][j].get()))  # M2

    font_for_matrices = my_font if n < 4 else my_font3
    la4 = Label(tab2, text='Матрицы взаимных связей:', font=font_for_matrices, bg='aquamarine')
    la4.grid(row=0, column=0, columnspan=3 * n + 3)
    la5 = Label(tab2, text='Матрицы направленных связей:', font=font_for_matrices, bg='aquamarine')
    la5.grid(row=n + 1, column=0, columnspan=3 * n + 3)
    la61, la62 = Label(tab2, text='C: ', font=font_for_matrices), Label(tab2, text='B: ', font=font_for_matrices)
    la61.grid(row=1, column=0)
    la62.grid(row=1, column=n + 1)
    la63, la64 = Label(tab2, text='M: ', font=font_for_matrices), Label(tab2, text='C: ', font=font_for_matrices)
    la63.grid(row=1, column=2 * n + 2)
    la64.grid(row=n + 2, column=0)
    la65, la66 = Label(tab2, text='B: ', font=font_for_matrices), Label(tab2, text='M: ', font=font_for_matrices)
    la65.grid(row=n + 2, column=n + 1)
    la66.grid(row=n + 2, column=2 * n + 2)
    la71, la72 = [[[] for __ in range(n)] for _ in range(n)], [[[] for __ in range(n)] for _ in range(n)]
    la73, la74 = [[[] for __ in range(n)] for _ in range(n)], [[[] for __ in range(n)] for _ in range(n)]
    la75, la76 = [[[] for __ in range(n)] for _ in range(n)], [[[] for __ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            la71[i][j] = Label(tab2, text=f' {str(lon[0][i][j])} ', font=font_for_matrices)
            la71[i][j].grid(row=i + 1, column=j + 1)
            la72[i][j] = Label(tab2, text=f' {str(lon[1][i][j])} ', font=font_for_matrices)
            la72[i][j].grid(row=i + 1, column=n + j + 2)
            la73[i][j] = Label(tab2, text=f' {str(lon[2][i][j])} ', font=font_for_matrices)
            la73[i][j].grid(row=i + 1, column=2 * n + j + 3)
            la74[i][j] = Label(tab2, text=f' {str(lon[3][i][j])} ', font=font_for_matrices)
            la74[i][j].grid(row=n + i + 2, column=j + 1)
            la75[i][j] = Label(tab2, text=f' {str(lon[4][i][j])} ', font=font_for_matrices)
            la75[i][j].grid(row=n + i + 2, column=n + j + 2)
            la76[i][j] = Label(tab2, text=f' {str(lon[5][i][j])} ', font=font_for_matrices)
            la76[i][j].grid(row=n + i + 2, column=2 * n + j + 3)

    for i in range(n):
        for j in range(n):
            if lon[0][i][j] != lon[0][j][i] or lon[1][i][j] != lon[1][j][i] or lon[2][i][j] != lon[2][j][i]:
                showerror('Ошибка!', 'Матрицы взаимных связей должны быть симметричными!')
                return None
            if i == j and (lon[3][i][j] != 0 or lon[4][i][j] != 0 or lon[5][i][j] != 0):
                showerror('Ошибка!', 'Матрицы направленных связей должны иметь нулевые диагонали!')
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

    l2 = Label(tab1, text='Собственные числа:', font=my_font)
    l2.place(relx=0.5, rely=0.4, anchor='center')
    l2i, l3i, l4i = [], [], []
    for i in range(n):
        l2i.append(Label(tab1, text=str(round(w1[i], 5)), font=my_font))
        l2i[i].place(relx=1.0 / float(n + 1) * (i + 1), rely=0.475, anchor='center')
        l3i.append(Label(tab1, text=str(round(w1[i] / (2 * math.pi), 5)) + ' Гц', font=my_font))
        l3i[i].place(relx=1.0 / float(n + 1) * (i + 1), rely=0.625, anchor='center')
        l4ij = []
        for j in range(n):
            l4ij.append(Label(tab1, text=str(round(v[i][j], 5)), font=my_font))
        l4i.append(l4ij)
        for j in range(n):
            l4i[i][j].place(relx=1.0 / float(n + 1) * (j + 1), rely=0.775 + i * 0.05, anchor='center')
    l3 = Label(tab1, text='Cобственные частоты:', font=my_font)
    l3.place(relx=0.5, rely=0.55, anchor='center')
    l4 = Label(tab1, text='Собственные векторы:', font=my_font)
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
    img = ImageTk.PhotoImage(Image.open('graph.png'))
    panel = Label(tab3, image=img)
    panel.place(relx=0.5, rely=0.5, anchor='center')
    window2.destroy()

    cycles = clean_cycles(cycles)
    work, work_for_cycles = total_work(cycles, lon, v, w1), [[] for _ in cycles]

    font_for_cycles = my_font if len(cycles) < 15 and n < 4 else my_font3
    label11 = Label(tab4, text='цикл', font=font_for_cycles, bg='aquamarine')
    label11.grid(row=0, column=0, columnspan=n * n + n - 1)
    labels2.append(label11)
    for i in range(n):
        label2 = Label(tab4, text=f' работа по \n {i + 1}-й форме ', font=font_for_cycles, bg='aquamarine')
        label2.grid(row=0, column=i + n * n + n)
        labels2.append(label2)
    row, new_row, delta = 1, 0, n * (n + 2)
    for ind, i in enumerate(cycles):
        label3 = Label(tab4, text=str(i.cycle[0] + 1), font=font_for_cycles, bg='aquamarine')
        if row > 26:
            label3.grid(row=new_row, column=delta)
        else:
            label3.grid(row=row, column=0)
        labels2.append(label3)
        column = 1
        for j in i.cycle[1:]:
            label31 = Label(tab4, text=arrows_styles2[j[1] // 3], font=font_for_cycles, fg=arrows_colors[j[1] % 3],
                            bg='aquamarine')
            if row > 26:
                label31.grid(row=new_row, column=delta + column)
            else:
                label31.grid(row=row, column=column)
            labels2.append(label31)
            column += 1
            label31 = Label(tab4, text=str(j[0] + 1), font=font_for_cycles, bg='aquamarine')
            if row > 26:
                label31.grid(row=new_row, column=delta + column)
            else:
                label31.grid(row=row, column=column)
            labels2.append(label31)
            column += 1
        for j in range(n):
            sum, i_ind = 0, i.cycle[0]
            for k in i.cycle[1:]:
                j_ind, arrow_type = k[0], k[1]
                if arrow_type > 2:
                    sum += lon[arrow_type][i_ind][j_ind] * v[i_ind][j] * v[j_ind][j] * (w1[j] / 2 / math.pi) **\
                           (arrow_type - 2)
                i_ind = j_ind
            if sum:
                work_for_cycles[ind].append((round(sum, 2), round(abs(sum) / work[j][1] * 100, 2)))
                label4 = Label(tab4, text=f' {str(round(sum, 2))} - {str(round(abs(sum) / work[j][1] * 100, 2))}% ',
                               font=font_for_cycles, bg='aquamarine')
            else:
                work_for_cycles[ind].append((round(sum, 2), 0.0))
                label4 = Label(tab4, text=f' {str(round(sum, 2))} - 0.0% ', font=font_for_cycles, bg='aquamarine')
            if row > 26:
                label4.grid(row=new_row, column=delta + j + n * n + n)
            else:
                label4.grid(row=row, column=j + n * n + n)
            labels2.append(label4)
        row += 1
        if row > 26:
            new_row += 1
            label12 = Label(tab4, text='цикл', font=font_for_cycles, bg='aquamarine')
            label12.grid(row=0, column=delta, columnspan=n * n + n - 1)
            labels2.append(label12)
            for k in range(n):
                label2 = Label(tab4, text=f' работа по \n {k + 1}-й форме ', font=font_for_cycles, bg='aquamarine')
                label2.grid(row=0, column=k + n * n + n + delta)
                labels2.append(label2)
    label5 = Label(tab4, text='суммарная работа', font=font_for_cycles, bg='aquamarine')
    if new_row:
        label5.grid(row=new_row, column=delta, columnspan=n * n + n - 1)
    else:
        label5.grid(row=row, column=0, columnspan=n * n + n - 1)
    labels2.append(label5)
    for i in range(n):
        label51 = Label(tab4, text=str(round(work[i][0], 5)), font=font_for_cycles, bg='aquamarine')
        if new_row:
            label51.grid(row=new_row, column=delta + n * n + n + i)
        else:
            label51.grid(row=row, column=n * n + n + i)
        labels2.append(label51)

    forms = unstable_forms(work)
    font_for_forms = my_font if len(forms) < 2 else 'Courier 28'
    if len(forms) > 1:
        label6 = Label(tab5, text=f'Потенциально неустойчивыми являются формы {forms[0] + 1} и {forms[1] + 1}',
                       font=font_for_forms, bg='aquamarine')
    else:
        label6 = Label(tab5, text=f'Потенциально неустойчивой является форма {forms[0] + 1}', font=font_for_forms,
                       bg='aquamarine')
    label6.grid(row=0, column=0, columnspan=n * n + n)
    labels2.append(label6)
    row = 1
    for form in forms:
        label61 = Label(tab5, text=f'наиболее значимые циклы по {form + 1}-й форме:', font=font_for_forms,
                        bg='aquamarine')
        label61.grid(row=row, column=0, columnspan=n * n + n)
        labels2.append(label61)
        row += 1
        res = the_most_important_cycles_by_form(cycles, work_for_cycles, form)
        for j in range(len(res[0])):
            label71 = Label(tab5, text=str(res[0][j].cycle[0] + 1), font=font_for_forms, bg='aquamarine')
            label71.grid(row=row, column=0)
            labels2.append(label71)
            column = 1
            for k in res[0][j].cycle[1:]:
                label72 = Label(tab5, text=arrows_styles2[k[1] // 3], font=font_for_forms, fg=arrows_colors[k[1] % 3],
                                bg='aquamarine')
                label72.grid(row=row, column=column)
                labels2.append(label72)
                label73 = Label(tab5, text=str(k[0] + 1), font=font_for_forms, bg='aquamarine')
                label73.grid(row=row, column=column + 1)
                labels2.append(label73)
                column += 2
            label74 = Label(tab5, text=f'{res[1][j][0]} - {res[1][j][1]}%', font=font_for_forms, bg='aquamarine')
            label74.grid(row=row, column=n * n + n - 1)
            labels2.append(label74)
            row += 1
    for form in forms:
        the_most_important_connections, sum = [[], []], 0
        for i in range(3, 6):
            for j in range(n):
                for k in range(n):
                    the_most_important_connections[0].append((j, k, i))
                    the_most_important_connections[1].append(lon[i][j][k] * v[j][form] * v[k][form] *
                                                             (w1[form] / 2 / math.pi) ** (i - 2))
                    sum += abs(lon[i][j][k] * v[j][form] * v[k][form] * (w1[form] / 2 / math.pi) ** (i - 2))
        the_most_important_connections[1] =\
            list(map(lambda el: abs(el) / abs(sum),
                     the_most_important_connections[1])) if sum != 0 else [0] * len(the_most_important_connections[1])
        the_most_important_connections = clean_the_most_important_connections(the_most_important_connections)
        label8 = Label(tab5, text=f'наиболее значимые связи по {form + 1}-й форме:', font=font_for_forms,
                       bg='aquamarine')
        label8.grid(row=row, column=0, columnspan=n * n + n)
        labels2.append(label8)
        row += 1
        if the_most_important_connections[0]:
            for i in range(len(the_most_important_connections[0])):
                label81 = Label(tab5, text=str(the_most_important_connections[0][i][0] + 1), font=font_for_forms,
                                bg='aquamarine')
                label81.grid(row=row, column=0)
                labels2.append(label81)
                label82 = Label(tab5, text='-->', font=font_for_forms, bg='aquamarine',
                                fg=arrows_colors[the_most_important_connections[0][i][2] % 3])
                label82.grid(row=row, column=1)
                labels2.append(label82)
                label83 = Label(tab5, text=str(the_most_important_connections[0][i][1] + 1), font=font_for_forms,
                                bg='aquamarine')
                label83.grid(row=row, column=2)
                labels2.append(label83)
                label84 = Label(tab5, text=f'{round(the_most_important_connections[1][i] * 100, 2)}%',
                                font=font_for_forms, bg='aquamarine')
                label84.grid(row=row, column=3, columnspan=n * n + n - 3)
                labels2.append(label84)
                row += 1
    window.mainloop()


try:
    remove('graph.png')
except:
    pass
window = Tk()
window['bg'] = 'aquamarine'
window.state('zoomed')
window.title('Поиск наиболее значимых причин самовозбуждения колебаний механических систем')
window.geometry('1600x900')
tab_control = ttk.Notebook(window)
s = ttk.Style()
s.theme_create('MyStyle', parent="alt", settings={'TNotebook.Tab': {'configure': {'font': my_font2,
                                                                                  'background': 'light cyan'}}})
s.theme_use('MyStyle')
tab1, tab2 = Frame(tab_control, background='aquamarine'), Frame(tab_control, background='aquamarine')
tab3, tab4 = Frame(tab_control, background='aquamarine'), Frame(tab_control, background='aquamarine')
tab5 = Frame(tab_control, background='aquamarine')
tab_control.add(tab1, text='Размерность системы, собственные формы ')
tab_control.add(tab2, text='Введённые матрицы ')
tab_control.add(tab3, text='Граф ')
tab_control.add(tab4, text='Работа циклов ')
tab_control.add(tab5, text='Потенциально неуст. формы, наиболее значимые связи ')
l2 = l3 = l4 = Label(tab1)
la4 = la5 = la61 = la62 = la63 = la64 = la65 = la66 = Label(tab2)
l2i = l3i = l4i = labels2 = list()
la71 = la72 = la73 = la74 = la75 = la76 = [[] for _ in range(n)]
l1 = Label(tab1, text='Введите размерность системы', font=my_font, bg='aquamarine')
e1 = Entry(tab1, width=10, font=my_font, justify=CENTER, bg='light cyan')
l1.place(relx=0.5, rely=0.1, anchor='center')
e1.place(relx=0.5, rely=0.2, anchor='center')
b1 = Button(tab1, text='OK', command=click_button1, font=my_font, bg='aquamarine')
b1.place(relx=0.5, rely=0.3, anchor='center')
tab_control.pack(expand=1, fill='both')
window.mainloop()
