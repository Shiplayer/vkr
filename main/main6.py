# coding: utf-8

# In[1]:


# Необходимые далее системные библиотеки
import time

# Библиотека для вывода в окно и чтения с устройств ввода
from PyQt5 import QtOpenGL, QtWidgets, QtCore

# Библиотека для отрисовки 3D графики (прослойка над OpenGL)
import moderngl as mgl

# import autograd.numpy as np
# import autograd as ag

import numpy as np
import scipy.optimize

import matplotlib.pyplot as plt


# In[2]:


# Этот блок взят из примеров moderngl, можно смело его пропускать.

# Обьекты этого класса хранят состояние окна, нажатые клавиши и т.п.
class WindowInfo:
    def __init__(self):
        self.size = (0, 0)  # Размер окна в пикселях
        self.mouse = (0, 0)
        self.wheel = 0
        self.time = 0
        self.ratio = 1.0  # Отношиние ширины и длины окна
        self.viewport = (0, 0, 0, 0)  # Координаты и размеры окна
        self.keys = np.full(256, False)  # Для каждого кода клавиши True, если клавиша нажата.
        self.old_keys = np.copy(self.keys)
        self.delta = 0.0

    def key_down(self, key):
        return self.keys[key]

    def key_pressed(self, key):
        return self.keys[key] and not self.old_keys[key]

    def key_released(self, key):
        return not self.keys[key] and self.old_keys[key]


# Класс окна, абстрагирующий большинство деталей взаимодействия с оконным менеджером.
class ExampleWindow(QtOpenGL.QGLWidget):
    def __init__(self, size, title):
        fmt = QtOpenGL.QGLFormat()
        fmt.setVersion(3, 3)  # Минимальная версия OpenGL. Берем 3.3 так как ничего сложного не делаем.
        fmt.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        fmt.setSwapInterval(1)  # Синхронизировать смену кадра с разверткой монитора
        fmt.setSampleBuffers(True)  # Использовать мальтисемплинг для сглаживания границ обьектов.
        fmt.setDepthBufferSize(24)  # Не важно, буфеп глубины для двумерной графики вообще не нужен

        super(ExampleWindow, self).__init__(fmt, None)
        self.resize(size[0], size[1])
        self.move(QtWidgets.QDesktopWidget().rect().center() - self.rect().center())
        self.setWindowTitle(title)

        self.start_time = time.clock()
        self.example = lambda: None
        self.ex = None

        self.wnd = WindowInfo()

        self.time_mark = time.time()

    def resizeEvent(self, event):  # Устанавливаем размер окна
        size = (event.size().width(), event.size().height())
        self.wnd.size = size
        self.wnd.viewport = (0, 0) + (size[0], size[1])
        self.wnd.ratio = size[0] / size[1] * self.devicePixelRatio()  # Соотношение видимой высота и ширины окна

    def keyPressEvent(self, event):  # Здесь обновляется список нажатых клавиш
        # Quit when ESC is pressed
        if event.key() == QtCore.Qt.Key_Escape:
            QtCore.QCoreApplication.instance().quit()

        self.wnd.keys[event.nativeVirtualKey() & 0xFF] = True

    def keyReleaseEvent(self, event):
        self.wnd.keys[event.nativeVirtualKey() & 0xFF] = False

    def mouseMoveEvent(self, event):
        self.wnd.mouse = (event.x(), event.y())

    def wheelEvent(self, event):
        self.wnd.wheel += event.angleDelta().y()

    def paintGL(self):  # Эта функция вызывается для отрисовки содержимого окна
        if self.ex is None:
            self.ex = self.example()

        # Находим время, прошедшее с прошлого кадра.
        mark = time.time()
        self.wnd.delta = mark - self.time_mark
        self.time_mark = mark
        self.setWindowTitle("FPS: {:.1f} {}".format(1.0 / self.wnd.delta, self.ex.info))

        self.wnd.time = time.clock() - self.start_time
        self.ex.render()
        self.wnd.old_keys = np.copy(self.wnd.keys)
        self.wnd.wheel = 0
        self.update()


def run_example(example):  # Эта функция инициализирует Qt приложение и создает окно для 3D графики.
    app = QtWidgets.QApplication([])
    widget = ExampleWindow(example.WINDOW_SIZE, getattr(example, 'WINDOW_TITLE', example.__name__))
    example.wnd = widget.wnd
    widget.example = example
    widget.show()
    app.exec_()
    del app


class Example:
    WINDOW_SIZE = (800, 800)
    wnd = None  # type: WindowInfo

    def render(self):
        pass


# Вспомним немного механику твердого тела. 
# Рассматриваемая механическая система состоит из совокупности твердых недеформируемых тел.
# Каждое из тел имеет систему координат, связанную с самим телом, в этой системе координат
# мы храним координаты вершин треугольников, которые далее будут использоваться для отрисовки графики.
# Центр тяжести в системе координат тела находится в нуле (начале отсчета).
# Также у нас имеется мировая система координат - это инерциальная система координат, которая однозначно связана
# с оконными координатами.
# Переход из локальных координат ${}^Br\in\mathbb R^2$ в мировые ${}^Wr\in\mathbb R^2$ осуществляется с помощью изометрии
# $${}^Wr=R\,{}^Br+L,$$
# где $L\in\mathbb R^2$ - положение центра масс, $U$ - матрица вращения:
# $$
# R=\begin{pmatrix}\cos\phi & \sin\phi\\ -\sin\phi & \cos\phi\end{pmatrix},
# $$
# где $\phi$ - угол поворота.
# Удобно однако это преобразование записывать в виде одного умножения на матрицу,
# для чего вектор координат дополняют единицей и рассматривают расширенную матрицу изометрии
# $${}^WS=U\,{}^BS,\quad 
# {}^{W/B}S=\begin{pmatrix}{}^{W/B}r \\ 1\end{pmatrix}\in\mathbb R^3,\quad
# U=\begin{pmatrix}R & L \\ 0 & 1\end{pmatrix}.
# $$

# In[3]:


# Наиболее полезные трансформации пространства:
# параллельный перенос центра координат в точку (x,y)
def translate(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=np.float32)


# поворот на угол a в радианах.
def rotate(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float32)


# А это растяжение в x раз по оси Ox и y раз по оси Oy.
# Это не изометрия, поэтоиу не должна использоваться с для преобразования
# координат твердых тел, однако мы используем его для визуализации обьектов.
def stretch(x, y):
    return np.array([[x, 0, 0], [0, y, 0], [0, 0, 1]], dtype=np.float32)


# На практике мы часто не знаем преобразование $U$ явно, однако знаем, что чтобы получить
# текущее положение тела, то его нужно сдвинуть в определенном направлении, повернуть на такой-то угол,
# потом еще раз сдвинуть и т.п. 
# Т.о. преобразование $U$ разложено на множители, каждый из которых задает элементарное преобразование и зависит от одного параметра:
# $$U=\ldots U_2(q_2)U_1(q_1)U_0(q_0).$$
# Порядок перемножения важен, первым применяется крайнее правое преобразование $U_0$.
# Преобразования $U_k$ не обязательно различны, их число может быть произвольным, параметры $q_k$
# могут идти в произвольном порядке и повторяться. 
# Набор всех $q_k$ который однозначно (желательно взаимнооднозначно) определяет конфигурацию системы (положение всех твердых тел в системе) называется обобщенными координатами системы.
# Описание динамики удобно производить в обобщенных координатах, так как таких координат меньше, чем декартовых, а также на обобщенные координаты не накладываются ограничения (например, точка на окружности задается одним углом, а не ограничением вида $x^2+y^2=1$).
# 
# Для расчета динамики тела, нам далее потребуется дифференциировать матрицу $U$ по обобщенным координатам.
# Это можно сделать с помощью пакета autograd, однако так мы хорошо знаем структуру матрицы $U$, мы можем
# вычислить производные быстрее, чем это делает autograd.

# In[4]:


# Класс для вычисления матриц преобразования U одновременно с производными по q[k].
class AD(object):
    def __init__(self, value):  # Конструктор для постоянной матрицы, не зависящей от обобщенных координат.
        self._value = value  # Сама матрица преобразования.
        self._difs = {}  # Все производные равны нулю, не указываем их, так как ноль значение по умолчанию.
        self._hess = {}

    def value(self):  # Читает матрицу пребразования.
        return self._value

    def grad(self, N):  # Возвращает G, такое что G[:,:,k] есть производная self.value() по q[k].
        # Склеивает элементы словаря self._difs в один вектор, заполняя нулями отсутствующие.
        a = np.array([self._difs[n] if n in self._difs else np.zeros((3, 3)) for n in range(N)])
        return np.transpose(a, (1, 2, 0))

    def hess(self, v):
        N = v.shape[0]
        assert (v.shape == (N,))
        a = []
        for n in range(N):
            b = np.zeros((3, 3), dtype=np.float32)
            for m in range(N):
                x, y = (n, m) if n < m else (m, n)
                if (x, y) in self._hess:
                    b = b + self._hess[x, y] * v[m]
            a.append(b)
        return np.array(a).transpose((1, 2, 0))

    def hessian(self, N):
        a = np.zeros((3, 3, N, N), dtype=np.float32)
        for n in range(N):
            for m in range(N):
                x, y = (n, m) if n < m else (m, n)
                if (x, y) in self._hess:
                    a[:, :, n, m] += self._hess[x, y]
        return a

    @staticmethod
    def identity():  # Конструктор для тождественного преобразования.
        return AD(np.eye(3))

    @staticmethod
    def translate_x(q, n):  # Сдвиг по оси Ox на значение обобщенной координаты q[n]
        ad = AD(np.array([[1, 0, q[n]], [0, 1, 0], [0, 0, 1]]))
        ad._difs[n] = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])  # Производная матрицы в предыдущей строке по q[n].
        return ad

    @staticmethod
    def rotate(q, n):  # Поворот вокруг оси Oz на значение угла из обобщенной координаты q[n].
        c, s = np.cos(q[n]), np.sin(q[n])
        ad = AD(np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]))
        ad._difs[n] = np.array([[-s, c, 0], [-c, -s, 0], [0, 0, 0]])
        ad._hess[n, n] = np.array([[-c, -s, 0], [s, -c, 0], [0, 0, 0]])
        return ad

    def __mul__(self, o):  # Композиция преобразований (произведение матриц).
        ad = AD(self._value @ o._value)  # Сами преобразование перемножаются как матриц.

        for n in self._difs.keys():  # Производные считаются по правилу дифференциирования произведения.
            ad._difs[n] = self._difs[n] @ o._value  # Сначала производная первого на второе,
        for n in o._difs.keys():
            t = self._value @ o._difs[n]  # затем первое на производную второго,
            ad._difs[n] = ad._difs[n] + t if n in ad._difs else t  # и складываем.

        for n in self._hess.keys():
            ad._hess[n] = self._hess[n] @ o._value
        for n in o._hess.keys():
            t = self._value @ o._hess[n]
            ad._hess[n] = ad._hess[n] + t if n in ad._hess else t
        for n in self._difs.keys():
            for m in o._difs.keys():
                t = self._difs[n] @ o._difs[m]
                if n < m:
                    ad._hess[n, m] = ad._hess[n, m] + t if (n, m) in ad._hess else t
                elif n > m:
                    ad._hess[m, n] = ad._hess[m, n] + t if (m, n) in ad._hess else t
                else:
                    ad._hess[n, n] = ad._hess[n, n] + 2 * t if (n, n) in ad._hess else 2 * t

        # print("First", self)
        # print("Second", o)
        # print("Result", ad)

        return ad

    def __str__(self):  # Вывод преобразования и производных для отладки.
        return "\nAD >>> \n{}\n{}\n{}\n<<< AD\n".format(self._value, self._difs, self._hess)

        # import numpy.testing as tst


# import numpy.random as rnd
# def test_ad(oper, dim=2):
#     print("test_ad", dim)
#     q = rnd.randn(dim)
#     v = rnd.randn(dim)
#     a = oper(q)
#     da = np.sum(a.grad(dim)*v[None,None,:],axis=2)
#     dda = a.hessian(dim)
#     vda = a.hess(v)
#     b = ag.make_jvp(lambda q: oper(q).value())(q)(v)
#     tst.assert_almost_equal(a.value(), b[0])
#     tst.assert_almost_equal(da, b[1])
#     g = ag.jacobian(lambda q: oper(q).value())(q)
#     tst.assert_almost_equal(a.grad(dim), g)
#     h = ag.hessian(lambda q: oper(q).value())(q)
#     # for n in range(dim):
#     #     for m in range(dim):
#     #         print(np.hstack((dda[:,:,n,m], h[:,:,n,m])))
#     tst.assert_almost_equal(dda, h)
#     h2 = ag.jacobian(lambda q: oper(q).grad(dim))(q)
#     tst.assert_almost_equal(dda, h2)    
#     c = ag.make_jvp(lambda q: oper(q).grad(dim))(q)(v)
#     hv = np.sum(h*v[None,None,None],axis=3)
#     tst.assert_almost_equal(a.grad(dim), c[0])
#     tst.assert_almost_equal(hv, c[1])
#     # for n in range(dim):
#     #     print(np.hstack((vda[:,:,n], hv[:,:,n])))
#     tst.assert_almost_equal(vda, hv)
#     tst.assert_almost_equal(vda, c[1])

# test_ad(lambda q: AD.translate_x(q, 0), dim=1)
# test_ad(lambda q: AD.rotate(q, 0), dim=1)
# test_ad(lambda q: AD.translate_x(q, 0)*AD.translate_x(q, 0), dim=1)
# test_ad(lambda q: AD.rotate(q, 0)*AD.rotate(q, 0), dim=1)
# test_ad(lambda q: AD.rotate(q, 0)*AD.translate_x(q, 0), dim=1)
# test_ad(lambda q: AD.translate_x(q, 0)*AD.rotate(q, 0), dim=1)
# test_ad(lambda q: AD.translate_x(q, 0)*AD.translate_x(q, 1), dim=2)
# test_ad(lambda q: AD.rotate(q, 0)*AD.rotate(q, 1), dim=2)
# test_ad(lambda q: AD.rotate(q, 0)*AD.translate_x(q, 1), dim=2)
# test_ad(lambda q: AD.translate_x(q, 0)*AD.rotate(q, 1), dim=2)
# test_ad(lambda q: AD.translate_x(q, 1)*AD.translate_x(q, 0), dim=2)
# test_ad(lambda q: AD.rotate(q, 1)*AD.rotate(q, 0), dim=2)
# test_ad(lambda q: AD.rotate(q, 1)*AD.translate_x(q, 0), dim=2)
# test_ad(lambda q: AD.translate_x(q, 1)*AD.rotate(q, 0), dim=2)
# exit(1)

# In[5]:


# Мы рассматриваем робота в виде "руки", т.е. у нас есть некий корневой блок, 
# к которому крепяться другие, к которым крепяться третьи и т.д.
# Для задания структуры такого робота мы определяемм класс Compound.
class Compound(object):
    # Конструктор обьекта.
    # Обьект характеризуется полной массой mass и моментом инерции (вокруг оси параллельной Oz 
    # и проходящей через центра масс) moi/
    # Для отрисовки будет использоваться обьект figure (определен дальше)
    # цвета color, к которому перед отрисовкой будет применено преобразование modifier,
    # которое используется только для отрисовки и может быть не ортогональным.
    # Программа никак не проверяет корректность задания moi и положения центра масс.
    def __init__(self, figure=None, color=(0, 0, 0, 1), mass=None, moi=None, modifier=np.eye(3)):
        self._figure = figure
        self._color = tuple(color)
        self._modifier = modifier

        self.mass = mass
        self.moi = moi

        # Обьекты образуют иерархию. Обьекты создаются свободными.
        self._parent = None  # Это к чему присоединен текущий обьект.
        self._children = {}  # Это присоединенные к текущему обьекты.

    # Этот метод прикрепляет к текущему обьекту self обьект child.
    # Прикрепленный обьект child можно затем получить по его имени name. 
    def adopt(self, name, child):
        assert (child._parent is None)
        child._parent = self
        self._children[name] = child

    # Возвращает список всех детей обьекта.
    def children(self):
        return self._children.values()

    # Возвращает child, соответствующий имени name, как было передано в adopt.
    def __getitem__(self, name):
        return self._children[name]

    # Задает преобразование, которое нужно сделать, чтобы перейти от системы 
    # координат текущего обьекта self в систему координат его предка self.parent
    # для данных обобщенных координат q.
    # Для обьекта с self.parrent==None, это переход из системы координат тела
    # в мировую систему координат.
    def propagator(self, _q):
        return AD.identity()

    # Этот метод далее используется для отрисовки обьекта на экране.
    # Метод сам вычисляет переход из систем координат тела в мировую 
    # для данных обобщенных координат,
    # затем применяет преобразование transform, которое задает положение
    # окна (камеры) в мировой системе координат.
    # Вектор scale задает масштаб по горизонтали и вертикали на экране,
    # что далее используется для компенсации различных размеров окна по ширине и высоте,
    # отклонения пикселей от квадратной формы.
    def render(self, q, transform=np.eye(3), scale=(1.0, 1.0)):
        u = transform @ self.propagator(q).value()
        # assert(isinstance(u, AD))
        if not self._figure is None:
            self._figure.render(color=self._color, transform=u @ self._modifier, scale=scale)
        for child in self.children():
            child.render(q, transform=u, scale=scale)


# In[6]:


# Этот класс содержит форму двухмерного обьекта, используемую для отрисовки обьектов на экран.
class Figure(object):
    # Обьект должен сам уметь себя отрисовывать, однако он принимает некоторые модификаторы,
    # которые изменяют его цвет color (в формате RGBA), размеры и положение с помощью transform.
    # Параметр scale задает масштаб по ширине и высоте окна.
    def render(self, color=[0.0, 0.0, 0.0, 1.0], transform=np.eye(3), scale=(1.0, 1.0)):
        raise NotImplementedError


# Класс для фигуры, состоящей из треугольников.
class Solid(Figure):
    # Конструктор сохраняет OpenGL контекст ctx, который генерирует moderngl.
    # Также конструктор сохраняет массив вершин vertices и способ соединения вершин в треугольники indices.
    def __init__(self, ctx=None, vertices=[], indices=[]):
        self.ctx = ctx

        # Это программа на GLSL, которая собственно и выводит треугольники с нужным цветами
        # и в нужных положениях.
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330


                uniform mat3 u_transform;
                uniform vec2 u_scale;

                in vec2 in_vert;

                void main() {
                    vec3 pos = vec3(in_vert, 1.0)*u_transform;
                    gl_Position = vec4(pos.xy*u_scale, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330

                uniform vec4 u_color;

                out vec4 f_color;

                void main() {
                    f_color = u_color;
                }
            ''',
        )

        self.scale = self.prog['u_scale']  # Вектор с масштабами по обоим на экране.
        self.color = self.prog['u_color']  # Цвет для примитива в формате RGBA.
        self.transform = self.prog['u_transform']  # Матрица замены координат на плоскости (аналогично U).

        # Массивы для хранения вершин и треугольников на графической карте.
        self.vbo = self.ctx.buffer(np.asarray(vertices, dtype=np.float32).tobytes())
        self.ibo = self.ctx.buffer(np.asarray(indices, dtype=np.int32).tobytes())

        vao_content = [
            (self.vbo, '2f', 'in_vert')
        ]

        self.vao = self.ctx.vertex_array(self.prog, vao_content, self.ibo)

    # Метод для отрисовки обьекта.
    def render(self, color=(0.0, 0.0, 0.0, 1.0), transform=np.eye(3), scale=(1.0, 1.0)):
        # Устанавливаем uniform в программе OpenGL
        self.scale.value = scale
        self.color.value = color
        self.transform.value = tuple(transform.flatten())
        # Запускаем саму отрисовку.
        self.vao.render()


# Квадрат размера 2 на 2
class Rectangle(Solid):
    def __init__(self, ctx):
        vertices = [
            -1.0, -1.0,
            +1.0, -1.0,
            +1.0, +1.0,
            -1.0, +1.0,
        ]
        indices = [1, 2, 0, 2, 0, 3]
        super().__init__(ctx=ctx, vertices=vertices, indices=indices)


# Окружность единичного радиуса, приближенна вписанным n-угольником.
class Circle(Solid):
    def __init__(self, ctx, n=12):
        vertices = []
        indices = []
        for k in range(n):
            vertices.append(np.cos(2 * np.pi * k / n))
            vertices.append(np.sin(2 * np.pi * k / n))
            indices.append(n)
            indices.append(k)
            indices.append((k + 1) % n)
        vertices.append(0.0)
        vertices.append(0.0)
        super().__init__(ctx=ctx, vertices=vertices, indices=indices)


# Кинетическая энергия объекта с массой колес $m$ и линейной скоростью для колес и тела равно:
# $$T=\frac{m_W}{2}({v_L}^2 + {v_R}^2) + \frac{M}{2}{v_M}^2$$
# где m_W - масса колес, v_L, v_R, v_M - линейная скорость центра масс для колес и для тела
# Запишем кинетическую инрегию вращения для нашего тела с колесами:
# $$T_{rot} = \frac{1}{2}(\omega^L)^T I_L \omega^L + \frac{1}{2}(\omega^R)^T I_R \omega^R + \frac{1}{2}(\omega^B)^T I_B \omega^B$$

# Для нашей системы кинетическая $T$ и потенциальная энергия $V$ суть суммы
# кинетической и потенциальной энергий для отдельных обьектов.
# Кинетическая энергия обьекта массы $M$ с моментои инерции $I$ равна:
# $$T=\frac{M}{2}v^2+\frac{I}{2}\omega^2,$$
# где $v$ - линейная скорость центра масс, $\omega$ - угловая скорость вращения 
# вокруг оси Oz.
# Потенциальная энергия у нас отвечает только силе тяжести:
# $$U=Mgy,$$
# где $y$ - высота обьекта (вторая координата в мировой системе координат).
# Для каждого обьекта мы знаем зависимость от обобщенных координат $q$ 
# его матрицы перехода $U$ от координат тела к мировым координатам.
# Для данных $q$ это сразу дает нам $y$ и потенциальную энергию.
# 
# Кинетическую энергию можно выразить через производную $\dot U$
# матрицы перехода $U$ по времени $t$.
# Согласно введенным ранее обозначениям
# $$\dot U=\begin{pmatrix}\dot R & \dot L\\0 & 0\end{pmatrix}.$$
# Скорость $\dot L$ изменения положения $L$ центра масс очевидно равна 
# линейной скорости тела $v$ в мировой системе координат.
# Производная $\dot R$ матрицы вращения равна:
# $$\dot R=\begin{pmatrix}
# -\sin\phi & \cos\phi \\ -\cos\phi & -\sin\phi
# \end{pmatrix}\dot\phi,$$
# где $\dot\phi=\omega$ - угловая скорость по определению.
# Умножая производную матрицы вращения на транспонированную, можно избавиться от углов:
# $$\dot R^T\dot R=\omega^2\begin{pmatrix}
# 1 & 0 \\ 0 & 1
# \end{pmatrix}.$$
# Введем матрицу
# $$
# \mathbf{M}=\begin{pmatrix}
# \frac{I}{2} & 0 & 0 \\ 0 & \frac{I}{2} & 0 \\ 0 & 0 & M
# \end{pmatrix}.
# $$
# Тогда кинетическая энергия выражается в виде:
# $$
# T=\frac{1}{2}\mathrm{Tr}\,\dot U^T \mathbf{M} \dot U
# =\frac{1}{2}\mathrm{Tr}\,\begin{pmatrix}
# \frac{I}{2}\dot R^T\dot R & \frac{I}{2}\dot R^T\dot L \\
# \frac{I}{2}\dot L^T\dot R & \dot L^2
# \end{pmatrix}
# =\frac{1}{2}\mathrm{Tr}\,\begin{pmatrix}
# \frac{I}{2}\omega^2 & 0 & ? \\
# 0 & \frac{I}{2}\omega^2 & ? \\
# ? & ? & Mv^2
# \end{pmatrix}.
# $$

# Состояние системы описывается с помощью обобщенных координат $q$ и их производных $\dot q$ по времени $t$.
# Далее в коде мы будем использовать обозначение $\dot q=v$, не путать с линейной скоростью $\dot L$.
# Динамика системы описывается уравнением Эйлера-Лагранжа:
# $$\frac{\partial L}{\partial q}-\frac{d}{dt}\frac{\partial L}{\partial \dot q}=Q,$$
# где $L$ - функция Лагранжа:
# $$L(q,v)=T(q,v)-V(q),$$
# а $Q$ - обобщенные силы.
# Частные производные здесь обозначают градиенты по обобщенным координатами в первом слагаемом,
# и обобщенным скоростям во втором.
# Чтобы решить уравнение Э-Л, нам нужно явно выразить производные состояния по времения через
# само состояние. 
# Для координат очевидно:
# $$\frac{d}{dt}q=\dot q,$$
# однако производные скоростей входят в уравнение Э-Л неявно и его придется решить.
# 
# Начнем со вкалада потенциальной энергии $V$.
# Так как потенциальные силы не зависят от скорости, то производные по $\dot q$ равны нулю.
# Ранее мы выразили потенциальную энергию через матрицу $U(q)$, 
# следовательно производные по $q$ можно найти по правилу дифференциирования сложной функции:
# $$\frac{\partial V}{\partial q}=\frac{\partial V}{\partial U}\cdot \frac{\partial U}{\partial q}.$$
# Здесь нам пригодиться класс AD, который и вычисляет градиент $U$ по $q$.
# 
# Выше мы показали, что кинетическая энергия выражается через производную матрицы $U$ по времения,
# теперь же мы отделим зависимость $U$ от $q$ и обобщенные скорости $\dot q$:
# $$T=\frac{!}{2}\mathrm{Tr}\, \dot U^T\mathbf{M}\dot U
# =\frac{!}{2}\mathrm{Tr}\, \dot q^T\cdot \frac{\partial U^T}{\partial q}\mathbf{M}\frac{\partial U}{\partial q} \cdot \dot q.$$
# Заметим, что $\frac{\partial U}{\partial q}$ является тензором с тремя размерностями: два первых аналогичны индексам матрицы $U$, а последний обозначает координату $q_k$, по которой осуществлялось дифференциирование.
# Операция $\cdot$ здесь обозначает скалярное произведение по последнему индексу, а при перемножении матриц суммирование происходит по двум первым индексам, по стандартному правилу умножения матриц.
# Соответственно $\frac{\partial U^T}{\partial q}\mathbf{M}\frac{\partial U}{\partial q}$ тензор с четырмя индексами, два последних отвечают производным по $q$.
# Так как след береться только по первым индексам, то можно переставить след и скалярное произведение "$\cdot$":
# $$T=\frac{1}{2}\dot q^T\mathcal{M}\dot q,$$
# где мы обозначили через $\mathcal M$ матрицу масс:
# $$\mathcal{M}(q)=\mathrm{Tr}\,\bigg(\frac{\partial U^T}{\partial q}\mathbf{M}\frac{\partial U}{\partial q}\bigg).$$
# Отсюда легко видеть, что 
# $$\frac{\partial T}{\partial \dot q}=\mathcal{M}\dot q,$$
# так как матрица масс $\mathcal M$ - симметрична.

# Таким образом уравнение Э-Л принимает вид:
# $$\frac{1}{2}\dot q^T\frac{\partial \mathcal M}{\partial q}\dot q-\frac{\partial V}{\partial q}\dot q
# -\frac{d}{dt}\big[\mathcal{M}\dot q\big]=Q.$$
# Теперь раскроем производную по времени:
# $$\frac{1}{2}\dot q^T\frac{\partial \mathcal M}{\partial q}\dot q-\frac{\partial V}{\partial q}\dot q
# -\dot q^T\frac{\partial \mathcal{M}}{\partial q}\dot q-\mathcal{M}\ddot q=Q.$$
# Приводя подобные, перенося все кроме ускорения в правую часть, получаем второе уравнение движения:
# $$\ddot q=\frac{d}{dt}\dot q=-\mathcal{M}^{-1}\bigg[
# Q+\frac{1}{2}\dot q^T\frac{\partial \mathcal M}{\partial q}\dot q+\frac{\partial V}{\partial q}\dot q
# \bigg].$$
# У нас остался не выраженным явно градиент матрицы масс по обобщенным координатам,
# который может быть достаточно громоздким.
# Мы будем вычислять этот градиент автоматически с помощью пакета autograd. 
# 
# Интегрирование движения проводим методом Рунге-Кутты.
# Лучше было бы использовать симплектический интегратор, однако в данном случае это не очень просто,
# так как матрица масс $\mathcal{M}$ (а значит и кинетическая энергия $T$) зависит от координат $q$.

# In[7]:

# Обходчик по всему дереву обьектов.
# Принимает на вход обобщенные координаты q, обобщенные скорости v
# и обьект obj класса Сompound, который является корнем дерева обьектов.
# Возвращает кортеж (
#   градиент потенциальной энергии \frac{\partial V}{\partial q},
#   матрица масс \mathcal{M},
#   градиент матрицы масс \frac{\partial\mathcal{M}}{\partial q}
# )
# Результат получается сложение отдельных компонент для всех обьектов из дерева обьектов.
def walker(obj: Compound, q, v, parent_U=AD.identity()):
    # Матрица масс = dL/dv.
    U = parent_U * obj.propagator(q)
    dU = U.grad(len(q))  # градиент матрицы U по обобщенным координатам
    ddU = U.hess(v)
    I = np.array([obj.moi / 2.0, obj.moi / 2.0, obj.mass])[None, :, None, None]  # Матрица \mathrm{M}
    M = np.sum(dU[:, :, None, :] * dU[:, :, :, None] * I, axis=(0, 1))  # матрица масс \mathcal{M}
    dM = np.sum((ddU[:, :, None, :] * dU[:, :, :, None] + dU[:, :, None, :] * ddU[:, :, :, None]) * I,
                axis=(0, 1))  # матрица масс \mathcal{M}
    d_potential_energy = obj.mass * dU[1, -1] * gravity  # градиент потенциальной энергии V

    for child in obj.children():
        dV, M1, dM1 = walker(child, q, v, parent_U=U)
        d_potential_energy = d_potential_energy + dV
        M = M + M1
        dM = dM + dM1
    return d_potential_energy, M, dM


def energy(obj: Compound, q, v, parent_U=AD.identity()):
    U = parent_U * obj.propagator(q)
    dU = U.grad(len(q))
    I = np.array([obj.moi / 2.0, obj.moi / 2.0, obj.mass])[None, :, None, None]  # Матрица \mathrm{M}
    M = np.sum(dU[:, :, None, :] * dU[:, :, :, None] * I, axis=(0, 1))  # матрица масс \mathcal{M}
    kinetic_energy = np.sum(M * v[None, :] * v[:, None]) / 2.0  # кинетическая энергия T
    potential_energy = obj.mass * U.value()[1, -1] * gravity  # потенциальная энергия V
    e = kinetic_energy + potential_energy  # полная энергия H=T+V

    for child in obj.children():
        e = e + energy(child, q, v, parent_U=U)

    return e


# Правая часть уравнения динамики:
# d q/d t = rhs(obj,q,v,force)[0]
# d v/d t = rhs(obj,q,v,force)[q]
# Аргументы: 
#   q - обобщенные координаты q
#   v - обобщенные скорости \dot q
#   force - обобщенные силы Q
#   obj - корень дерева обьектов
def rhs(obj, q, v, force):
    dV, M, dM_v = walker(obj, q, v)
    # Считаем правую часть второго уравнения динамики без умножения на обратную к матрице масс
    # \mathcal{M}\ddot q:
    Ma = force + dV + np.sum(v[:, None] * dM_v, axis=0) / 2.0
    # Умножая на обратную к \mathcal{M} получаем ускорение.
    a = - np.linalg.solve(M, Ma)
    return v, a


# Функция делает один шаг метода Рунге-Кутты четвертого порядка.
def runge(dt, obj, q, v, force):
    dq1, dv1 = rhs(obj, q, v, force)
    dq2, dv2 = rhs(obj, q + dt / 2.0 * dq1, v + dt / 2.0 * dv1, force)
    dq3, dv3 = rhs(obj, q + dt / 2.0 * dq2, v + dt / 2.0 * dv2, force)
    dq4, dv4 = rhs(obj, q + dt * dq3, v + dt * dv3, force)
    q = q + (dq1 + 2 * (dq2 + dq3) + dq4) * (dt / 6.0)
    v = v + (dv1 + 2 * (dv2 + dv3) + dv4) * (dt / 6.0)
    return (q, v)


# Интегрирует уравнение динамики на отрезе времени dt
# с начальными условиями q, v для дерева обьектов с корнем obj,
# считае обобщенные силы на интервале полстоянными и равными force.
# Делит интервал на меньшие длины microstep, для которых делает 
# шаги методом Рунге-Кутты.
def integrate(dt, obj, q, v, force, microstep=0.1):
    while dt > microstep:
        q, v = runge(microstep, obj, q, v, force)
        dt = dt - microstep
    q, v = runge(dt, obj, q, v, force)
    return (q, v)


# In[8]:


# Наконец зададии реальный робот.
# Он состоит из колес Wheel, прикрепленном к нему прямоугольнике Body,
# к которому в свою очередь прикреплен еще один прямоугольник Head.
# В местах соединения части могут свободно вращаться, но не могут смещаться.
# Колеса движутся по наклонной прямой без проскальзывания.
# Обобщенными координатами будут:
#   q[0] - суммарный угол поворота колеса, меряется от оси Oy,
#   q[1] - угол поворота Body относительно Wheel,
#   q[2] - угол поворота Head относительно Body.

# Выбираем далее физические константы, не имеющего никакого отношения к реальным роботам.
gravity = 2.0  # Ускорение свободного падения.

q_collection = []
v_collection = []
a_collection = []

slope = 0  # Угол наколна поверхности, по которой катиться колесо.

wheel_radius = 1.0  # Радиус колеса.
wheel_mass = 0.3  # Масса колеса
wheel_moi = 0.5 * wheel_mass * wheel_radius * wheel_radius  # Момент инерции колеса (диск постоянной плотности)

body_width = 0.5  # Высота части Body
body_height = 5.0  # Ширина Body.
body_offset = 2.0  # Расстояние от центра до места закрепления колеса Wheel.
body_mass = 0.3  # Масса Body.
# Момент инерции Body считается как момент инерции прямоугольника постоянной плотности
# вокруг оси, смещенной относительно центра на body_offset.
body_moi = (body_height ** 2 + body_width ** 2) * body_mass / 12.0 + body_mass * body_offset ** 2

head_width = 0.5  # Высота
head_height = 2.0  # и ширина части Head.
head_offset = 0.5  # Смещение точки закрепления Head к Body относительно центра масс Head.
head_mass = 0.2  # Масса Head.
head_moi = (head_height ** 2 + head_width ** 2) * head_mass / 12.0 + head_mass * head_offset ** 2  # момент инерции

# Начальное состояние
# q0 = np.array([0.0, 0.0, 0.0], dtype=np.float32) # обобщенные координаты
# v0 = np.array([0.0, 0.0, 0.0], dtype=np.float32) # обобщенные скорости

q0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # обобщенные координаты
v0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # обобщенные скорости
# q0 = np.array([0.0, -0.03141593], dtype=np.float32) # обобщенные координаты
# v0 = np.array([-1.6479, 1.6479], dtype=np.float32) # обобщенные скорости

target_delta = 5.0


# Перечисляем все части робота.

# Колесо
class Wheel(Compound):
    # Принципиально важно переопределить метод, возвращающий матрицу перехода U для колеса.
    # Сначала мы поворачиваем колесо, чтобы выпрямить наклон пола,
    # затем поворачиваем на угол поворота колеса,
    # затем смещаем колесо на пройденное им расстояние (нет проскальзывания),
    # поднимаем колесо над полом на его радиус,
    # поворачиваем сцену обратно, чтобы пол оказался под нужным углом.
    def propagator(self, q):
        return AD(rotate(slope) @ translate(0, wheel_radius)) * AD.translate_x(q, 0) * AD.rotate(q, 0) * AD(
            rotate(-slope))

    # Переопределяем значения по умолчанию.
    def __init__(self, ctx):
        super().__init__(figure=Circle(ctx),
                         color=(0.7, 0.8, 0.0, 1.0),
                         mass=wheel_mass,
                         moi=wheel_moi,
                         modifier=stretch(wheel_radius, wheel_radius)
                         )


# Тело, стоящее на колесе
class Body(Compound):
    def propagator(self, q):
        return AD.rotate(q, 1) * AD(translate(0.0, body_offset))

    def __init__(self, ctx):
        super().__init__(figure=Rectangle(ctx)
                         , modifier=stretch(body_width / 2.0, body_height / 2.0)
                         , color=(0.1, 0.4, 0.7, 0.5)
                         , mass=body_mass
                         , moi=body_moi
                         )


# Голова, прикрепленная к телу
class Head(Compound):
    def propagator(self, q):
        return AD(translate(0.0, body_height / 2.0 - head_offset)) * AD.rotate(q, 2) * AD(translate(0.0, head_offset))

    def __init__(self, ctx):
        super().__init__(figure=Rectangle(ctx)
                         , modifier=stretch(head_width / 2.0, head_height / 2.0)
                         , color=(0.4, 0.7, 0.1, 0.7)
                         , mass=head_mass
                         , moi=head_moi
                         )


############################################################################
## Контроллер

timestamp = None
err0 = None
ierr = 0.0


def controller_pid(obj, q, v, target, dt):  # -> force
    global err0, ierr, timestamp
    err = q[0] + q[1] - target
    ierr = ierr + err * dt
    if err0 is None:
        f = 0.0
        timestamp = time.time()
    else:
        derr = (err - err0) / dt
        f = 3 * err + 100 * derr + 100.0 * ierr
        # print(err)
        # print("err {} derr {} ierr f {}".format(err, derr, f), end="\r")
    err0 = err
    return np.array([0, f], dtype=np.float32)


last_error = 0
ITerm = 0


def controller_pid_2(obj, q, v, target, dt):
    global last_error, ITerm, timestamp
    if timestamp is None:
        timestamp = time.time()
    Kp = 3
    Ki = 3
    Kd = 0.0
    error = q[0] + q[1] - target
    delta_time = dt
    delta_error = error - last_error

    PTerm = Kp * error
    ITerm += error * delta_time

    DTerm = 0.0
    if delta_time > 0:
        DTerm = delta_error / delta_time

    # Remember last time and last error for next calculation
    last_error = error
    print("pit_controller {}, {}, {}".format(PTerm, ITerm, DTerm))
    output = PTerm + (Ki * ITerm) + (Kd * DTerm)

    return np.array([0, output], dtype=np.float32)

new_target_force = []

def controller_const(obj, q, v, target, dt):
    global new_target_force
    target_acc = np.minimum(1.4, 0.2 * np.abs(target - v[0])) * np.sign(target - v[0])
    res = find_angle_by_acceleration(obj, target_acc)
    delta_q = res[0]
    force0 = res[1]
    force = force0 + 100.0 * (q[0] + q[1] - delta_q) + 100.0 * (v[0] + v[1])
    new_target_force.append(res)
    # force2 = force + 3.0 * (q[1] + q[2] - delta_q) + 3.0 * (v[1] + v[2])
    print(force)
    print("V {:0.5} {:0.5} A {:0.5} Q {:0.5} F {:0.5} {:0.5}     ".format(target, v[0], target_acc, delta_q, force0,
                                                                          force), end="\r")
    return np.array([0.0, force, 0.0], dtype=np.float32)


def find_angle_by_acceleration(obj, target):
    def fn(arg):  # Return zero vector for stationary motion with given velocity target.
        global q_collection, v_collection, a_collection
        q = np.array([0.0, arg[0], arg[1]])  # Body slope is to be computed.
        v = np.array([0.1, -0.1, -0.1])  # Can be arbitrary.
        force = np.array([0, arg[1], arg[2]])  # The force is the second unknown.
        v, a = rhs(obj, q, v, force)
        print("A {:0.5} {:0.5} {:0.5}".format(a[0], a[1], a[2]), end="\r")
        q_collection.append(q)
        v_collection.append(v)
        a_collection.append(a)
        return a - np.array([target, -target, -target])

    arg0 = np.array([0.0, 0.0, 0.0])  # Initial approximation
    sol = scipy.optimize.root(fn, arg0)
    return sol.x if sol.success else None


controllers = {
    "CONST": controller_const,
    "PID": controller_pid,
}


# In[9]:


# Класс для окна, откуда вызывается отрисовка сцены, обсчет динамики и взаимодействие с пользователем.
class MainWindow(Example):
    def __init__(self):
        global timestamp
        self.ctx = mgl.create_context()  # Создаем OpenGL контекст

        # Здесь будут храниться текущие координаты и скорости.
        self.q = np.array(q0)  # Инициализируем начальными значениями.
        self.v = np.array(v0)

        # Создаем объект для пола, по которому катится робот.
        # Динамику далее для него не обсчитываем, с колесом он не взаимодествует,
        # используется только для визуализации.
        floor = Compound(modifier=rotate(slope) @ stretch(1000, 10) @ translate(0, -1)
                         , figure=Rectangle(self.ctx)
                         , color=(0.6, 0.5, 0.3, 1.0)
                         )
        # Отметка для начального положения робота.
        # Также используется только для визуализации.
        origin = Compound(modifier=stretch(0.01, 0.5) @ translate(0, -1)
                          , figure=Rectangle(self.ctx)
                          , color=(0.3, 0.0, 0.0, 1.0)
                          )
        # Собираем дерево обьектов для робота.
        self.wheel = Wheel(self.ctx)  # Корнем будет колесо.
        self.wheel.adopt("body", Body(self.ctx))  # Добавляем к нему зависимых оьект Body.
        self.wheel["body"].adopt("head", Head(self.ctx))  # К Body прикрепляем Head.
        # Теперь собираем сцену, т.е. все обьекты, которые есть на экране:
        # пол, зарубка в начале, корневой обьект для робота.
        self.objects = [floor, origin, self.wheel]
        # На эту точку в мировых координатах смотрит камера.
        self.pov = (0.0, 0.0)

        self.info = ""

        self.controller = controllers[next(iter(controllers))]
        self.target_speed = 1.0
        timestamp = time.time()

    # Этот методы вызывается при отрисовке каждого кадра.
    def render(self):
        self.update_camera()  # Обновляем положение камеры, чтобы она всегда глядела на робота.
        # Вычисляем масштаб по ширине и высоте.
        scale = (1.0, self.wnd.ratio) if self.wnd.ratio < 1 else (1.0 / self.wnd.ratio, 1.0)
        # Настраиваем OpenGL контекст.
        self.ctx.enable_only(mgl.BLEND)  # Разрешаем отрисовку полупрозрачных обьектов.
        self.ctx.viewport = self.wnd.viewport  # Размер области отрисовки совпадает со всем окном.
        self.ctx.clear(0.8, 0.8, 0.9)  # Заливаем фон.
        for obj in self.objects:  # Отрисовываем все обьекты в сцене.
            obj.render(self.q, scale=scale, transform=stretch(0.1, 0.1) @ translate(-self.pov[0], -self.pov[
                1]))  # Slow, if obj has lot of parents

        self.physics()  # обновляем положения обьектов.

    def update_camera(self):  # обновляем положение камеры.
        U = self.wheel.propagator(self.q).value()  # Матрица U для робота.
        pos = (U[0, 2], U[1, 2])  # Центр масс для колеса = желаемое положение камеры.
        # a = 1.0-np.power(0.3, self.wnd.delta) # Скорость стремления к желаемому положению.
        self.pov = pos

    # Здесь обсчитывается физика и взаимодействие с пользователем.
    def physics(self):
        global timestamp
        # if np.any(self.wnd.keys): print(np.nonzero(self.wnd.keys)) # Показывает нажатые клавиши

        # Меняем контроллер при необходимости:
        for n, k in enumerate(controllers.keys()):
            c = controllers[k]
            if self.wnd.keys[49 + n] and self.controller != c:
                print("Controller: {}".format(k))
                self.controller = c

        # При нажатии проблема сбрасываем состояние робота в начальное.
        if self.wnd.keys[32]:  # SPACE
            self.q = np.zeros_like(self.q)
            self.v = np.zeros_like(self.v)
            self.target_speed = 0.0

        # Считаем момент силы (тяга двигателя) приложенной к колесу.
        if self.wnd.keys[115] or self.wnd.keys[84]:
            self.target_speed = 0.0
        if self.wnd.keys[97] or self.wnd.keys[81]:
            self.target_speed -= target_delta * self.wnd.delta
        if self.wnd.keys[100] or self.wnd.keys[83]:
            self.target_speed += target_delta * self.wnd.delta

        # Считаем момент силы, вращающей голову.
        torque2 = 0.0
        if self.wnd.keys[119] or self.wnd.keys[82]:
            torque2 += 1.0
        if self.wnd.keys[115] or self.wnd.keys[84]:
            torque2 -= 1.0

        # Делаем шаг динамики.
        dt = self.wnd.delta  # Приращение времени симуляции за один шаг.
        force = self.controller(self.wheel, self.q, self.v, self.target_speed, dt)  # Вектор обобщенных сил.

        # Интегрируем уравнения динамики.
        self.q, self.v = integrate(dt, self.wheel, self.q, self.v, force)
        print(self.target_speed, self.v, timestamp is None, abs(self.target_speed) - abs(self.v[0]))
        if abs(self.target_speed - self.v[0]) < 0.0001:
            print(time.time() - timestamp)
            exit()
        # Пересчитываем энергию и выводим частоту кадров в консоль.
        e = energy(self.wheel, self.q, self.v)
        self.info = "E: {:.4}".format(e)

        # Эта функция запускает все приложение.


def run():
    global new_target_force
    print("Balancer robot.\n  LEFT or A - tilt to the left.\n  RIGHT or D - tilt to the right.\n  SPACE - reset.\n")
    run_example(MainWindow)
    fig, axs = plt.subplots(3, 1)
    q_collect = np.asarray(new_target_force)
    axs[0].plot(q_collect[:, 0])
    axs[1].plot(q_collect[:, 1])
    axs[2].plot(q_collect[:, 2])

    fig.tight_layout()
    plt.show()

# In[10]:


run()
