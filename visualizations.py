#TODO: x, y ==> class attributes
import matplotlib.pyplot as plt
import numpy as np
import time

plt.ion()
class DynamicChart:
    def __init__(self, tickshape='-', grid=True, autoX=True, autoY=True,
                 minX=None, maxX=None, minY=None, maxY=None):
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([], [], tickshape)

        if autoX:
            self.ax.set_autoscalex_on(True)
        else:
            if not (minX and maxX):
                raise Exception('must provide range for x-axis [minX, maxX]')
            self.ax.set_xlim(minX, maxX)

        if autoY:
            self.ax.set_autoscaley_on(True)
        else:
            if not (minY and maxY):
                raise Exception('must provide range for y-axis [minY, maxY]')
            self.ax.set_xlim(minY, maxY)

        if grid:
            self.ax.grid()


    def updateChart(self, ydata, xdata=None):
        if not xdata:
            xdata = np.arange(len(ydata))

        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        self.ax.relim()
        self.ax.autoscale_view()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def plot(self):
        x = []
        y = []
        for i in np.arange(0, 10, 1):
            x.append(i)
            y.append(np.exp(-i ** 2) + 10 * np.exp(-(i - 7) ** 2))
            self.updateChart(xdata=x, ydata=y)
            plt.pause(0.01)
        return x, y


c = DynamicChart()
c.plot()