import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Simulation:
    def __init__(self, charging_stations, origin, destination):
        self.charging_stations = charging_stations
        self.origin = origin
        self.destination = destination
        self.ev_position = origin  # initialize ev_position as the origin
        self.fig, self.ax = plt.subplots()
        self.anim = animation.FuncAnimation(self.fig, self.update, interval=1000)

    def update_ev_position(self, new_position):
        self.ev_position = new_position
        self.anim.event_source.start()  # restart the animation

    def plot(self):
        self.ax.clear()
        self.ax.scatter(*zip(*self.charging_stations), color='blue', label='Charging stations')
        self.ax.scatter(*self.origin, color='green', label='Origin')
        self.ax.scatter(*self.destination, color='red', label='Destination')
        self.ax.scatter(*self.ev_position, color='orange', label='EV Position')
        self.ax.legend()

    def update(self, frame):
        self.plot()
        plt.draw()

    def show(self):
        plt.show()