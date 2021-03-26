import numpy as np
import matplotlib.pyplot as plt
import math

class Logger(object):
    def __init__(self, stats=None):
        if stats == None:
            stats = ['scores', 'avg_scores']

        assert len(stats) >= 2

        self.stats_dict = {}
        self.stats_names = stats
        for stat in self.stats_names:
            self.stats_dict[stat] = []

    def log(self, ep):
        assert len(self.stats_names) == len(ep)
        for idx, val in enumerate(self.stats_names):
            self.stats_dict[val].append(ep[idx])

    def plot(self, sub_plots=1, stat_per_plot=2, scale=2, filename=None):
        assert sub_plots * stat_per_plot == len(self.stats_names)
        assert sub_plots < 7

        if sub_plots > 1:
            stats_names_idxs = {}

            # grids
            fig = plt.figure(figsize=(8,8))
            gs = fig.add_gridspec(2*scale, (sub_plots//2)*scale)
            axes = {}
            y_count = 0.0
            for i in range(sub_plots):
                y_idx = math.floor(y_count)*scale
                name = self.stats_names[i*stat_per_plot]
                idx = (i % 2)*scale
                print(idx+scale, y_idx+scale)
                axes[i] = fig.add_subplot(gs[int(idx):int(idx+scale), int(y_idx):int(y_idx+scale)])
                y_count += 0.5

            for i in range(sub_plots):
                idx = i*stat_per_plot
                names_to_add = []
                for j in range(stat_per_plot):
                    names_to_add.append(self.stats_names[idx+j])
                print(names_to_add, " a")

                stats_names_idxs[axes[i]] = names_to_add


            # plots
            for idx, axis in enumerate(stats_names_idxs):

                data_names = stats_names_idxs[axis]
                data = {}
                for name in data_names:
                    data[name] = self.stats_dict[name]


                for i in data:
                    axis.plot(data[i], label=i)
                axis.legend()

        else:
            print('aa')
            for i in self.stats_dict:
                print(self.stats_dict[i])
                plt.plot(self.stats_dict[i], label=i)
            plt.legend()

        if filename:
            plt.savefig(filename)

        plt.show()

    def get(self,name):
        return self.stats_dict[name]

    @property
    def length(self):
        for i in self.stats_dict:
            return len(self.stats_dict[i])

if __name__ ==  '__main__':
    logger = Logger(['scores', 'avgs', 'epsilon', 'aloo', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    for i in range(100):
        logger.log([1,2,3,4,5,6,7,8,9,10,11,12])
    logger.plot(sub_plots=2, stat_per_plot=2, filename='test.png')
