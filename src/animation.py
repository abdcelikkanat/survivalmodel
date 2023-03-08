import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
from utils import pairIdx2flatIdx

class Animation:

    def __init__(self, embs, frame_times: np.asarray = None, data: tuple = (None, None),
                 figsize=(12, 10), node_sizes=100, node2color: list = None,
                 color_palette="rocket_r", padding=0.1, fps=6,):

        self._figsize = figsize
        self._anim = None
        self._fps = fps

        # Data properties
        self._embs = embs
        self._frame_times = frame_times
        pair2id, self._event_pairs, self._event_times = dict(), [], []
        for pair, e in zip(data[0], data[1]):
            pair_idx = pairIdx2flatIdx(pair[0], pair[1], embs.shape[1])
            if pair_idx not in pair2id:
                pair2id[pair_idx] = len(pair2id)
                self._event_pairs.append(pair)
                self._event_times.append([e])
            else:
                self._event_times[pair2id[pair_idx]].append(e)
        # self._event_pairs = data[0]
        # self._event_times = data[1]
        self._frames_num = embs.shape[0]
        self._nodes_num = embs.shape[1]
        self._dim = embs.shape[2]

        # Visual properties
        sns.set_theme(style="ticks")
        node2color = [0]*self._nodes_num if node2color is None else node2color
        self._color_num = 1 if node2color is None else len(set(node2color))
        self._palette = sns.color_palette(color_palette, self._color_num)
        self._node_colors = [self._palette.as_hex()[node2color[node]] for node in range(self._nodes_num)]
        self._node_sizes = [node_sizes]*self._nodes_num if type(node_sizes) is int else node_sizes
        self._linewidths = 1
        self._edgecolors = 'k'
        self._padding = padding

    def _render(self, fig, repeat=False):
        global sc, ax

        def __set_canvas():

            xy_min = self._embs.min(axis=0, keepdims=False).min(axis=0, keepdims=False)
            xy_max = self._embs.max(axis=0, keepdims=False).max(axis=0, keepdims=False)
            xlen_padding = (xy_max[0] - xy_min[0]) * self._padding
            ylen_padding = (xy_max[1] - xy_min[1]) * self._padding
            ax.set_xlim([xy_min[0] - xlen_padding, xy_max[0] + xlen_padding])
            ax.set_ylim([xy_min[1] - ylen_padding, xy_max[1] + ylen_padding])

        def __init_func():
            global sc, ax

            sc = ax.scatter(
                [0]*self._nodes_num, [0]*self._nodes_num,
                s=self._node_sizes, c=self._node_colors,
                linewidths=self._linewidths, edgecolors=self._edgecolors
            )

            __set_canvas()

        def __func(f):
            global sc, ax

            for line in list(ax.lines):
                ax.lines.remove(line)

            # Plot the nodes
            sc.set_offsets(np.c_[self._embs[f, :, 0], self._embs[f, :, 1]])

            # Plot the event links
            if self._event_times is not None and self._event_pairs is not None:

                for pair_events, pair in zip(self._event_times, self._event_pairs):

                    # Get the i, j indices of the pair
                    i, j = pair

                    # Get the number of events that are smaller or equal the frame time
                    w = sum(self._frame_times[f] - pair_events >= 0)  

                    # By assumption, initally links are not present so 'w' must be an odd number to plot a link
                    if w % 2 == 1:

                        # Find the index of the list of event times that is the closest to the frame time
                        idx = np.argmin(self._frame_times[f] - pair_events)

                        # Since the number of events that are smaller or equal to the frame time is odd, self._frame_times[f] >= pair_events[idx]
                        # Set the alpha value to 1 if there is only one event (idx must be 0 in this case) or the index is the last event
                        if idx == len(pair_events)-1:
                            alpha = 1
                        else:
                            assert self._frame_times[f] >= pair_events[idx], "OHA!"
                            alpha = 1 - (self._frame_times[f]-pair_events[idx])/(pair_events[idx+1]-pair_events[idx])

                        ax.plot(
                            [self._embs[f, i, 0], self._embs[f, j, 0]],
                            [self._embs[f, i, 1], self._embs[f, j, 1]],
                            color='k',
                            alpha=alpha
                        )

        anim = animation.FuncAnimation(
            fig=fig, init_func=__init_func, func=__func, frames=self._frames_num, interval=200, repeat=repeat
        )

        return anim

    def save(self, filepath, format="mp4"):
        global sc, ax

        fig, ax = plt.subplots(figsize=self._figsize, frameon=True)
        ax.set_axis_off()
        x_min, y_min = self._embs.min(axis=0).min(axis=0)
        x_max, y_max = self._embs.max(axis=0).max(axis=0)
        self._anim = self._render(fig)

        # fig.set_size_inches(y_max-y_min, x_max-x_min, )
        if format == "mp4":
            writer = animation.FFMpegWriter(fps=self._fps)

        elif format == "gif":
            writer = animation.PillowWriter(fps=self._fps)

        else:
            raise ValueError("Invalid format!")

        self._anim.save(filepath, writer)


# embs = np.random.randn(100, 10, 2)
# anim = Animation(embs)
# anim.save("./deneme.mp4")