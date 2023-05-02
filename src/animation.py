import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
from utils import pairIdx2flatIdx
from src.dataset import Dataset

class Animation:

    def __init__(self, embs, frame_times: np.asarray = None, dataset: Dataset = None, directed=False,
                 figsize=(12, 10), node_sizes=100, node2color: list = None,
                 color_palette="rocket_r", padding=0.1, fps=6,):

        self._figsize = figsize
        self._anim = None
        self._fps = fps

        # Data properties
        self._embs_s = embs[0]
        self._embs_r = embs[1] if directed else None
        self._frame_times = frame_times
        pair2id, self._event_pairs, self._event_times, self._event_states = dict(), [], [], []
        for pair, e, s in zip(dataset.get_edges().T.tolist(), dataset.get_times(), dataset.get_states()): #zip(data[0], data[1]):
            pair_idx = pairIdx2flatIdx(pair[0], pair[1], self._embs_s.shape[1], directed=directed)
            if pair_idx not in pair2id:
                pair2id[pair_idx] = len(pair2id)
                self._event_pairs.append(pair)
                self._event_times.append([e])
                self._event_states.append([s])
            else:
                self._event_times[pair2id[pair_idx]].append(e)
                self._event_states[pair2id[pair_idx]].append(s)

        self._directed = directed
        self._frames_num = self._embs_s.shape[0]
        self._nodes_num = self._embs_s.shape[1]
        self._dim = self._embs_s.shape[2]

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
        global sc, sc_r, ax

        def __set_canvas():

            xy_min = self._embs_s.min(axis=0, keepdims=False).min(axis=0, keepdims=False)
            xy_max = self._embs_s.max(axis=0, keepdims=False).max(axis=0, keepdims=False)
            if self._directed:
                xy_min_r = self._embs_r.min(axis=0, keepdims=False).min(axis=0, keepdims=False)
                xy_max_r = self._embs_r.max(axis=0, keepdims=False).max(axis=0, keepdims=False)

                xy_min = np.minimum(xy_min, xy_min_r)
                xy_max = np.maximum(xy_max, xy_max_r)

            xlen_padding = (xy_max[0] - xy_min[0]) * self._padding
            ylen_padding = (xy_max[1] - xy_min[1]) * self._padding
            ax.set_xlim([xy_min[0] - xlen_padding, xy_max[0] + xlen_padding])
            ax.set_ylim([xy_min[1] - ylen_padding, xy_max[1] + ylen_padding])

        def __init_func():
            global sc, sc_r, ax

            sc = ax.scatter(
                [0]*self._nodes_num, [0]*self._nodes_num,
                s=self._node_sizes, c=self._node_colors,
                linewidths=self._linewidths, edgecolors=self._edgecolors
            )
            sc_r = None
            if self._directed:
                sc_r = ax.scatter(
                    [0]*self._nodes_num, [0]*self._nodes_num,
                    s=self._node_sizes, c=self._node_colors,
                    linewidths=self._linewidths, edgecolors=self._edgecolors,
                    marker='>'
                )

            __set_canvas()

        def __func(f):
            global sc, sc_r, ax

            for line in list(ax.lines):
                ax.lines.remove(line)

            ax.set_title("Time (t={:0.2f})".format(self._frame_times[f]), fontsize=16)

            # Plot the nodes
            sc.set_offsets(np.c_[self._embs_s[f, :, 0], self._embs_s[f, :, 1]])
            
            if self._directed:
                sc_r.set_offsets(np.c_[self._embs_r[f, :, 0], self._embs_r[f, :, 1]])

            # Plot the event links
            if self._event_times is not None and self._event_pairs is not None:

                for pair_events, pair_states, pair in zip(self._event_times, self._event_states, self._event_pairs):

                    # Get the i, j indices of the pair
                    i, j = pair

                    # Find the index of the list of event times that is the closest to the frame time
                    idx = np.argmin(abs(self._frame_times[f] - pair_events))

                    current_state = None
                    if self._frame_times[f] < pair_events[idx]:
                        if idx == 0:
                            current_state = not pair_states[idx]
                        else:
                            current_state = pair_states[idx - 1]
                    else:
                        current_state = pair_states[idx]

                    # Since the number of events that are smaller or equal to the frame time is odd, self._frame_times[f] >= pair_events[idx]
                    # Set the alpha value to 1 if there is only one event (idx must be 0 in this case) or the index is the last event
                    if current_state:
                        alpha = 1
                    else:
                        # assert self._frame_times[f] >= pair_events[idx], f"Ohh No! {self._frame_times[f]}, {pair_events[idx]}"
                        alpha = 0 #1 - (self._frame_times[f]-pair_events[idx])/(pair_events[idx+1]-pair_events[idx])

                    if self._directed:
                        ax.plot(
                            [self._embs_s[f, i, 0], self._embs_r[f, j, 0]],
                            [self._embs_s[f, i, 1], self._embs_r[f, j, 1]],
                            color='k',
                            alpha=alpha
                        )
                    else:
                        ax.plot(
                            [self._embs_s[f, i, 0], self._embs_s[f, j, 0]],
                            [self._embs_s[f, i, 1], self._embs_s[f, j, 1]],
                            color='k',
                            alpha=alpha
                        )

        anim = animation.FuncAnimation(
            fig=fig, init_func=__init_func, func=__func, frames=self._frames_num, interval=200, repeat=repeat
        )
        
        return anim

    def save(self, filepath, format="mp4"):
        global sc, sc_r, ax

        fig, ax = plt.subplots(figsize=self._figsize, frameon=True)
        # ax.set_axis_off()
        # fig.subplots_adjust(left=0, bottom=0, right=1, top=0.95)

        self._anim = self._render(fig)
        

        if format == "mp4":
            writer = animation.FFMpegWriter(fps=self._fps)

        elif format == "gif":
            writer = animation.PillowWriter(fps=self._fps)

        else:
            raise ValueError("Invalid format!")

        self._anim.save(filepath, writer)