# -*- coding: utf-8 -*-
# env.py
# 栅格地图 + A* 路径规划

import numpy as np
import heapq


class GridMap:
    def __init__(self, xlim=(0, 10), ylim=(0, 6), res=0.1):
        self.xlim = xlim
        self.ylim = ylim
        self.res = res
        self.nx = int((xlim[1] - xlim[0]) / res)
        self.ny = int((ylim[1] - ylim[0]) / res)
        self.grid = np.zeros((self.nx, self.ny), dtype=bool)

    def add_rect_obstacle(self, x0, y0, x1, y1):
        i0 = int((x0 - self.xlim[0]) / self.res)
        i1 = int((x1 - self.xlim[0]) / self.res)
        j0 = int((y0 - self.ylim[0]) / self.res)
        j1 = int((y1 - self.ylim[0]) / self.res)
        self.grid[i0:i1, j0:j1] = True

    def world_to_grid(self, p):
        i = int((p[0] - self.xlim[0]) / self.res)
        j = int((p[1] - self.ylim[0]) / self.res)
        return i, j

    def grid_to_world(self, i, j):
        return (self.xlim[0] + (i + 0.5) * self.res,
                self.ylim[0] + (j + 0.5) * self.res)

    def is_free(self, i, j):
        if i < 0 or i >= self.nx or j < 0 or j >= self.ny:
            return False
        return not self.grid[i, j]

    def astar(self, start, goal):
        s = self.world_to_grid(start)
        g = self.world_to_grid(goal)
        if not self.is_free(*s) or not self.is_free(*g):
            raise ValueError("起点或终点在障碍内")
        open_set = [(0, s)]
        came = {}
        gscore = {s: 0}
        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur == g:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return [self.grid_to_world(i, j) for i, j in path]
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nb = (cur[0] + di, cur[1] + dj)
                if not self.is_free(*nb):
                    continue
                step = np.hypot(di, dj)
                ng = gscore[cur] + step
                if nb not in gscore or ng < gscore[nb]:
                    gscore[nb] = ng
                    came[nb] = cur
                    h = np.hypot(nb[0] - g[0], nb[1] - g[1])
                    heapq.heappush(open_set, (ng + h, nb))
        raise RuntimeError("A* 未找到路径")


def build_default_map():
    m = GridMap()
    m.add_rect_obstacle(4.0, 2.0, 4.5, 4.0)
    m.add_rect_obstacle(7.5, 1.0, 8.0, 3.5)
    m.add_rect_obstacle(1.5, 4.5, 3.0, 5.0)
    return m


def resample_path(path, ds=0.05):
    p = np.array(path)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    n = max(int(total / ds), 2)
    s_new = np.linspace(0, total, n)
    x = np.interp(s_new, s, p[:, 0])
    y = np.interp(s_new, s, p[:, 1])
    return np.stack([x, y], axis=1), s_new