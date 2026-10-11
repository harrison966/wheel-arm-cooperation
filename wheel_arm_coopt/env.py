# env.py
# 作用：定义栅格地图、障碍物、A* 路径规划、路径重采样
# 所有长度单位：米；栅格分辨率默认 0.1 米

import numpy as np
import heapq  # 优先队列，用于 A* 的 open set


class GridMap:
    """栅格地图类：把连续世界离散成 0.1m 的格子，标记障碍物"""

    def __init__(self, xlim=(0, 10), ylim=(0, 6), res=0.1):
        self.xlim = xlim          # x 方向范围 [0, 10] 米
        self.ylim = ylim          # y 方向范围 [0, 6] 米
        self.res = res            # 每个栅格的边长，0.1 米
        # 栅格数量：100 x 60
        self.nx = int((xlim[1] - xlim[0]) / res)
        self.ny = int((ylim[1] - ylim[0]) / res)
        # grid[i, j] = True 表示该格子是障碍物
        self.grid = np.zeros((self.nx, self.ny), dtype=bool)

    def add_rect_obstacle(self, x0, y0, x1, y1):
        """添加一个矩形障碍物，用世界坐标 (x0,y0)-(x1,y1) 指定"""
        # 世界坐标 -> 栅格索引
        i0 = int((x0 - self.xlim[0]) / self.res)
        i1 = int((x1 - self.xlim[0]) / self.res)
        j0 = int((y0 - self.ylim[0]) / self.res)
        j1 = int((y1 - self.ylim[0]) / self.res)
        # 把这块区域标记为障碍
        self.grid[i0:i1, j0:j1] = True

    def world_to_grid(self, p):
        """世界坐标 (x, y) -> 栅格索引 (i, j)"""
        i = int((p[0] - self.xlim[0]) / self.res)
        j = int((p[1] - self.ylim[0]) / self.res)
        return i, j

    def grid_to_world(self, i, j):
        """栅格索引 (i, j) -> 该格子中心的世界坐标"""
        return (self.xlim[0] + (i + 0.5) * self.res,
                self.ylim[0] + (j + 0.5) * self.res)

    def is_free(self, i, j):
        """判断栅格 (i, j) 是否可通行（在边界内且非障碍）"""
        if i < 0 or i >= self.nx or j < 0 or j >= self.ny:
            return False
        return not self.grid[i, j]

    def astar(self, start, goal):
        """
        A* 路径规划。
        输入：start, goal 为世界坐标 (x, y)
        输出：从 start 到 goal 的世界坐标列表，形如 [(x0,y0), (x1,y1), ...]
        """
        s = self.world_to_grid(start)
        g = self.world_to_grid(goal)

        # 起点或终点落在障碍里，直接报错
        if not self.is_free(*s) or not self.is_free(*g):
            raise ValueError("起点或终点在障碍内")

        # open_set 是优先队列，元素为 (f_score, 栅格索引)
        open_set = [(0, s)]
        came = {}        # 记录每个节点的父节点，用于回溯路径
        gscore = {s: 0}  # 从起点到该节点的实际代价

        while open_set:
            _, cur = heapq.heappop(open_set)  # 取出 f 最小的节点

            if cur == g:
                # 到达终点，回溯路径
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                # 转成世界坐标返回
                return [self.grid_to_world(i, j) for i, j in path]

            # 8 邻域扩展
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nb = (cur[0] + di, cur[1] + dj)
                if not self.is_free(*nb):
                    continue
                # 对角移动代价是 sqrt(2)，直行是 1
                step = np.hypot(di, dj)
                ng = gscore[cur] + step

                # 找到更优路径就更新
                if nb not in gscore or ng < gscore[nb]:
                    gscore[nb] = ng
                    came[nb] = cur
                    # 启发式：欧氏距离
                    h = np.hypot(nb[0] - g[0], nb[1] - g[1])
                    heapq.heappush(open_set, (ng + h, nb))

        raise RuntimeError("A* 未找到路径")


def build_default_map():
    """构建实验用的默认地图：三个障碍物，模拟仓储货架"""
    m = GridMap()
    m.add_rect_obstacle(4.0, 2.0, 4.5, 4.0)   # 中部障碍
    m.add_rect_obstacle(7.5, 1.0, 8.0, 3.5)   # 右侧障碍
    m.add_rect_obstacle(1.5, 4.5, 3.0, 5.0)   # 左下障碍
    return m


def path_length(path):
    """计算路径总长度（所有相邻点距离之和）"""
    p = np.array(path)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def resample_path(path, ds=0.05):
    """
    按弧长等距重采样路径。
    原因：A* 输出的点间距不均匀（直行 0.1m，对角 0.14m），
    重采样后便于后续做速度规划和弧长积分。
    返回：重采样后的点集 (N,2)，以及对应的弧长数组 s
    """
    p = np.array(path)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)  # 每段长度
    s = np.concatenate([[0], np.cumsum(seg)])         # 累积弧长
    total = s[-1]
    n = max(int(total / ds), 2)                       # 采样点数
    s_new = np.linspace(0, total, n)                  # 均匀弧长
    # 按弧长插值出新的 x, y
    x = np.interp(s_new, s, p[:, 0])
    y = np.interp(s_new, s, p[:, 1])
    return np.stack([x, y], axis=1), s_new