from typing import List
import numpy as np
import numba as nb
from numba import njit
from numba.experimental import jitclass
from numpy.typing import ArrayLike

OPENING = +1  # constants for events
CLOSING = -1  # -1 has higher priority

spec = [
    ("N", nb.int64),
    ("c", nb.float64[:]),
    ("s", nb.float64[:]),
    ("w", nb.float64[:])
]

@jitclass(spec=spec)
class CoverQuery:
    """Segment tree to maintain a set of integer intervals
    and permitting to update their coverage and query the length of all covered parts.
    """
    def __init__(self, L: List[float]):
        """creates a structure, where all possible intervals
        will be included in [0, L - 1].
        """
        self.N = 1
        while self.N < len(L):
            self.N *= 2
        self.c = np.zeros(2 * self.N)         # --- covered
        self.s = np.zeros(2 * self.N)         # --- score
        self.w = np.zeros(2 * self.N)         # --- length
        for i, _ in enumerate(L):
            self.w[self.N + i] = L[i]
        for p in range(self.N - 1, 0, -1):
            self.w[p] = self.w[2 * p] + self.w[2 * p + 1]


    def get_total_covered_length(self):
        """:returns: the size of the union of the stored intervals
        """
        return self.s[1]

@njit
def get_total_covered_length(CQ: CoverQuery):
    """:returns: the size of the union of all stored intervals that are currently covered
    """
    return CQ.s[1]

@njit
def update_coverage_from_to_wrapper(CQ: CoverQuery, i:int, k:int, offset:int):
        """when offset = +1, adds an interval [i, k],
        when offset = -1, removes it
        :complexity: O(log L)
        """
        update_coverage_from_to(CQ, 1, 0, CQ.N, i, k, offset)

@njit
def update_coverage_from_to(CQ: CoverQuery, p: int, start:int, span:int, i:int, k:int, offset:int) -> None:
    if start + span <= i or k <= start:   # --- disjoint
        return
    if i <= start and start + span <= k:  # --- included
        CQ.c[p] += offset
    else:
        update_coverage_from_to(CQ, 2 * p, start, span // 2, i, k, offset)
        update_coverage_from_to(CQ, 2 * p + 1, start + span // 2, span // 2,
                        i, k, offset)
    if CQ.c[p] == 0:
        if p >= CQ.N:                   # --- leaf
            CQ.s[p] = 0
            return
        else:
            CQ.s[p] = CQ.s[2 * p] + CQ.s[2 * p + 1]
            return
    else:
        CQ.s[p] = CQ.w[p]
        return

@njit
def calc_area_of_union_of_rectangles(R):
    """
    :param R: list of rectangles defined by (x1, y1, x2, y2)
       where (x1, y1) is top left corner and (x2, y2) bottom right corner
    :returns: Area of union of rectangles
    :complexity: :math:`O(n \\log n)`
    """
    if R.shape[0]==0:               # segment tree would fail on an empty list
        return 0
    X = set()                 # set of all x coordinates in the input
    events = []               # events for the sweep line
    for Rj in R:
        (x1, y1, x2, y2) = Rj
        assert x1 <= x2 and y1 <= y2
        X.add(x1)
        X.add(x2)
        events.append((y1, OPENING, x1, x2))
        events.append((y2, CLOSING, x1, x2))
    i_to_x = list(sorted(X))
    # inverse dictionary
    x_to_i = {i_to_x[i]: i for i in range(len(i_to_x))}
    L = [i_to_x[i + 1] - i_to_x[i] for i in range(len(i_to_x) - 1)]
    CQ = CoverQuery(L)
    area = 0
    previous_y = 0  # arbitrary initial value,
    #                 because C.cover() is 0 at first iteration
    for y, offset, x1, x2 in sorted(events):
        area += (y - previous_y) * CQ.get_total_covered_length()
        i1 = x_to_i[x1]
        i2 = x_to_i[x2]
        update_coverage_from_to_wrapper(CQ, i1, i2, offset)
        previous_y = y
    return area

@njit
def create_clipped_rectangles_around_points(x: ArrayLike, y: ArrayLike, edge_length: float):
    half_edge_length = 0.5*edge_length
    arrays = (np.maximum(np.subtract(x,half_edge_length),0), 
        np.maximum(np.subtract(y,half_edge_length),0), 
        np.minimum(np.add(x,half_edge_length),1), 
        np.minimum(np.add(y,half_edge_length),1))
    return np.vstack(arrays).T

@njit
def fast_rank_data(data: ArrayLike, n: int) -> ArrayLike:
    temp = np.argsort(data)
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(n)+1
    return ranks

@njit
def calc_rta_n(x,y, coverage_factor=1.0, edge_length=None):
    n = len(x)
    x = fast_rank_data(x,n)/(n+1)
    y = fast_rank_data(y,n)/(n+1)
    edge_length = edge_length or np.sqrt(coverage_factor/n)
    rectangles = create_clipped_rectangles_around_points(x,y,edge_length)
    return calc_area_of_union_of_rectangles(rectangles)
