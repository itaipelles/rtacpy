import numpy as np
from etacorpy.calc_rta_n import calc_area_of_union_of_rectangles

# lists of rectangles are defined by (x1, y1, x2, y2)

def helper_function_test_rects_area(rects, expected_area):
    area = calc_area_of_union_of_rectangles(rects)
    assert(np.abs(area - expected_area) < 1e-12)

def test_disjoint():
    helper_function_test_rects_area(np.array([[0,0,1,1],[1,1,2,2]]), 2)
    helper_function_test_rects_area(np.array([[0,0,1,1],[0,1,1,2]]), 2) # same x
    helper_function_test_rects_area(np.array([[0,0,1,1],[1,0,2,1]]), 2) # same y
    helper_function_test_rects_area(np.array([[0,0,1,1],[10,10,12,12]]), 5)
    

def test_contains():
    helper_function_test_rects_area(np.array([[0,0,1,1],[0,0,2,2]]), 4)
    helper_function_test_rects_area(np.array([[0,0,1,1],[1,1,2,2],[0,0,3,3]]), 9)
    
    
def test_partial_overlap():
    helper_function_test_rects_area(np.array([[0,0,1,2],[0,0,2,1]]), 3)
    helper_function_test_rects_area(np.array([[0,0,1,2],[0,0,2,1]]), 3)
    helper_function_test_rects_area(np.array([[0,0,2,2],[1,1,3,3]]), 7)
    

    
    
    