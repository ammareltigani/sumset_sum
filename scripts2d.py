from scipy.spatial import ConvexHull
from itertools import combinations_with_replacement
from sympy import *

import ast
import csv
import itertools
import re
import numpy as np
import matplotlib.pyplot as plt


def process(l):
    assert len(l) > 0
    if type(l[0]) == int:
        l = [[e] for e in l]
    return np.array(l)


def deprocess(l):
    d = np.shape(l)[1]
    if d == 1:
        return sorted([e for e in l])
    return sorted([tuple(e) for e in l])


def sum_sets(set1, set2):
    assert np.shape(set1)[1] == np.shape(set2)[1]
    res = np.unique(np.array([np.add(e1, e2) for e1 in set1 for e2 in set2]), axis=0)
    return res


def sum_set_n_times(set1, n):
    assert n > 0
    set1 = np.array(set1)
    res = set1
    for _ in range(n-1):
        res = sum_sets(res, set1)
    return res

def sum_set_n_times_and_save(set1, n):
    assert n > 0
    res = [set([(0,0,0),]), set1]
    for _ in range(n-1):
        res.append(sum_sets(res[-1], set1))
    return res


# def intersections(basis, translations, iters):
#     rtn = []
#     # get all dilations
#     basis_dilations = [[(0,0)], basis]
#     curr_basis = basis.copy()
#     for _ in range(iters):
#         curr_basis = sum_sets(curr_basis, basis)
#         basis_dilations.append(curr_basis)
# 
#     for h in range(iters):
#         points_list = []
#         # i'th dilation
#         for i in range(h+1,-1,-1):
#             dilation = basis_dilations[i]
#             # want all partitions of i. So j*1st trans, and (i-j)*2nd trans
#             if len(translations) == 2:
#                 for j in range(h+1-i, -1, -1):
#                     tdilation = [(e[0] + j * translations[0][0], e[1] + j * translations[0][1]) for e in dilation]
#                     tdilation = [(e[0] + (h+1-i-j) * translations[1][0], e[1] + (h+1-i-j) * translations[1][1]) for e in tdilation]
#                     points_list.extend(tdilation)
#             elif len(translations) == 1:
#                 tdilation = [(e[0] + (h+1-i) * translations[0][0], e[1] + (h+1-i) * translations[0][1]) for e in dilation]
#                 points_list.extend(tdilation)
#             else:
#                 # Not implemented yet for d+4
#                 assert False
# 
#         unique, counts = np.unique(np.array(points_list), axis=0, return_counts=True)
#         l = max(iters, max(counts))
#         all_intersections = [[] for _ in range(l)]
#         for j in range(l):
#             all_intersections[j] = unique[counts > j]
#         
#         rtn.append(all_intersections)
# 
#     return rtn

def sum_list(B):
    rtn = []
    for e in B:
        e_transpose = list(zip(*e))
        e_transpose_summed = tuple([sum(x) for x in e_transpose])
        rtn.append(e_transpose_summed)
    return rtn

def intersections(A, iters):
    rtn = []
    for i in range(1, iters+1):
        combs = list(combinations_with_replacement(A, i))
        points_list = np.array(sum_list(combs))
        unique, counts = np.unique(np.array(points_list), axis=0, return_counts=True)
        l = max(iters, max(counts))
        all_intersections = [[] for _ in range(l)]
        for j in range(l):
            all_intersections[j] = unique[counts > j]
        rtn.append(all_intersections)
    return rtn


def single_sumset(A, iterations, slice=None, plot=False, print_info=False):

    A = process(A) if type(A) == list else A
    status, k, poly, lenghts_arry, _ =  run_exps_nd(A, iterations)
    if status == False:
        return False

    if len(poly) == 3:
        poly_str = f'{poly[0]}*h^2 + {poly[1]}*h + {poly[2]}'
    elif len(poly) == 4:
        poly_str = f'{poly[0]}*h^3 + {poly[1]}*h^2 + {poly[2]}*h + {poly[3]}'
    else:
        assert False

    if print_info:
        print(f'p(h) = {poly_str}  for all k >= {k}')
        print(f'Lengths Array: {lenghts_arry}')

    if plot:
        assert slice is not None
        inters = intersections(A, iterations)
        plot_sumset(
            iterations,
            slice,
            inters=inters,
        )
    
    return poly_str, lenghts_arry
            


def plot_sumset(iterations, slice, points_arry=None, inters=None, threeD=False):
    assert points_arry is not None or inters is not None
    assert points_arry is None or inters is None
    assert slice[1] <= iterations + 1

    if inters is not None:
        points_arry = [e[0] for e in inters]

    iterations = slice[1] - slice[0]
    if threeD:
        axis = plt.figure().add_subplot(projection='3d')
        for j in range(iterations):
            hull = ConvexHull(points_arry[j+slice[0]])
            points = np.array(points_arry[j+slice[0]])
            axis.plot(points[:,0], points[:,1], 'o', color='blue', zs=j, markersize=4, zdir='z')
            for simplex in hull.simplices:
                axis.plot(points[simplex, 0], points[simplex, 1], 'k--', zs=j, zdir='z', linewidth=1)

            if inters is not None:
                colors = ['yellow', 'orange', 'red', 'grey', 'black']
                for i in range(len(colors)):
                    ith_intersection = np.array(inters[j+slice[0]][i+1])
                    if ith_intersection.size != 0:
                        axis.plot(ith_intersection[:,0], ith_intersection[:,1], 'o', color=colors[i], zs=j, zdir='z', markersize=4)
    else:
        a = int(np.ceil(iterations / 2))
        _, axis = plt.subplots(a,2)
        for j in range(iterations):
            m = 1 if j % 2 == 1 else 0
            n = j // 2

            hull = ConvexHull(points_arry[j+slice[0]])
            points = np.array(points_arry[j+slice[0]])
            axis[n,m].plot(points[:,0], points[:,1], 'o', markersize=4)

            if inters is not None:
                colors = ['yellow', 'orange', 'red', 'grey', 'black']
                for i in range(len(colors)):
                    ith_intersection = np.array(inters[j+slice[0]][i+1])
                    if ith_intersection.size != 0:
                        axis[n,m].plot(ith_intersection[:,0], ith_intersection[:,1], 'o', color=colors[i], markersize=4)

            for simplex in hull.simplices:
                axis[n,m].plot(points[simplex, 0], points[simplex, 1], 'k--')
            axis[n,m].set_title(f'iter = {j+slice[0]+1}')
            axis[n,m].grid(color = 'gray', linestyle = '--', linewidth = 0.5)
            print()

    plt.show()

                

def round_volume(vol):
    orig_vol = vol
    for i in range(12):
        vol = orig_vol * (10**i) 
        if np.abs(np.round(vol) - vol)< 1e-8:
            return np.round(vol) / (10**i)
    return orig_vol


def volume_of_convex_hull(points):
    if np.shape(points)[1] == 1:
        return max(points)[0]
    return round_volume(ConvexHull(points).volume)


def run_exps(curr_set, min_iterations):
    m = volume_of_convex_hull(curr_set)
    n_set = np.copy(curr_set)
    points_list = [np.copy(n_set)]
    lengths_arry = [len(n_set)]
    k = None
    i = 0
    once = False
    while k is None or i  < min_iterations:
        n_set = sum_sets(n_set, curr_set)
        if i >= 3 and k is None:
            d11 = lengths_arry[i-2] - lengths_arry[i-3]
            d12 = lengths_arry[i-1] - lengths_arry[i-2]
            d13 = lengths_arry[i] - lengths_arry[i-1]
            d21, d22 = d12 - d11, d13 - d12
            d31 = d22 - d21

            if d31 == 0 and d21 == 2*m:
                if once:
                    k = i-3
                else:
                    once = True
            else:
                once = False
        points_list.append(n_set)
        lengths_arry.append(len(n_set))
        i+=1
    assert k is not None
    # Temp fix. TODO: make this better later.
    mm = m
    # Calculate b and c
    b = (lengths_arry[k] - lengths_arry[k-1]) - (mm*((2*(k))+1))
    c = lengths_arry[k-1] - (mm*((k)**2)) - (b*(k))
    return  m, mm, b, c, k, deprocess(points_list[-3]), lengths_arry, points_list 

def run_exps_nd(curr_set, iters):
    n = len(curr_set[0])
    m = volume_of_convex_hull(curr_set)
    n_set = np.copy(curr_set)
    lengths_array = [len(curr_set)]
    poly = None
    k = -1

    for i in range(1, max(n,iters)+1):
        n_set = sum_sets(n_set, curr_set)
        lengths_array.append(len(n_set))
        if i >= n and k == -1:
            xs = np.arange(i-n+1,i+2)
            ys = lengths_array[-(n+1):]
            new_poly = np.polyfit(xs, ys, n)
            if poly is not None and np.allclose(new_poly,poly) and np.isclose(m,poly[0]):
                k = i-n
            else:
                poly = new_poly
    
    if k != -1:
        # stabilized (bool), stabilized index (int), khovanskii poly (int list), length array (int list), normalized volume (int)
        return True, k, np.round(poly,7), lengths_array, np.round(poly[0]*np.math.factorial(n), 5)
    else:
        return False, float('inf'), [], [], 0
    

def same_line(points):
    x0,y0  = points[0]
    points = [ (x,y) for x,y in points if x != x0 or y != y0 ]
    slopes = [ (y-y0)/(x-x0) if x!=x0 else None for x,y in points ]
    return all( s == slopes[0] for s in slopes)


def random_set(m, size):
    points = np.random.randint(0, m, size=(size-1, 2))
    points = np.unique(np.insert(points, 0, [0,0], axis=0), axis=0)
    while same_line(points):
        points = np.random.randint(0, m, size=(size, 2))
        points = np.unique(np.insert(points, 0, [0,0], axis=0), axis=0)
    return points


def write_to_csv(fname, rows, legend=('b' , 'c', 'k', 'A')):
    with open(fname, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(legend)
        writer.writerows(rows)

def read_rows_from_csv(fname):
    rtn = []
    with open(fname, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            rtn.append(row)
    return rtn

def get_SL_n_Z_gens(n):
    u1 = np.zeros((n,n), dtype=int)
    for i in range(n):
        if i !=  n-1:
            u1[i][i+1] = 1
        else:
            u1[i][0] = 1

    u2 = np.zeros((n,n), dtype=int)
    for i in range(n):
        u2[i][i] = 1
        if i ==  1:
            u2[i][0] = 1

    u3 = np.identity(n, dtype=int)
    u3[0][0] = -1

    u4 = np.identity(n, dtype=int)
    u4[[0, 1]] = u4[[1, 0]]

    return u1,u2,u3,u4

    


def primitive_triangle_basis(max_iterations, d=2):
    if d == 2:
        # generators for SL_2(Z)
        gens = [np.array([[1,1], [1,0]]), np.array([[0,-1], [1,0]])]
        # `word` is a random sequence of 0s and 1s of length <= max_iters
        iters = np.random.randint(1, max_iterations)
        word = np.random.randint(0,2, size=iters)
        # generate
        current = np.identity(2)
        for i in word:
            current = np.matmul(current, gens[i])
        # add third element to make it d+2
        a,b,c,d = current[0,0], current[1,0], current[0,1], current[1,1] 
        return [(0,0), (int(a),int(b)), (int(c),int(d))]
    else:
        # generators for SL_n(Z)
        gens = get_SL_n_Z_gens(d)
        # `word` is a random sequence of 0s and 1s of length <= max_iters
        iters = np.random.randint(1, max_iterations)
        word = np.random.randint(0,4, size=iters)
        # generate
        current = np.identity(d, dtype=int)
        for i in word:
            current = np.matmul(current, gens[i])

        rtn = [d*(0,)]
        for row in current.T:
            rtn.append(tuple(row))
        return rtn


def random_primitive_dPn(k, max_iterations, d=2):
    rtn = primitive_triangle_basis(max_iterations, d=d) 
    # # translate if not in quadrant I
    # xs, ys = list(zip(*rtn))
    # # vertically first
    # if np.min(ys) < 0:
    #     ymin_arg = np.argmin(ys)
    #     yoffset = - rtn[ymin_arg][1]
    #     rtn = [(e[0], e[1] + yoffset) for e in rtn]
    # #  then horizontally
    # if np.min(xs) < 0:
    #     xmin_arg = np.argmin(xs)
    #     xoffset = - rtn[xmin_arg][0]
    #     rtn = [(e[0] + xoffset, e[1]) for e in rtn]

    basis = rtn.copy()
    translations = []
    max_element_range = 20
    for _ in range(k-1):
        x = tuple(np.random.randint(-max_element_range // 2, max_element_range // 2, size=d))
        while x in rtn:
            x = tuple(np.random.randint(-max_element_range // 2, max_element_range // 2, size=d))
        translations.append(x)
        rtn.append(x)
    return rtn, basis, translations


def random_primitive_dPn_exps(k, maxx, maxk, iters):
    results = set()
    j=0
    for _, basis, translations in [random_primitive_dPn(k, maxx) for _ in range(iters)]:
        print(j)
        j += 1
        m, mm, b, c, k, _, _, _ =  run_exps(basis + translations, maxk)
        if m != mm:
            continue
        res = (m,b,c,k, tuple(sorted(basis)), tuple(sorted(translations)))
        results.add(res)
    return results
    
def binexp(x,y):
    h = symbols('h')
    expr = combsimp(binomial(h+4,4)-binomial(h-x+4,4)-binomial(h-y+4,4)+binomial(h-x-y+4,4))
    return expand(expr).evalf()


def binomial_expr(both_heights, alpha):
    positive_heights, negative_heights = both_heights
    h = symbols('h')
    pre_expr = binomial(h+alpha,alpha)
    for x in positive_heights:
        pre_expr += binomial(h+alpha-x, alpha)
    for y in negative_heights:
        pre_expr -= binomial(h+alpha-y, alpha)
    expr = pre_expr
    return expr.evalf()


def get_khovanskii_binomial(A, d, actual_poly_str, lengths_arry):
    h = symbols('h')
    k = len(A) - d
    both_heights = [[], []]

    actual_poly = sympify(actual_poly_str)
    expr = binomial_expr(both_heights, d+k-1)
    last = 0
    while(simplify(actual_poly - expr).replace(lambda x: x.is_Number and abs(x) < 1e-6, lambda x: 0) != 0):
        for i in range(last, len(lengths_arry)):
            diff = lengths_arry[i] - expr.evalf(subs={h:i+1})
            if diff > 0:
                both_heights[0].append(i+1)
                last = i
                break
            if diff < 0:
                both_heights[1].append(i+1)
                last = i
                break
        expr = binomial_expr(both_heights, d+k-1)
    
    both_heights[0].insert(0,0)
    return len(both_heights[0]) + len(both_heights[1]), both_heights

    

def satisfy_simple(basis, translations, maxk, c1=None, c2=None, m=None, b=None, c=None):
    if any(e is None for e in [m,b,c]):
        m, _, b, c, _, _, _, _ = run_exps(basis+translations, maxk)

    if c1 is None or c2 is None:
        inters = intersections(basis, translations, maxk)
        x, y = 0, 0
        for i in range(maxk):
            if len(inters[i][1]) != 0:
                x = i+1
                break
        # if the # of orange in inters[j][2] for j>=x is larger than the # of yellow in inters[j-x] then j contains a minimal element
        if len(inters[x-1][1]) > 1:
            y = x
        else:
            for j in range(x, maxk):
                if any(len(inters[j][i+1]) > len(inters[j-x][i]) for i in range(maxk-1)):
                    y = j+1
                    break
        
    x = c1 if c1 is not None else x
    y = c2 if c2 is not None else y

    expr = binexp(x,y)
    expected = sympify(f'{m}*h**2 + {b}*h + {c}')
    return simplify(expr - expected) == 0, x, y, expected, expr
    

def filter_dP3_satisfy_simple(basis_depth, maxk, iters):
    not_res = []
    res = []
    for m, b, c, k, basis, translations in random_primitive_dPn_exps(3, basis_depth, maxk, iters):
        basis = list(basis)
        translations = list(translations)
        satisfies, x, y, expected, actual = satisfy_simple(basis, translations, maxk, m=m, b=b, c=c)
        if not satisfies:
            actualstr = f'{actual}'
            expectedstr = f'{expected} for k >= {k}'
            not_res.append((expectedstr, actualstr, x, y, basis+translations))
        else:
            polystr = f'{expected} for k >= {k}'
            res.append((polystr, x, y, basis+translations))

    return res, not_res


def all_combinations_binexp(maxx):
    res = []
    for i in range(2, maxx):
        for j in range(i, maxx):
            res.append((binexp(i,j),i,j))
    return res


def view_plots_from_csv(fname, min_sets, max_sets=None):
    # filtered = []
    for i, rowstr in enumerate(read_rows_from_csv(fname)):
        if i == 0 or len(rowstr) == 0 or i < 2*min_sets:
            continue
        if max_sets is not None and i >= 2 * max_sets:
            break

        Astr = re.search('"(.*)"', rowstr[0]).group(1)
        A = ast.literal_eval(Astr)
        print(f'A = {A}')
        # print(get_khovanskii_binomial(A, 2))

    #     if len(get_hull_points(A)) == 3:
    #         filtered.append(rowstr)
    # write_to_csv('random_2d_exps/simplex_filter_dp3_simple_6_15_1000.csv', filtered,('Actual P(h)','Expected P(h)','c_1','c_2','A'))

        minimal_elements, inverse_minimal_elements = get_minimal_elements(A, 12)
        print(f'negative elementaries: {minimal_elements}')
        print(f'positive elementaries: {inverse_minimal_elements}')
        print(f'minimal elements: {get_michael_min_elements(A, 12)}')
        single_sumset(
            A,
            10,
            basis=A[:3],
            translations=A[3:],
            slice=(0,10),
            plot=True,
            show_intersections=True,
        )


def get_cones(A, iters):
    cones = []
    inters = intersections(A, iters)
    max_color = max(len(iteration) for iteration in inters)
    for h in range(max_color):
        cone = set()
        points = []
        # some heights might have darker colors than the height
        for iteration in inters:
            if h < len(iteration):
                points.append(iteration[h])
            else:
                points.append([])
        # add heights
        for i, iteration in enumerate(points):
            for j in range(len(iteration)):
                if len(iteration[j]) == 2:
                    cone.add((iteration[j][0], iteration[j][1], i+1))
                elif len(iteration[j]) == 3:
                    cone.add((iteration[j][0], iteration[j][1], iteration[j][2],i+1))
                else:
                    assert False
        cones.append(cone)
    return cones


def get_cone(A, iters):
    _, _, _, _, _, _, _, points_list = run_exps(A,iters)
    lifted_points_list = set([(e[0], e[1], i+1) for i in range(len(points_list)) for e in points_list[i]])
    lifted_points_list.add((0,0,0))
    return lifted_points_list


def separate_iters_in_cone(cone):
    l = max(e[2] for e in cone) + 1
    res = [[] for _ in range(l)]
    for e in cone:
        res[e[2]].append(e)
    return res


def get_hull_points(A):
    hull = ConvexHull(np.array(A))
    hull_points = []
    for v in hull.vertices:
        point = A[v]
        hull_points.append((point[0], point[1], 1))
    return sorted(hull_points)


def get_colors(cones):
    # computes the largest i for which cones[i] contains e
    def color(e):
        for i, cone in enumerate(cones):
            if e not in cone:
                return i-1
        return len(cones)-1

    colors_dict = dict()
    colors_dict[(0,0,0)] = 0
    for e in cones[0]:
        colors_dict[e] = color(e)
    return colors_dict


def get_michael_min_elements(A, iters, hull_points=None):
    coneA = get_cone(A, iters)
    if hull_points == None:
        hull_points = get_hull_points(A)
        cone_larger = coneA
    else:
        projected_hull_points = [(e[0], e[1]) for e in hull_points]
        larger = list(set(A).union(set(projected_hull_points)))
        cone_larger = get_cone(larger, iters)
    minimal_elements = []
    for element in coneA:
        if all(tuple(np.subtract(element, hull_point)) not in coneA for hull_point in hull_points):
                minimal_elements.append(element)
    return sorted(minimal_elements, key=lambda x: x[2])


def get_minimal_elements(A, iters):
    """
    Refactor this method in the following way to find deep minimal elements.
        0) Precompute len(iters) iterations of A
        1) Compute shallow minimal elements using defn from Khovanski paper and set M := shallow
        minimal elements.
        2) Iterate over elements one sumset iteration k at a time
        3) For each element e at height k, count the number of ways N_e one can get back to a
        minimal element (or inverse minimal element) (a_i,b_i,c_i) using elements from (k-c_i)A. 
        4) If (x,y,z) is such an element i.e., e - (x,y,z) = (a_i,b_i,c_i), then increment N_e
        by color(x,y,z) if it minimal or decrement if it inverse-minimal.
        5) Compare N_e the color(e). If color(e) == N, do nothing. 
        If color(e) < N then this point is an inverse minimal element. If color(e) > N then 
        this is a deep minimal element, so add it to M.
    """
    lift_of_A = [(e[0], e[1], 1) for e in A]
    cones = get_cones(A, iters)
    sums_of_A = sum_set_n_times_and_save(lift_of_A, iters)
    minimal_elements = []
    inverse_minimal_elements = []
    colors = get_colors(cones)

    for k, iteration in enumerate(sums_of_A):
        for e in iteration:
            e = tuple(e)
            # if e == (2,2,2):
            #     print(colors[e])
            N = 0
            for minimal_element in minimal_elements:
                for vertex in sums_of_A[k-minimal_element[2]]:
                    vertex = tuple(vertex)
                    if tuple(np.subtract(e, vertex)) == minimal_element:
                        N += colors[vertex] + 1
            for inverse_minimal_element in inverse_minimal_elements:
                for vertex in sums_of_A[k-inverse_minimal_element[2]]:
                    vertex = tuple(vertex)
                    if tuple(np.subtract(e, vertex)) == inverse_minimal_element:
                        N -= colors[vertex] + 1
            difference = N - colors[e]
            if difference < 0:
                minimal_elements += ([e] * (-1 * difference))
            elif difference > 0:
                inverse_minimal_elements += ([e] * difference)
    
    return minimal_elements, inverse_minimal_elements


def get_negative_elementaries_quick(A, iters):
    rtn = []
    cones = get_cones(A,iters)
    elements = cones[0]
    duplicates = cones[1]
    duplicates_prefix_by_height = [[] for _ in range(iters+1)]

    for e in duplicates:
        height = e[-1]
        for i in range(height, iters+1):
            duplicates_prefix_by_height[i].append(e)

    for e1 in duplicates: 
        height = e1[-1]
        if all(tuple(np.subtract(e1, e2)) not in elements for e2 in duplicates_prefix_by_height[height-1]):
            rtn.append(e1)
    return rtn


def exps_get_r_fixed_d_k(max_basis_iters=15, d=2, k=3, no_samples=100):
    res = []
    for i in range(no_samples):
        A = random_primitive_dPn(k, max_iterations=max_basis_iters, d=d)[0]
        m, mm, _, _, _, _, _, _ =  run_exps(A, 10)
        assert m == mm
        b = get_khovanskii_binomial(A, d=d)
        entry = (m, b[0], b[0]/m)
        res.append((entry, A))
        print(i)
        print(A)
        print(entry)
        print()
    print(list(zip(*res))[0])
    print(max(res, key=lambda x: x[0][2]))
 

def fix_d_vary_k(max_basis_iters=8, d=2, k_range=(3,4), no_samples=100):
    if d != 2:
        assert False

    res = []
    for h in range(*k_range):
        l = []
        for i in range(no_samples):
            print(i)
            A = random_primitive_dPn(h, max_iterations=max_basis_iters, d=d)[0]
            m, mm, bb, c, k, _, lengths_arry, _ =  run_exps(A, 30)
            b = get_khovanskii_binomial(A, d=d)
            max1 = max(b[1][0])
            max2 = max(b[1][1]) if len(b[1][1]) != 0 else 0


            if max(max1, max2) < d*m:
                continue
            print(A)
            print(f'p(x) = ({m}) {mm}x^2 + {bb}x + {c}  for all k >= {k}')
            print(lengths_arry)
            print(b)
            print()
            l.append(b[0])
        if len(l) != 0:
            avg = sum(l) / len(l)
            res.insert(0, avg)
        res.append(l)
    return res


def min_height(B, elems, iters):
    maxx = 10 
    heights = set([e[-1] for e in elems])
    for i in range(1, maxx+1):
        for a in range(i+1):
            for b in range(i+1):
                for c in range(i+1):
                    for d in range(i+1):
                        for e in range(i+1):
                            A = [(B[0][0], B[0][1], a), (B[1][0], B[1][1], b),
                                 (B[2][0], B[2][1], c), (B[3][0], B[3][1], d),
                                 (B[4][0], B[4][1], e)]
                            try:
                                vol = volume_of_convex_hull(A)
                            except:
                                continue

                            for value in heights:
                                norm_vol = vol*6
                                if np.isclose(norm_vol, value):
                                    status, _, _, _, m = run_exps_nd(A, iters)
                                    if status and np.isclose(norm_vol,m): 
                                        heights.remove(value)
                                        break

                            if len(heights) == 0:
                                return i
                            

def lifting_conj(rounds=1000, max_iter=38):
    counts_number_elems = dict()
    counts_min_height = dict()

    for i in range(rounds):
        print(f'iteration {i}')

        # get random 3-element basis in 2D
        basis = primitive_triangle_basis(10,d=2)
        # generate points in the positive quadrant that are not the basis
        k_temp = [max(e) for e in basis]
        k = min(k_temp)+1
        x1,x2,y1,y2 = np.random.randint(k,k+18,size=4)
        A = basis+[(x1,x2),(y1,y2)]

        status, thresh, _, _, m = run_exps_nd(A, max_iter)
        if not status:
            print(f"did not stabilize under {max_iter}")
            print()
            continue
        print(f'A: {A}, k: {thresh}, vol: {m}')

        if len(set(A)) != len(A):
            print(f"has a duplicate in construction")
            print()
            continue

        elems = get_negative_elementaries_quick(A, thresh+3)

        if len(elems) == 1:
            print(f"bug of only one minimal duplicate")
            print()
            continue

        if len(elems) in counts_number_elems:
            counts_number_elems[len(elems)] += 1
        else:
            counts_number_elems[len(elems)] = 1

        max_height_min_dup = max(elems, key= lambda x: x[-1])
        max_height = max_height_min_dup[-1]
        print(f'max height min dup: {max_height}')
        if max_height >= m:
            print("broke conj 1")

        minh = min_height(A, elems, thresh+3)
        if minh == None:
            print(f"broke conj 2")
            minh = -1

        if minh in counts_min_height:
            counts_min_height[minh] += 1
        else:
            counts_min_height[minh] = 1

        if (minh > 3 or len(elems) > 4) and minh != -1:
            print(f'number of min duplicates: {len(elems)}')
            print(f'min lift height required: {minh}')

        print()
        
    print(f'number of min elems counts: {counts_number_elems}')
    print(f'min lift height needed counts: {counts_min_height}')
        


lifting_conj()

#TODO: bug? following set allegedly has one min element at h=2 but an elementary at h=27
# A = [(0, 0), (3, 2), (-5, -3), (8, 2), (-2, 2)]


# max_iters = 38 
# dimension = 2
# temp_set = [
#     ]

# for new_A in temp_set:
#     print(new_A)
#     rtn = single_sumset(new_A, max_iters, print_info=True)
#     if rtn == False:
#         continue
#     actual_poly_str, lengths_arry = rtn
#     print(get_khovanskii_binomial(new_A, dimension, actual_poly_str, lengths_arry))
#     elems = get_negative_elementaries_quick(new_A, max_iters)
#     print(elems)
#     print(min_height(new_A, elems, max_iters))
#     print()


"""
TODO: Q0: Why are there some lifts with unique minimal elements that do not correspond to minimal duplicates
in the projected set? We know that every duplicate in the lift is a duplicate in the projection. But not
necessarily a minimal duplicate from the examples we've seen. Is there a way to guarantee is is minimal in
the projection?

TODO: Q1: how large does the volume of the lift have to be? How far up must we lift the points?
Conjecture: Let r denote the number of minimal duplicates. Is there a bound on the max height of any lifted
point based on the parameter r?
-> Yes for case of r=2 since can show direclty that h_1*h_2 = vol*d! / (|A|-d-1), which is stronger
-> Not obvious from using explicit formular for any r > 2. Need to do more experiements on the max height
of lifted point while varying r. 

Granville et. al are able to get a universal boujd using just the language of linear algebra and nothing else.

TODO: Q2: If can understand the maximal height needed for a lift based on parameter r, then next question to ask is:
does there exists a bound on r based on the size of the set A, or maybe the polyhedral complexity (number of facets)
i.e. how far it is from being a simplex?

TODO: Q3: is there a bound on the magnitude of the Khovanskii polynomial coefficients based on the number or
 max height of the minmal duplicates? Maybe try equating coefficients of explicit formula for the non-leading term

TODO: Q4: any connection with resolution of singulartities for Veronese varieties? Going up dimensions removes duplicates
i.e. resolves singularities in the algebraic space. Is there work that says how far have to go up in magnitude or dimension
to resolve a singularity. Can obtain bound like that.

#TODO: see if number of minimal duplicates is bounded if simplex

Conjecture 1: if h_1,...,h_m are the distinct heights of minimal duplicates, then h_1 <= vol(A)*d!

Conjecture 2:  max height needed for any lift is |A|*(d+1) So in this case it is 10. Which bounds the max height of minimal
duplicate to vol(A)*(d+1)!*|A|
TODO: run systematic exps written on phone to support or disprove this

TODO: proof incorrect as what if there are multiple minimal duplicates at the same height? change wording of theorem to say
"one of each height"
"""