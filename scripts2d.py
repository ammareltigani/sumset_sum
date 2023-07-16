from scipy.spatial import ConvexHull
from sympy import *

import ast
import csv
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


def mult_tuple(e, times):
    return tuple(e[i] * times for i in range(len(e)))

def intersections(basis, translations, iters):
    rtn = []
    # get all dilations
    basis_dilations = [[(0,0)], basis]
    curr_basis = basis.copy()
    for _ in range(iters):
        curr_basis = sum_sets(curr_basis, basis)
        basis_dilations.append(curr_basis)

    for h in range(iters):
        points_list = []
        # i'th dilation
        for i in range(h+1,-1,-1):
            dilation = basis_dilations[i]
            # want all partitions of i. So j*1st trans, and (i-j)*2nd trans
            if len(translations) == 2:
                for j in range(h+1-i, -1, -1):
                    tdilation = [(e[0] + j * translations[0][0], e[1] + j * translations[0][1]) for e in dilation]
                    tdilation = [(e[0] + (h+1-i-j) * translations[1][0], e[1] + (h+1-i-j) * translations[1][1]) for e in tdilation]
                    points_list.extend(tdilation)
            elif len(translations) == 1:
                tdilation = [(e[0] + (h+1-i) * translations[0][0], e[1] + (h+1-i) * translations[0][1]) for e in dilation]
                points_list.extend(tdilation)
            else:
                # Not implemented yet for d+4
                assert False

        unique, counts = np.unique(np.array(points_list), axis=0, return_counts=True)
        l = max(iters, max(counts))
        all_intersections = [[] for _ in range(l)]
        for j in range(l):
            all_intersections[j] = unique[counts > j]
        
        rtn.append(all_intersections)

    return rtn


def single_sumset(A, iterations, slice=None, plot=False, show_intersections=False, basis=None, translations=None):
    if show_intersections is not False:
        assert basis is not None
        assert translations is not None
        assert iterations is not None

    # print(f'A = {A}')
    A = process(A) if type(A) == list else A
    m, mm, b, c, k, _, lenghts_arry, points_arry =  run_exps(A, iterations)
    # print(f'p(x) = ({m}) {mm}x^2 + {b}x + {c}  for all k >= {k}')
    # print(f'Lengths Array: {lenghts_arry}\n')

    if plot:
        assert slice is not None
        if show_intersections:
            inters = intersections(basis, translations, iterations)
            plot_sumset(
                iterations,
                slice,
                inters=inters,
            )
        else:
            plot_sumset(
                iterations,
                slice,
                points_arry=points_arry,
            )
    
    return f'{mm}*h^2 + {b}*h + {c}', lenghts_arry
            


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
    real_m = None
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

            if d31 == 0:
                if once:
                    if d21 != 2*m:
                        real_m = d21 / 2
                    k = i-3
                else:
                    once = True
            else:
                once = False
        points_list.append(n_set)
        lengths_arry.append(len(n_set))
        i+=1

    assert k is not None
    mm = m if real_m is None else real_m 
    # if mm != m:
    #     print(mm, m)
    #     assert False

    # Calculate b and c
    b = (lengths_arry[k] - lengths_arry[k-1]) - (mm*((2*(k))+1))
    c = lengths_arry[k-1] - (mm*((k)**2)) - (b*(k))

                
    return  m, mm, b, c, k, deprocess(points_list[-3]), lengths_arry, points_list 


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
    max_element_range = 10
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


def get_khovanskii_binomial(A, d):
    h = symbols('h')
    k = len(A) - d
    both_heights = [[], []]
    actual_poly_str, lengths_arry = single_sumset(A, 30)
    actual_poly = sympify(actual_poly_str)
    expr = binomial_expr(both_heights, d+k-1)
    last = 0
    while(simplify(actual_poly - expr).replace(lambda x: x.is_Number and abs(x) < 1e-8, lambda x: 0) != 0):
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
        print(get_khovanskii_binomial(A, 2))
        # print(satisfy_simple(A[:3], A[3:], 12))

    #     if len(get_hull_points(A)) == 3:
    #         filtered.append(rowstr)
    # write_to_csv('random_2d_exps/simplex_filter_dp3_not_simple_6_15_1000.csv', filtered,('Actual P(h)','Expected P(h)','c_1','c_2','A'))
        # minimal_elements, inverse_minimal_elements = get_minimal_elements(A, 12)
        # print(f'deep minimal elements: {minimal_elements}')
        # print(f'inverse minimal elements: {inverse_minimal_elements}')
        # single_sumset(
        #     A,
        #     10,
        #     basis=A[:3],
        #     translations=A[3:],
        #     slice=(0,10),
        #     plot=True,
        #     show_intersections=True,
        # )




def get_cones(A, iters):
    cones = []
    inters = intersections(A[:3], A[3:], iters)
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
                cone.add((iteration[j][0], iteration[j][1], i+1))
        cones.append(cone)

    return cones


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
    

#TODO: Is there any geometry for the deep minimal elements or the inverse minimal elements?
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
            color_e = colors[e]
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
            difference = N - color_e
            if difference < 0:
                minimal_elements += ([e] * (-1 * difference))
            elif difference > 0:
                inverse_minimal_elements += ([e] * difference)
    
    return minimal_elements, inverse_minimal_elements



def fix_d_vary_k(max_basis_iters=5, d=2, k_range=(2,20), no_samples=50):
    if d != 2:
        assert False

    res = []
    for h in range(*k_range):
        l = []
        for i in range(no_samples):
            print(i)
            A = random_primitive_dPn(h, max_iterations=max_basis_iters, d=d)[0]
            m, mm, bb, c, k, _, lengths_arry, _ =  run_exps(A, 30)
            if m != mm:
                print(f'the set A={A} does not compute correctly in run_exps()\n')
                continue
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
        avg = sum(l) / len(l)
        res.insert(0, avg)
        res.append(l)
    return res

"""

------------------------TODOS----------------------------
#TODO: check if the argument that bound the height minimal elements for d-simplices in Michael paper can be adopted to bound
# the height of elementaries.
#TODO: look at specific (simple) example of a set that stabilizes before its largest h_i (so, almost all sets)
#TODO: run_exps() does not calculate the correct khovanskii polynomial for [(0, 0), (1, 1), (1, 0), (-4, 2), (3, 3), (-5, 1), (-2, 3)] 
# Investigate why and fix it.
#TODO: implement the ability to run_exps for d=3 and run r-experiments in 3d as well as 2d
#TODO: PROVE lemma 4.3, Theorem 4.4, and conjecture 4.6. Think of how to do wihtout resorting to simplical case


------------------------Workspace----------------------------

"""
# nice sets that hve more than 2 minimal elements under our filteration method
# sets1 = [[(0, 0), (1, 0), (1, 1), (0, 2), (3, 3)],
#         [(0, 0), (1, 1), (2, 1), (5, 3), (9, 1)],
#         [(0, 0), (1, 0), (1, 1), (1, 9), (7, 4)],
#         [(0, 0), (1, 0), (1, 1), (1, 4), (9, 1)],
#         [(0, 0), (0, 1), (1, 1), (5, 1), (6, 9)],
#         [(0, 0), (1, 0), (1, 1), (1, 7), (7, 6)]]

sets5 = [
    # [(0, 0), (0, 1), (-1, 0), (-5, -5), (-4, -4)], # counterexample to conj 4.6
    # [(0, 0), (-2, -1), (-1, -1), (-4, 0), (1, 0)], # counterexample to conj 4.6
    # [(0, 0), (1, 1), (1, 0), (0, 1), (3, 1)], # counterexample to conj 4.6
    # [(0, 0), (1, 1), (2, 1), (7, 8), (7, 9)],
    ]


# for A in sets5:
    # print(get_minimal_elements(A, 10))
    # print(get_khovanskii_binomial(A, 2))
    # single_sumset(
    #     A,
    #     10,
    #     basis=A[:3],
    #     translations=A[3:],
    #     slice=(0,8),
    #     plot=True,
    #     show_intersections=True,
    # )


print(fix_d_vary_k(max_basis_iters=7, d=2, no_samples=1000, k_range=(3,5)))


# view_plots_from_csv('random_2d_exps/filter_dP3_not_simple_6_15_1000_sorted.csv', 20, 30)
# view_plots_from_csv('random_2d_exps/filter_dP3_not_simple_6_15_1000.csv', 0, 30)
# view_plots_from_csv('random_2d_exps/filter_dP3_simple_6_15_1000_sorted.csv', 0, 30)
# view_plots_from_csv('random_2d_exps/simplex_filter_dp3_not_simple_6_15_1000.csv', 2)

# write_to_csv(f'random_2d_exps/privimite_{2+n}gons_{maxx}_{iters}.csv', random_primitive_dPn_exps(n,maxx,iters))
# res, not_res = filter_dP3_satisfy_simple(6, 20, 1000)
# write_to_csv('random_2d_exps/filter_dP3_not_simple_6_15_1000.csv', not_res, legend=('Actual P(h)', 'Expected P(h)', 'c_1', 'c_2', 'A'))
# write_to_csv('random_2d_exps/filter_dP3_simple_6_15_1000.csv', res, legend=('P(h)', 'c_1', 'c_2', 'A'))
# write_to_csv('random_2d_exps/all_comb_binexp_30.csv', all_combinations_binexp(30), legend=('P(x)', 'c1', 'c2'))
