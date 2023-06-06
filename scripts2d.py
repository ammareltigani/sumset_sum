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

    print(f'A = {A}')
    A = process(A) if type(A) == list else A
    m, mm, b, c, k, _, lenghts_arry, points_arry =  run_exps(A, iterations)
    print(f'p(x) = ({m}) {mm}x^2 + {b}x + {c}  for all k >= {k}')
    print(f'Lengths Array: {lenghts_arry}\n')

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

def primitive_triangle_basis(max_iterations):
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


def random_primitive_dPn(n, max_iterations):
    rtn = primitive_triangle_basis(max_iterations) 
    # translate if not in quadrant I
    xs, ys = list(zip(*rtn))
    # vertically first
    if np.min(ys) < 0:
        ymin_arg = np.argmin(ys)
        yoffset = - rtn[ymin_arg][1]
        rtn = [(e[0], e[1] + yoffset) for e in rtn]
    #  then horixontally
    if np.min(xs) < 0:
        xmin_arg = np.argmin(xs)
        xoffset = - rtn[xmin_arg][0]
        rtn = [(e[0] + xoffset, e[1]) for e in rtn]

    basis = rtn.copy()
    translations = []
    max_element_range = 10
    for _ in range(n-1):
        x = tuple(np.random.randint(0, max_element_range, size=2))
        while x in rtn:
            x = tuple(np.random.randint(0, max_element_range, size=2))
        translations.append(x)
        rtn.append(x)
    return rtn, basis, translations


def random_primitive_dPn_exps(n, maxx, maxk, iters):
    results = set()
    j=0
    for _, basis, translations in [random_primitive_dPn(n, maxx) for _ in range(iters)]:
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


def view_plots_from_csv(fname, min_sets, max_sets):
    for i, rowstr in enumerate(read_rows_from_csv(fname)):
        if i == 0 or len(rowstr) == 0 or i < 2*min_sets:
            continue
        if i >= 2 * max_sets:
            break

        Astr = re.search('"(.*)"', rowstr[0]).group(1)
        A = ast.literal_eval(Astr)
        print(f'A = {A}')
        print(satisfy_simple(A[:3], A[3:], 16))
        print(f'deep minimal elements: {get_minimal_elements(A, 16)}')
        single_sumset(
            A,
            11,
            basis=A[:3],
            translations=A[3:],
            slice=(0,10),
            plot=True,
            show_intersections=True,
        )


def magnitue_grows_with_linear_term_dP2():
    for i in range(20):
        A, _ = primitive_triangle_basis(6)
        print(f"\ni = {i}")
        for j in range(8):
            print(single_sumset(A + [(j+3, j+3)]))


def get_cones(A, iters):
    cones = []
    inters = intersections(A[:3], A[3:], iters)
    # keep only degree of intersection that we care about
    for deg in range(iters):
        points = [iteration[deg] for iteration in inters]
        cone = set()
        # add heights
        for i, iteration in enumerate(points):
            for j in range(len(iteration)):
                # print(iteration[j])
                cone.add((iteration[j][0], iteration[j][1], i+1))
        cones.append(cone)
    return cones

def get_hull_points(A):
    points = np.array(A)
    hull_points = set()
    hull = ConvexHull(A)
    for simplex in hull.simplices:
        point = (points[simplex, 0][0], points[simplex, 1][0], 1)
        hull_points.add(point)
    hull_points.add((0,0,1))
    return [(e[0], e[1], 1) for e in A]
    # return hull_points

def get_minimal_elements(A, iters):
    all_minimal_elements = [[]]
    cones = get_cones(A, iters)
    hull_points = list(get_hull_points(A))
    hull_set = set(hull_points)
    print(f'hull points: {hull_points}')
    for i in range(1, len(cones)):
        cone = cones[i]
        minimal_elements = []
        for e in cone:
            if all(tuple(np.subtract(e, vertex)) not in cone for vertex in hull_points):
                minimal_elements.append(e)
        all_minimal_elements.append(sorted(minimal_elements, key=lambda x: x[2]))
    print(f'standard min elems: {all_minimal_elements[1]}')

    all_filtered_minimal_elements = [all_minimal_elements[1]]
    for i in range(2, len(all_minimal_elements)):
        minimal_elements = all_minimal_elements[i]
        filtered_minimal_elements = []
        for e in minimal_elements:
            # print(e)
            # given definition of minimal element from paper
            not_self_similar = all(tuple(np.subtract(e,vertex)) not in cones[i] for vertex in hull_set)

            # generalize to multi-color by filtering similar minimal elements across colors
            prev_translates = [j for i in [e for e in all_minimal_elements[:i]] for j in i]
            prev_base = set(cones[i-1]).union(set(hull_points)).union(set(prev_translates))
            not_self_similar = all(tuple(np.subtract(e,vertex)) not in prev_base for vertex in prev_translates) and not_self_similar

            # similarity_set = set([min_elem for min_list in all_filtered_minimal_elements for min_elem in min_list])
            # similarity_set = all_comb(similarity_set, e[2] - sim[2] - 1)
            # print(f'similarity set {similarity_set}')
            # not_self_similar = all(tuple(np.subtract(e,vertex)) not in cones[i] for vertex in hull_set)
            # print(not_self_similar)
            # not_self_similar = all(tuple(np.subtract(e,vertex)) not in cones[i-1] for vertex in similarity_set) and not_self_similar
            # print(not_self_similar)


            if not_self_similar:
                filtered_minimal_elements.append(e)
        
        all_filtered_minimal_elements.append(sorted(filtered_minimal_elements, key=lambda x: x[2]))


    return all_filtered_minimal_elements




"""------------------------Conjectures----------------------------"""

# Conj1: For every d+3 elements set A, p(x) = BinExp(c1,c2) for some c1,c2 >= 2 where
# BinExp(c1,c2) = choose(h+4,4)-choose(h-c1+4,4)-choose(h-c2+4,4)+choose(h-c1-c2+4,4) 
# iff A has exactly 2 minimal elements iff the heights of the first two minimal elements
# product to the normalized volume of the convex hull of A

# Conj2: Every set can be written as p(x) = choose(x+d+2, d+2) + sum_{i \in I} choose(x-i+d+2,d+2) + sum_{j>max(I)} a_j choose(x-j+d+2,d+2) 
# where each I is the set of heights of the generalized minimal elements of A

# TODO: how does this generalize past sets of size d+3?
# Fact: 100% of sets with d+2 elements have exactly 1 minimal element
# Conj3: 25% of sets with d+3 elements have exactly 2 minimal elements (i.e. satisfy BinExp form)
# Conj4: In d dimensions if A has exactly d minimal elements then there is a generalized BinExp (continue the
# inclusion exclusion) that describes the size of hA for sufficiently large h

# Conj5: d-simplices have exactly d minimal elements.
# FALSE. Counterexample: [(0, 0), (2, 1), (1, 1), (1, 3), (3, 0)]

# Conjecture: only the first three 3 deep minimal elements matter in the binomial formula for non-nice sets
# (i.e. non-nice sets that agree with their first 3 minimal elements have the same khovanskii polynomial)
# FALSE. see ipad p.109 

"""------------------------Tasks----------------------------"""
# it seems like the original definition of minimal elements matches with 0th degree minimal elements but misses
# the minimal elements that arise in further degree intersections. the original definition is sometimes not enough
# to find the first two (any degree) minimal elements that produce BinExp in the case of a nice set (i.e there are nice
# sets with the second minimal element not in the 0th degree intersection)

# also, some nonzero degree minimal elements with the current filtering method for nice sets that are not the first two still
# seem to be appearing (after filtering). e.g. 
# But they are obvsiousely not relevant to the BinExp form as this is a nice set
# (nice sets only depend on the heigh of the first two deep minmal elements). Do they become relevant in the case the set
# is not nice? A 'generalized' minimal element should resolve this issue

# there is also the case where a single minimal element has multiplicity 2. e.g. [(0, 0, 1), (0, 1, 1), (1, 1, 1), (2, 1, 1), (2, 2, 1)]
# sets like these are considered nice though, it is just that the code that selects the first two minimal elements cannot
# account for multipicities, so need to update it.

# TODO: develop a definition for a 'generalized' minimal element that resolves all these issues.
# TODO: try to generalize conjectures 1 and 2 to d+n?
# TODO: try to improve bound on minimal elements by excluding the family of sets that give same height minimal
# elements in d=1

# BIG TODO: PROVE CONJECTURES 1 and 2 in the general case. Think of how to do wihtout resorting to simplical case


"""------------------------Workspace----------------------------"""
# nice sets that hve more than 2 minimal elements under our filteration method
# sets1 = [[(0, 0), (1, 0), (1, 1), (0, 2), (3, 3)],
#         [(0, 0), (1, 1), (2, 1), (5, 3), (9, 1)],
#         [(0, 0), (1, 0), (1, 1), (1, 9), (7, 4)],
#         [(0, 0), (1, 0), (1, 1), (1, 4), (9, 1)],
#         [(0, 0), (0, 1), (1, 1), (5, 1), (6, 9)],
#         [(0, 0), (1, 0), (1, 1), (1, 7), (7, 6)]]

sets2 = [[(0, 0), (0, 1), (1, 1), (6, 2), (6, 3)],
        [(0, 0), (1, 0), (1, 1), (0, 2), (3, 3)]]


for A in sets2:
    print(get_minimal_elements(A, 16))
    print(satisfy_simple(A[:3], A[3:], 16))
    single_sumset(
        A,
        16,
        basis=A[:3],
        translations=A[3:],
        slice=(0,8),
        plot=True,
        show_intersections=True,
    )


# view_plots_from_csv('random_2d_exps/filter_dP3_not_simple_6_15_1000_sorted.csv', 10, 21)
# view_plots_from_csv('random_2d_exps/filter_dP3_simple_6_15_1000.csv', 0, 30)

# write_to_csv(f'random_2d_exps/privimite_{2+n}gons_{maxx}_{iters}.csv', random_primitive_dPn_exps(n,maxx,iters))
# res, not_res = filter_dP3_satisfy_simple(6, 20, 1000)
# write_to_csv('random_2d_exps/filter_dP3_not_simple_6_15_1000.csv', not_res, legend=('Actual P(h)', 'Expected P(h)', 'c_1', 'c_2', 'A'))
# write_to_csv('random_2d_exps/filter_dP3_simple_6_15_1000.csv', res, legend=('P(h)', 'c_1', 'c_2', 'A'))
# write_to_csv('random_2d_exps/all_comb_binexp_30.csv', all_combinations_binexp(30), legend=('P(x)', 'c1', 'c2'))
