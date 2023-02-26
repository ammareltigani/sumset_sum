# from math import gcd
# from functools import reduce
from scipy.spatial import ConvexHull, convex_hull_plot_2d
# from itertools import combinations
# from scipy import stats

# import csv
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
    return np.unique(np.array([np.add(e1, e2) for e1 in set1 for e2 in set2]), axis=0)


def single_sumset(A, iterations=None, plot=False):
    print(f'A = {A}')
    A = process(A) if type(A) == list else A
    m, mm, b, c, k, nA, lenghts_arry, points_arry =  run_exps(A, iterations)
    print(f'p(x) = ({m}) {mm}x^2 + {b}x + {c}  for all k >= {k}\n')
    print(f'Lengths Array: {lenghts_arry}')
    if plot:
        if iterations is None:
            # plot A and nA
            _, axis = plt.subplots(2)

            hull = ConvexHull(A)
            points = np.array(A)
            axis[0].plot(points[:,0], points[:,1], 'o')
            for simplex in hull.simplices:
                axis[0].plot(points[simplex, 0], points[simplex, 1], 'k-')
            axis[0].set_title("A")

            hull = ConvexHull(nA)
            points = np.array(nA)
            axis[1].plot(points[:,0], points[:,1], 'o')
            for simplex in hull.simplices:
                axis[1].plot(points[simplex, 0], points[simplex, 1], 'k-')
            axis[1].set_title("nA")
        else:
            a = int(np.ceil(iterations / 2))
            _, axis = plt.subplots(a,2)
            for j in range(iterations):
                m = 1 if j % 2 == 1 else 0
                n = j // 2
                hull = ConvexHull(points_arry[j])
                points = np.array(points_arry[j])
                axis[n,m].plot(points[:,0], points[:,1], 'o')
                for simplex in hull.simplices:
                    axis[n,m].plot(points[simplex, 0], points[simplex, 1], 'k-')
                axis[n,m].set_title("A")
        # elif iterations <= 3:
    
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


def run_exps(curr_set, iterations):
    m = volume_of_convex_hull(curr_set)
    n_set = np.copy(curr_set)
    last_three = [np.copy(n_set)]
    lengths_arry = [len(n_set)]
    k = None
    real_m = None
    i = 0
    # while True and (iterations is not None and i < iterations):
    while True:
        if iterations is not None and i >= iterations+3:
            break

        n_set = sum_sets(n_set, curr_set)
        if i >= 3:
            d11 = lengths_arry[i-2] - lengths_arry[i-3]
            d12 = lengths_arry[i-1] - lengths_arry[i-2]
            d13 = lengths_arry[i] - lengths_arry[i-1]
            d21 = d12 - d11
            d22 = d13 - d12
            if d22 - d21 == 0:
                if d21 != 2*m:
                    real_m = d21 / 2
                if k is None:
                    k = i-2
                if iterations is None:
                    break
        last_three.append(n_set)
        lengths_arry.append(len(n_set))
        i+=1

    assert k is not None
    mm = m if real_m is None else real_m 

    # Calculate b and c
    b = (lengths_arry[k] - lengths_arry[k-1]) - (mm*((2*(k))+1))
    c = lengths_arry[k-1] - (mm*((k)**2)) - (b*(k))
                
    return  m, mm, b, c, k, deprocess(last_three[-3]), lengths_arry, last_three 


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

def random_set_exp():
    #TODO: get random and check if primitive, if yes then proceed to plot, else throw out and
    # try again.
    pass


def random_primitive_triangles():
    generator = np.array([[1,1], [1,0]])
    current = generator.copy()
    for _ in range(15):
        a,b,c,d = current[0,0], current[1,0], current[0,1], current[1,1] 
        single_sumset([(0,0), (a,b), (c,d)], iterations=10, plot=False)
        current = np.matmul(current, generator)






# single_sumset([(0,0), (1,0), (0,1)], iterations=50, plot=True)
# single_sumset(random_set(20,3), iterations=50, plot=True)
# random_primitive_triangles()

single_sumset([(0,0), (1,0), (0,1)], iterations=4, plot=True)
# single_sumset([(0,0), (2,0), (0,2), (1,0), (0,1), (1,1)], iterations=4, plot=True)
single_sumset([(0,0), (3,0), (0,3), (1,0), (0,1)], iterations=4, plot=True)



"""
I - Take 3rd derivate instead 2nd to check where it stabilizes

(1) consider both the case where the second derivative is twice the volume and
when it is some c time the volume. Try to understand that this c is whenever
we scale all the points of A by some constant factor or perform other operations. 
In this respect, we are trying to understand what a simple rule of converting polygones
that don't have their volume = the leading coefficients into other that do and that
also have the same size polynomial 
(want to convert sets that are not primitive into those that are).

-- The constant is basically the product of proportion of points you miss in each dimension.
-- For triangles the rule is simple. Before describing it, note that given a non-primitive set 
A = {(0,0), (a,b), (c,d)}, then A-A generates Z^2 iff the matrix ((a,c) (b,d)) has determinant 1 
(see scrap notes for proof). Suppose we want to keep (a,b), if gcd(a,b) = 1 then we can always find (c,d) and ad-bc=1 (which
will also imply gcd(c,d)=1 btw). Then we try to keep (c,d) and this works by changing (a,b) this time
so long as gcd(c,d) = 1. If both gcd(a,b) and gcd(c,d) are not 1 then we cannot keep either point. One thing
we can do in this case is keep the diagonal terms a and d and replace the others with a matrix like
((a, d-1), (1, d)) (or any of its 3 other variations).

#TODO: look at the polynomials of such "substitute matrices" for those that are not primitve and 
see if they are any different or have any patterns etc.
--> Conj1: all primitive triangles with no points inside the hull have the same polynomial. See `random_primitive_triangles()`
So all "substitute matrices" for degenerate triangles have the same poly and denerate matrices have
the same poly up to some constant factor (proportion of missed points in each dimension multiplied by each other)

#TODO: what if I add things in the hull of the triangle
--> Conj2: all primitive 3-element sets don't have an integer point in their hull (can prove this)
So cannot add anything in the hull of a primitive 3-element set.

I -  What if we add points to any primitive triangle (with more than 3 elements)?

II - What about for quadrilaterals, or n-gons? Carefully come up with hand selected examples of simple primiive sets
to look at. Start with primitive quadrilaterals (maybe classify them too) and move on to 5-gon, etc.

The center of mass of the convex hull, all the points, the interior points only affects the distribution
of points as we iterate the sumset. Start with a triangle and then start placing 1,2,3,... points inside. Move
on to quadrilaterals, etc.
Specifically keep an eye out for self-similar behavior on the fringes and notice how the core grows. (if 
the set if not primitive the core will have some holes and the fringes will have other behavior too)

III - Generate random sets by simply throwing it out and trying again if the second derivative
is not twice the volume. 


"""
