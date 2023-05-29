from math import gcd
from functools import reduce
from itertools import combinations
from scipy import stats

import csv
import numpy as np
import matplotlib.pyplot as plt

# This is a work in progress playground to run some experiments on the sumset problem
# as part of my senior thesis project.

def sum_sets(set1, set2):
    return set((x+y for x in set1 for y in set2))

def random_set(m, size):
    num_list = np.random.randint(0, m, size=size)
    num_set = set(num_list)
    while len(num_set) < 2:
        num_list = np.random.randint(0, m, size=size)
        num_set = set(num_list)

    rtn = normalize_set(num_set)
    rtn.add(m)
    return rtn

def normalize_set(input_set):
    min_elem = min(input_set)
    rtn = {e - min_elem for e in input_set}
    hcf = find_gcd(rtn)
    assert hcf != 0
    return {e // hcf for e in rtn}

def find_gcd(s):
    x = reduce(gcd, s)
    return x

def all_subsets_of_size_k_plus_one(m,k):
    origin_set = set(range(1,m+1))
    all_subsets = combinations(origin_set, k)
    return [set([0]).union(subset) for subset in all_subsets if find_gcd(subset) == 1]
        
def all_subsets(m):
    return [all_subsets_of_size_k_plus_one(m,k) for k in range(1,m+1)]

def run_exps(sets, show_steps=False):
    res_data = []
    for curr_set in sets:
        m = max(curr_set)
        results = [curr_set]
        n_set = curr_set
        double_flag = False
        while True:
            prev_size = len(n_set)
            n_set = sum_sets(n_set, curr_set)
            results.append(n_set)
            curr_diff = len(n_set) - prev_size
            if show_steps:
                print(n_set)
                print(f'difference: {m - curr_diff}')
            if curr_diff == m:
                if double_flag:
                    break
                else:
                    double_flag = True
        assert len(sum_sets(n_set, curr_set)) - len(n_set) == m

        size_A = len(curr_set)
        # predicted_threshold = m - size_A + 2
        b = len(results[-3])
        k = len(results)-2
        const = b - (m * k)

        res_data.append([m, curr_set, size_A, b, k, const])
    return res_data

def cone_of_A(A, show=True, thresh=10):
    hA = A
    cone_list = add_height(hA, 1)
    for h in range(2, thresh):
        hA = sum_sets(hA, A)
        cone_list.extend(add_height(hA, h))
    cone_list.append((0,0))

    import matplotlib.ticker as ticker

    if show:
        ax = plt.axes()
        plt.scatter(*zip(*cone_list))
        plt.grid()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        plt.show()
    return cone_list

def get_minimal_elements(m, cone_list):
    minimal_elements = [[] for _ in range(m)] 
    for residue in range(m):
        residue_class = [e for e in cone_list if e[0] % m == residue]
        # make sure the next line works as intended
        for e in residue_class:
            if tuple(np.subtract(e, (0,1))) not in cone_list and tuple(np.subtract(e, (m,1))) not in cone_list:
                minimal_elements[residue].append(e)
    
    return minimal_elements


def add_height(A, h):
    return [(e, h) for e in A]

def write_to_csv(fname, rows, stats=False):
    with open(fname, 'a') as file:
        writer = csv.writer(file)
        if not stats:
            writer.writerow(['m', 'A', '|A|', 'b', 'k', 'const'])
        writer.writerows(rows)

def random_sets_exps():
    random_sets = []
    max_m = 50 
    for i in range(2000):
        m = np.random.randint(5, max_m)
        max_size = np.random.randint(4, m)
        random_sets.append(random_set(m, max_size))

    all_results = run_exps(random_sets)

    results_by_const = dict()
    for elem in all_results:
        curr_const = elem[5]
        if curr_const in results_by_const:
            results_by_const[curr_const] += [elem]
        else:
            results_by_const[curr_const] = [elem]

    for const, results in results_by_const.items():
        write_to_csv(f'random_sets_const={const}.csv', results)

def single_sumset(A, show_steps=False):
    # print("m, A, |A|, b, k, ~k")
    return run_exps([A], show_steps)

def single_cone_example(A=None, rand=False):
    # A = [0, 2, 9, 14]
    # A = [0, 1, 10]
    # A = [0, 1, 10, 18]
    # A = [0, 2, 9, 11, 18]

    # max_n_list = [0 for _ in range(16)]
    if rand:
        # setss = [[0,1,j//2 + 1,j] for j in range(34,52, 2)]
        for _ in range(4000):
            m = np.random.randint(5, 25)
            max_size = np.random.randint(3, m)
            A = random_set(m, max_size)
        # for sett in setss:
            # A = {0,1,3,4}
            b = max(A)
            cone = cone_of_A(A, show=False, thresh=b)
            min_elems = get_minimal_elements(max(A), cone)
            # heights = []
            lengths = []
            for i, res_class in enumerate(min_elems):
                # height = sum(e[1] for e in res_class)
                # heights.append((i, height))
                lengths.append((i, len(res_class)))
    

            # sorted_heights = sorted(heights, key=lambda x: x[1], reverse=True)
            sorted_lengths = sorted(lengths, key=lambda x: x[1], reverse=True)
            # if sorted_heights[0][1] > b+1:
            if sorted_lengths[0][1] > 2:
                print(A)
                for j, e in enumerate(sorted_lengths):
                    if j > 6:
                        break
                    k, height = e
                    print(f'{j+1}th largest residue class:')
                    print(f'height = {height}')
                    print(f'minimal elements = {min_elems[k]}')
                    print(f'b = {b}')
                print()

            # for j in range(1, 16):
            #     if any(e[1] == max(A)-j for e in min_elems[max_index]):
            #         if max_n_list[j] < len(min_elems[max_index]):
            #             max_n_list[j] = len(min_elems[max_index])
                    # print(A)
                    # print(f'max total height = {max_height}')
                    # print(f'min elems {min_elems[max_index]}')
                    # print(f'b-1 = {max(A)-1}\n')
        # print(max_n_list)



def get_moments(max_m, moment):
    m_by_k_plus_one = [[]]
    for m in range(1,max_m+1):
        subsets = all_subsets(m)
        m_by_k_plus_one.append([])
        for k in range(m):
            results = run_exps(subsets[k])
            constants = np.array(list(zip(*results))[5])
            if moment <= 1:
                mmt = round(np.mean(constants), 2)
            else:
                mmt = round(stats.moment(constants, moment=moment), 2)
            m_by_k_plus_one[m].append(mmt)
    m_by_k_plus_one.pop(0)
    return m_by_k_plus_one


moment_dict = {1: "mean", 2: "variance", 3: "skew", 4: "kortosis"}
def plot_moment_data(m, moment):
    with open(f'statistical_experiments/m={m}_{moment_dict[moment]}.csv', newline='') as f:
        reader = csv.reader(f)
        last_row = [float(e) for e in list(reader)[-1]]
        ks = np.arange(m)
        fig = plt.figure()
        ax = fig.add_subplot()
        fig.suptitle(f'{moment_dict[moment]} constant for subsets of [m] of size k', fontsize=14, fontweight='bold')
        ax.set_title(f'm={m}')
        ax.set_xlabel('size of subsets k')
        ax.set_ylabel(f'{moment_dict[moment]} constant c')
        plt.bar(ks,last_row)
        plt.show()




single_cone_example(rand=True)


# new example of set to look into: [0,n,n+1,n+2,...,n+k] where n gets large

# For now, setup [0,n,...,n+k,m-r,...,m]
# want to understands what happens when going from k=0 to k=1
# y = 5 
# for i in range(40):
#     print(single_sumset([0,y,y+1,y+2+i]))
# at y=4,5 it seems like the consants are at a minium? Is this true? If yes then why?


# want to understand what happens when going from r=0 to r=1
# x = 1 
# for j in range(100):
#     print(single_sumset([0,x,x+1+j,x+2+j]))
# for x = 1, conjecture 2 says c = 1 if m is even else c = -m + 4
# for x = 2, c is approx -0.5m
# for x = 3, c is approx -0.66m if m = 2 mod 3 and -m if m = 0,1 mod 3
# for x = 4, c is approx -m if m = 2,3 mod 4 and -1.5m if m = 0,1 mod 4
# for x = 5, c is approx -1.15m if m = 2,3,4 mod 5 and -2m if m = 0,1 mod 5
# for x = 6, c is approx -1.45m if m = 2,3,4,5 mod 6 and -2.5m if m = 0,1 mod 6
# for x = 7, c is approx -1.625m if m = 2,3,4,5,6 mod 7 and -3m if m = 0,1 mod 7
# for x = 8, c is approx -2m if m = 2,3,4,5,6,7 mod 8 and -3.5m if m = 0,1 mod 8
# for x = 9, c is approx -2.15m if m = 2,3,4,5,6,7,8 mod 9 and -4m if m = 0,1 mod 9

# Empirical bound maybe? : for some fixed x > 1, -1/5xm < c < -1/2xm 
# Once we can bound what happens in these transitions then we can just apply conjecture 2




