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


# Want to generate two lists of same size that have the same maximal element m. 
# One is multiples of 2 and the other is primes only.
def primes(n):
    """ Returns  a list of primes < n """
    sieve = [True] * n
    for i in range(3,int(n**0.5)+1,2):
        if sieve[i]:
            sieve[i*i::2*i]=[False]*((n-i*i-1)//(2*i)+1)
    return [2] + [i for i in range(3,n,2) if sieve[i]]

# Works for m >= 7
def generate_multiples_of_number_and_prime_lists(number, m):
    prime_list = [0] + primes(m) + [m]
    multiples_of_two_list = [number*i for i in range(m)]
    multiples_of_two_list.insert(2, 3)
    multiples_of_two_list = multiples_of_two_list[:len(prime_list)]
    multiples_of_two_list[-1] = m

    assert len(prime_list) == len(multiples_of_two_list)
    return prime_list, multiples_of_two_list

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

    if show:
        ax = plt.axes()
        plt.scatter(*zip(*cone_list))
        ax.set_xticks(range(50))
        ax.set_yticks(range(thresh))
        plt.grid()
        plt.show()
    return cone_list

def get_minimal_elements(m, cone_list):
    minimal_elements = set() 
    for residue in range(m):
        residue_class = [e for e in cone_list if e[0] % m == residue]
        # make sure the next line works as intended
        min_elem = min(residue_class)
        for e in residue_class:
            if e[1] <= min_elem[1] and e[0] != min_elem[0]:
                minimal_elements.add(residue)
    
    return minimal_elements


def add_height(A, h):
    return [(e, h) for e in A]

def write_to_csv(fname, rows, stats=False):
    with open(fname, 'a') as file:
        writer = csv.writer(file)
        if not stats:
            writer.writerow(['m', 'A', '|A|', 'b', 'k', 'const'])
        writer.writerows(rows)

def prime_and_evens_exps():
    cum_data = []
    for i in range(5, 40):
        lists = list(set(l) for l in generate_multiples_of_number_and_prime_lists(4, i))
        cum_data.extend(run_exps(lists, i))
    write_to_csv('primes_and_multiples_of_two.csv', cum_data)

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

def single_cone_example():
    A = [0,1,5,6,9]
    single_sumset(A)
    # thresh = 2*max(A)-4
    thresh = 7
    cone = cone_of_A(A, show=False, thresh=thresh)
    print("minimal elements")
    min_elems = get_minimal_elements(max(A), cone)
    print(min_elems, len(min_elems))


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


# new example of set with large and negative constant: [0,n,n+1,m]

# want to understands what happens when going from k=0 to k=1
# for i in range(40):
    # print(single_sumset([0,1,3+i]))
    # print(single_sumset([0,1,2,3+i]))


# want to understand what happens when going from r=0 to r=1
for j in range(40):
    if (5+j)%3 == 0:
        continue
    print(single_sumset([0,3,5+j]))
    print(single_sumset([0,3,4+j,5+j]))
    print()

# once we can bound what happens in these transitions then we can just apply conjecture 2

