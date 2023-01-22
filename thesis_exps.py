from math import gcd
from functools import reduce

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

def write_to_csv(fname, rows):
    with open(fname, 'a') as file:
        writer = csv.writer(file)
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
    max_m = 20 
    for i in range(1000):
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
    print("m, A, |A|, b, k, ~k")
    print(run_exps([A], show_steps))

def single_cone_example():
    A = [0,1,5,6,9]
    single_sumset(A)
    # thresh = 2*max(A)-4
    thresh = 7
    cone = cone_of_A(A, show=False, thresh=thresh)
    print("minimal elements")
    min_elems = get_minimal_elements(max(A), cone)
    print(min_elems, len(min_elems))
    
# random_sets_exps()
# single_sumset([0,4,5,6,7], show_steps=False)
for i in range(7, 80):
    single_sumset([0,3,4,5] + [i,i+1,i+2,i+3])

"""
 Idea 1
 rather than starting with a set and gneerating number. what if we started with fixed b and k. 
 which sets have b=1,2,whatever. can you generate sets that have this? do they have anything in common.
 maybe b doesn't have to do with A, but with something like A-A. things like this.

 calculus analogy: if you want to find the length of a curve, it depends on the derivative. 

First: study classes of sets with a fixed constant c = b - mk
Let n = min(A\{0}) and m=max(A). Then

 Simple observations:
 - -inf < c <= 1
 - if c=1 then 1 \in A, and similarly, if c=0 then 1 or 2 \in A, etc.
 - c <= -n+2

Conjectures:
 -[Sufficient Conditions] Note that the set A contains two chains (contiguous sequence of elements),
 one starting with n and the other ending at m.
 Let d = n-k if n>k else 0. Then
    + If A = [0,n,n+1,...,m] then c=-n-d+2 (this implies that for any set A, c is bounded by -n+2)
    + If A = [0,n,n+1,...,n+k,m-1,m] and m-1 > n(n+k), then c=-n-d+2 if m-1 != 0 mod n+k+1 or c=-n-d+1 otherwise. Similarly,
    + If A = [0,n,n+1,...,n+k,m-2,m-1,m] and m-2 > n(n+k), then c=-n-d+2 if m-2 != -1 mod n+k+2 or c=XXX otherwise. 
    + If A = [0,n,n+1,...,n+k,m-3,m-2,m-1,m] and m-2 > n(n+k), then c=-n-d+2 if m-3 != -2 mod n+k+3 or c=XXX otherwise. 
    General case:
    + If A = [0,n,n+1,...,n+k,m-r,...,m-1,m] and m-r > n(n+k), then c=-n-d+2 if m-r != -r+1 mod n+k+r or c=XXX otherwise. 


    TODO: Find out what XXX (value of constant on the degenerate residue class) is for the 2,3, and general case
    TODO: Find out what happens if we stick elements in between the short and long chain.
    TODO: Try to come up with an example set A that reaches its upper bound c=-n+2 but does not satisfy this sufficient condition.

 - if [0, i,i+1,...,i+j,i+j+r, i+j+r+1,...,m]'s sumset had constant c then 
 [0, i, i+1,...,i+j,i+j+r,i+j+r+1,...,m,m+1], and [0, i, i+1,...,i+j,i+j+1,i+j+r,i+j+r+1,...,m,m+1] also have constant c
 and  [0, i, i+1,...,i+j,i+j+1,i+j+r,i+j+r+1,...,m] 'almost' also has constant c


 Idea 2 about raking d+3 elements in d dimensions to d+1 dimensions:
 Try to put second element in tuple in first tuple - 1 (or maybe just duplicate: a -> (a,a) if that gives linear ind.)
 What if we project into a random line after augmenting dimension rather than y axis
 """

"""
    + The first chain has at least n+1 elements and the second chain has at least 2 elements
    TODO: try to come up with an example set A that satisfies these conditions but s.t. c != 0
        single_sumset([0,2,3,4,20,21])
"""