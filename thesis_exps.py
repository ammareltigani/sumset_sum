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
single_sumset([0,2,3,6,7,8,9,12,15,16,17],show_steps=False)
#for i in range(4, 29):
#    single_sumset([0,1,2,3] + [i,i+1] + [30,31],show_steps=False)


"""
Leo-
Idea 1
rather than starting with a set and generating number. what if we started with fixed b and k. 
which sets have b=1,2,whatever. can you generate sets that have this? do they have anything in common.
maybe b doesn't have to do with A, but with something like A-A. things like this.

calculus analogy: if you want to find the length of a curve, it depends on the derivative. 
"""

"""
Ammar-
Studying classes of sets with a fixed constant c (where p(h) = m(h-k) + b, valid for h>=k. So c = b-mk)
Assume only working with nonnegative integers, 0 \in A, and A-A generates Z. Also let n = min(A\{0}) and m=max(A).

Conjecture1:
[Bound] For any set A, c <= -n+2. So, in particular, -inf < c <= 1 since 1 is the smallest n we can have, and for a set with
 c, there exist an element x \in A s.t. x <= 2-c

Conjecture2:
Note that the set A contains two chains (contiguous sequence of elements), one starting with n (call this one the first chain) and 
the other ending at m (call this one the last chain). They may coincide. Let d = n-k if n>k else d=0. Then
    + If A = [0,n,n+1,...,n+k(=m)] then c=-n-d+2. TODO: prove this easy case first.
    + If A = [0,n,n+1,...,n+k,m-1,m] and m-1 > n(n+k), then c=-n-d+2 if m-1 != 0 mod n+k+1 or c=-n-d+1 otherwise. Similarly,
    + If A = [0,n,n+1,...,n+k,m-2,m-1,m] and m-2 > n(n+k), then c=-n-d+2 if m-2 != -1 mod n+k+2 or c=XXX otherwise. 
    + If A = [0,n,n+1,...,n+k,m-3,m-2,m-1,m] and m-3 > n(n+k), then c=-n-d+2 if m-3 != -2 mod n+k+3 or c=XXX otherwise. 
General case:
    + If A = [0,n,n+1,...,n+k,m-r,...,m-1,m] and m-r > n(n+k), then c=-n-d+2 if m-r != -r+1 mod n+k+r or c=XXX otherwise. 

Remarks:
    - [Almost-Invariant] This implies that the c value is unchanged after appending to the first chain and/or prepending to the last
        chain (so long as the mod case is still the first).
    - TODO: Find counterexample for condition m-r>n(n+k). I suspect that there is a correct threshold but not sure this is the one.
    - TODO: Find out what XXX (value of constant on the degenerate residue class) is for the 2,3, and general case
    - Q: What happens if we stick elements in between the first and last chain.
        + For a single chain with a single element the explicit pattern is hard to recognize, but heuristically, whenever the inserted
            element is small (close to the first chain) the constant is as if the insertd element was part of the first chain and
            whenever it is large (close to the last chain) the constant is as if the inserted element was part of the last chain.
            TODO: Find a threshold for what 'small' and 'large' means.
         TODO: try to answer same question for a chain with with multiples elements, then for multiples chains with multiples elements. 
    - To prove (1), have to show that there cannot be a set A s.t. c>-n+2. Can argue by looking at any arbitrary set A with
        min(A\{0})=n and max(A)=m as an set with two chains and potentially stuff in it. Then problem reduces to proving that adding
        stuff in the middle of the two chains can only decrease the constant or keep it fixed. 

Conjecture3:
+ Adding chains of elements (or just 'elements') in between the last chain only decreases the constant c or keeps it fixed
=> This furnishes a proof for (3) since we have to show that there cannot be a set A s.t. c>-n+2. Can argue by looking at any arbitrary
    set A with min(A\{0})=n and max(A)=m as an set with two chains and potentially stuff in it. 

Conjecture4:
+ [Sufficient] If A is comprised of exactly the two first and last chains described and satisfying the conditions above, then c=-n-d+2
    (nothing new, this is just restating the previous conjecture).
+ [Necessary] If c = -n-d+2 then A's first and last chains must satisfy the conditions described above.
"""


"""
Leo-
Idea 2 about raking d+3 elements in d dimensions to d+1 dimensions:
Try to put second element in tuple in first tuple - 1 (or maybe just duplicate: a -> (a,a) if that gives linear ind.)
What if we project into a random line after augmenting dimension rather than y axis
"""