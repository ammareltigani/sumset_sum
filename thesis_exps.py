from math import gcd
from functools import reduce

import csv
import numpy as np

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

def run_exps(sets):
    res_data = []
    for curr_set in sets:
        results = [curr_set]
        n_set = curr_set
        double_flag = False
        while True:
            prev_size = len(n_set)
            n_set = sum_sets(n_set, curr_set)
            results.append(n_set)
            curr_diff = len(n_set) - prev_size
            if curr_diff == max(curr_set):
                if double_flag:
                    break
                else:
                    double_flag = True
        assert len(sum_sets(n_set, curr_set)) - len(n_set) == max(curr_set)

        m = max(curr_set)
        size_A = len(curr_set)
        predicted_threshold = m - size_A + 2

        res_data.append([m, curr_set, size_A, len(results[-3]), len(results)-3, predicted_threshold])
    return res_data


def write_to_csv(fname, rows):
    with open(fname, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(['m', 'A', '|A|', 'b', 'k', '~k'])
        writer.writerows(rows)

def prime_and_evens_exps():
    cum_data = []
    for i in range(5, 40):
        lists = list(set(l) for l in generate_multiples_of_number_and_prime_lists(4, i))
        cum_data.extend(run_exps(lists, i))
    write_to_csv('primes_and_multiples_of_two.csv', cum_data)

def random_sets_exps():
    random_sets = []
    for i in range(50):
        m = np.random.randint(3, 20)
        max_size = np.random.randint(2, m)
        random_sets.append(random_set(m, max_size))

    results = run_exps(random_sets)
    write_to_csv('random_sets_thresh_small_m.csv', results)

#random_sets_exps()