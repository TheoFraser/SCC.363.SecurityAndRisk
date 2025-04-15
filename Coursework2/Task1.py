# from dhCheck_Task1 import dhCheckCorrectness
import random
import math
def Task1(a, b, c, point1, number_set, prob_set, num, point2, mu, sigma, xm, alpha, point3, point4):
    # TODO
    prob1, MEAN_t, MEDIAN_t = TriangularDistribution(a,b,c,point1)
    MEAN_d = calculate_mean(number_set, prob_set)
    VARIANCE_d = calculate_variance(number_set, prob_set, MEAN_d)
    impact_A = pareto_random(xm, alpha, num)
    impact_B = generate_log_normal_samples(mu, sigma, num)
    total_impact = [a + b for a, b in zip(impact_A, impact_B)] 
    prob2 = calculate_probability(total_impact, point2)
    prob3_4 = calculate_probability_between(total_impact, point3, point4)

    print("Probability that the total impact is greater than", point2, ":", prob2)
    print("Probability that the total impact is between", point3, "and", point4, ":", prob3_4)
    return (prob1, MEAN_t, MEDIAN_t, MEAN_d, VARIANCE_d, prob2, prob3, ALE)

def calculate_probability(total_impact, point):
    count = sum(1 for impact in total_impact if impact > point)
    return count / len(total_impact)

def calculate_probability_between(total_impact, point_lower, point_upper):
    count = sum(1 for impact in total_impact if point_lower < impact < point_upper)
    return count / len(total_impact)

def calculate_mean(numbers, probabilities):
    return sum(n * p for n, p in zip(numbers, probabilities))

def calculate_variance(numbers, probabilities, mean):
    return sum((n - mean) ** 2 * p for n, p in zip(numbers, probabilities))

def generate_log_normal_samples(mu, sigma, num_samples):
    samples = []
    for _ in range(num_samples):
        u1 = random.random()
        u2 = random.random()
        
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        log_normal_sample = math.exp(mu + sigma * z0)
        
        samples.append(log_normal_sample)
    
    return samples

## Task3: Pareto distribution
def pareto_random(alpha, x_m, size):
    samples = []
    for _ in range(size):
        u = random.random()  # Uniform random number between 0 and 1
        samples.append(x_m / (u ** (1 / alpha)))
    return samples

def TriangularDistribution(a, b, c, data):
    cdf = 0
    if data <= a:
        cdf = 0
    elif data <= c:
        cdf = (data - a)**2 / ((b - a) * (c - a))
    elif data < b:
        cdf = 1 - (b - data)**2 / ((b - a) * (b - c))
    else:
        cdf = 1

    mean = (a + b + c) / 3
    median = 0
    if c >= (a+b)/2:
        median = a + (((b-a)*(c-a))/ 2)**0.5
    else:
        median = b - (((b-a)*(b-c))/ 2)**0.5
    return cdf, mean, median

if __name__ == "__main__":
    random.seed(42)
    # Step 1: Generate random samples for the impacts caused by flaws A and B
    a, b, c, point1, number_set, prob_set, num, point2, mu, sigma, xm, alpha, point3, point4 \
    = 10000, 35000, 18000, 12000, [0,1,2,3,4,5,6,7,8,9], [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1], 500000, 25000, 0, 3, 1, 4, 30, 50
    (prob1, MEAN_t, MEDIAN_t, MEAN_d, VARIANCE_d, prob2, prob3, ALE) = Task1(a, b, c, point1, number_set, prob_set, num, point2, mu, sigma, xm, alpha, point3, point4)

    # a = 100
    # b = 50000
    # c = 20000
    # data = range(1, 60001)
    # result = TriangularDistribution(a, b, c, data)
    # print(result)

import numpy as np

def generate_log_normal_samples(mu, sigma, num_samples):
    return np.random.lognormal(mu, sigma, num_samples)

def generate_pareto_samples(xm, alpha, num_samples):
    return np.random.pareto(alpha, num_samples) + xm

# Step 1: Generate random samples for the impacts caused by flaws A and B
mu_A = 1  # Mean of log-normal distribution for flaw A
sigma_A = 0.5  # Standard deviation of log-normal distribution for flaw A
xm_B = 2  # Scale parameter of Pareto distribution for flaw B
alpha_B = 2  # Shape parameter of Pareto distribution for flaw B
num_samples = 10000  # Number of Monte Carlo samples
np.random.seed(42) 
impact_A = generate_log_normal_samples(mu_A, sigma_A, num_samples)
impact_B = generate_pareto_samples(xm_B, alpha_B, num_samples)

# Step 2: Calculate the total impact as the sum of impacts caused by flaws A and B
total_impact = impact_A + impact_B

# Step 3: Analyze the samples to determine the desired probabilities
point2 = 5  # Point for probability calculation 2
point3 = 6  # Lower bound for probability calculation 3
point4 = 8  # Upper bound for probability calculation 3

prob2 = np.mean(total_impact > point2)
prob3_4 = np.mean((total_impact > point3) & (total_impact < point4))

print("Probability that the total impact is greater than", point2, ":", prob2)
print("Probability that the total impact is between", point3, "and", point4, ":", prob3_4)
