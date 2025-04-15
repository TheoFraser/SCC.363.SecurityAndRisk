def Task2(num, table, probs):
    # Extracting values from inputs
    PX2, PX3, PX4, PX5, PY6, PY7 = probs
    a, b, c, d = table[0]
    e, f, g, h = table[1]
    i, j, k, l = table[2]

    # (1) To calculate the probability of 3 <= X <= 4 we need to calculate the sum of all the values of the table between 3 and 4 inclusive which is b + f + j + c + g + k.
    # Then to get it into a probability we need to divide it by the total number cases.
    prob1 = (b + c + f + g + j + k) / num  # Probability of 3 ≤ X ≤ 4
    # (1) To calculate the probability of X + Y <= 10 we need to calculate the sum of all the values of the table where its X value plus its Y value is less than or equal to 10
    # In this case a + b + c + e + f + i. Then to get it into a probability we need to divide it by the total number cases.
    prob2 = (a + b + c + e + f + i) / num  # Probability of X + Y <= 10

    # To calculate prob3 which is Y = 8 given that a case is tested positive firstly we need to represent it as P(Y = 8 | T).
    # Next we need to apply Bayes’ theorem to get P(Y = 8 | T) = (P(T | Y = 8) * P(Y = 8)) / P(T). 
    prob_X2 = (a + e + i) / num  # P(X = 2)
    prob_X3 = (b + f + j) / num  # P(X = 3)
    prob_X4 = (c + g + k) / num  # P(X = 4)
    prob_X5 = (d + h + l) / num  # P(X = 5)
    # Firstly we need to calculate P(T) (probability someone is tested positive)
    # so the first step to do that is to use Bayes’ theorem on P(X = 2 | T) to get (P(T | X = 2) * P(T)) / P(X = 2) and then we multiply both sides to get P(X = 2 | T) * P(X = 2) = P(T | X = 2) * P(T).
    # Then we do this for all the rest of the X's e.g X =3, X = 4 and add them all together to get 
    # (P(X = 2 | T) * P(X = 2)) + (P(X = 3 | T) * P(X = 3)) + (P(X = 4 | T) * P(X = 4)) + (P(X = 5 | T) * P(X = 5)) = (P(T | X = 2) * P(T)) + (P(T | X = 3) * P(T)) + (P(T | X = 4) * P(T)) + (P(T | X = 5) * P(T))
    # we can simplify the right hand side to P(T)*(P(T | X = 2) + P(T | X = 3) + P(T | X = 4) + P(T | X = 5)) where P(T | X = 2) + P(T | X = 3) + P(T | X = 4) + P(T | X = 5) = 1 
    # as this means that the probability that someone tested positive given that X = 2 plus probability that someone tested positive given that X = 3 etc till 5 which basically means what is the probability of 
    # someone testing given any X which is equal to 1. So P(T) is equal to (P(X = 2 | T) * P(X = 2)) + (P(X = 3 | T) * P(X = 3)) + (P(X = 4 | T) * P(X = 4)) + (P(X = 5 | T) * P(X = 5))  which we have all the values for.
    prob_T = (PX2 * prob_X2) + (PX3 * prob_X3) + (PX4 * prob_X4) + (PX5 * prob_X5)

    # Now we need to calculate P(Y = 8) which is equal to all the numbers in the table in the Y = 8 row which is i + j + k + l and then we need to divide it by num to get it as a probability. 
    prob_Y6 = (a + b + c + d) / num # P(Y = 6)
    prob_Y7 = (e + f + g + h) / num # P(Y = 7)
    prob_Y8 = (i + j + k + l) / num # P(Y = 8)
    # Lastly we need to calculate P(T | Y = 8) which we calculate by doing what we did to the X values getting the equations P(T)*(P(T | Y = 6) + P(T | Y = 7) + P(T | Y = 8)) = ((P(Y = 6 | T) * P(Y = 6)) + (P(Y = 7 | T) * P(Y = 7)) + (P(Y = 8 | T) * P(Y = 8)).
    # P(T)*(P(T | Y = 6) + P(T | Y = 7) + P(T | Y = 8)) = P(T) as (P(T | Y = 6) + P(T | Y = 7) + P(T | Y = 8)) equals 1 because of the same reason as the x's.
    # So now we get the equation P(T) = ((P(Y = 6 | T) * P(Y = 6)) + (P(Y = 7 | T) * P(Y = 7)) + (P(Y = 8 | T) * P(Y = 8)) and now we can rearrange the equation to get P(Y = 8 | T) = (P(T) - (P(Y = 6 | T) * P(Y = 6)) - (P(Y = 7 | T) * P(Y = 7))) / P(Y = 8)
    # Now we can solve this equation to get P(Y = 8 | T).
    probY_given_T = (prob_T - (PY6 * prob_Y6) - (PY7 * prob_Y7)) / prob_Y8

    # Now we have all the value for P(Y = 8),  P(Y = 8 | T) and P(T) we can use the equation P(Y = 8 | T) = (P(T | Y = 8) * P(Y = 8)) / P(T) to finally get P(Y = 8 | T).
    prob3 = (probY_given_T * prob_Y8) / prob_T

    return (prob1, prob2, prob3)



# num = 80
# probs = [0.3,0.4,0.3,0.2,0.1,0.5]
# table = [[9, 7, 8, 5], [10, 6, 3, 5], [6, 7, 5, 9]]
# (prob1, prob2, prob3) = Task2(num, table, probs)
# print(prob1, prob2, prob3)
