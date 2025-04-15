from scipy.optimize import linprog
def transpose(matrix):
    if isinstance(matrix[0], list):
        # If the first element of the matrix is a list, implying it's a 2-dimensional matrix:
        # Transpose the matrix by swapping rows and columns.
        # Create a new matrix where each row in the original matrix becomes a column in the transposed matrix.
        # Iterate over columns (i) and rows (j) of the original matrix to build the transposed matrix.
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    else:
        # If the first element of the matrix is not a list, implying it's a 1-dimensional vector:
        # Convert the vector into a 1-column matrix (column vector) by placing each element in its own list.
        # This effectively represents the vector as a single-column matrix.
        return [[row] for row in matrix]


def matrix_multiply(matrix1, matrix2):
    # Iterate over each row in matrix1 and calculate the dot product with each column in matrix2.
    # For each row in matrix1, iterate over each column in matrix2 and compute the dot product.
    # The dot product is calculated by multiplying corresponding elements of the row and column, 
    # then summing up the products.
    # The result is a list comprehension that iterates over each row in matrix1, and for each row,
    # it iterates over each column in the transposed matrix2 (to facilitate accessing columns as rows for computation).
    # The dot product is calculated for each combination of row in matrix1 and column in matrix2.
    result = [[sum(a*b for a, b in zip(row, col)) for col in transpose(matrix2)] for row in matrix1]

    # Return the resulting matrix after multiplication.
    return result
def matrix_inverse(matrix):
    n = len(matrix)
    identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]  # Creating an identity matrix

    # Forward elimination
    for i in range(n):
        pivot = matrix[i][i]  # Selecting pivot element
        for j in range(n):
            matrix[i][j] /= pivot  # Dividing the current row by the pivot
            identity[i][j] /= pivot  # Dividing the corresponding row in the identity matrix by the pivot
        for k in range(i+1, n):
            factor = matrix[k][i]  # Computing the factor for elimination
            for j in range(n):
                matrix[k][j] -= matrix[i][j] * factor  # Subtracting a multiple of the pivot row from the current row
                identity[k][j] -= identity[i][j] * factor  # Updating the corresponding row in the identity matrix

    # Back substitution
    for i in range(n-1, -1, -1):
        for k in range(i-1, -1, -1):
            factor = matrix[k][i]  # Computing the factor for elimination
            for j in range(n):
                matrix[k][j] -= matrix[i][j] * factor  # Subtracting a multiple of the pivot row from the current row
                identity[k][j] -= identity[i][j] * factor  # Updating the corresponding row in the identity matrix

    return identity  # Returning the inverted matrix

def Task3(x, y, z, x_initial, c, x_bound, se_bound, ml_bound):
    # To calculate the weights for b and z we have to use linear regression. To calculate linear regression we use the equation
    # B = (X^T * X)^-1 * X^T * Y (where ^T means transpose and * means multiply (dot product)). However the matrix we using for Y is a row matrix and 
    # Not a column matrix so the equation is B = (X^T * X)^-1 * X * Y^T. The first step is to add a row of 1 for x so we can calculate the intercept term. 
    # Now we can calculate the weights firstly we calculate the transpose for x and then we can multiply x and its transpose to get (x^T * x) then we need to inverse the matrix to get (x^T * x)^-1. 
    # The next step to calculate the weights is to times (x^T * x)^-1 by x to get (x^T * x)^-1 * x. 
    # The last part to calculates weights_b and weights_z is to transpose y and z and then to times (x^T * x)^-1 * x by transpose y to get weights_y and times (x^T * x)^-1 * x by transpose z to get weights_z.


    # adds 1 add the start of the matrix so we can calculate the intercept term.
    x = [[1,1,1,1,1,1,1,1,1]] + x
    # calculates the transpose of x, y and z using the function transpose
    transpose_x = transpose(x)
    transpose_y = transpose(y)
    transpose_z = transpose(z)
    # multiples x and transpose_x to get (x^T * x)
    temp = matrix_multiply(x,transpose_x)
    # inverses the matrix temp to get (x^T * x)^-1
    temp = matrix_inverse(temp)
    # multiplies temp by x to get (x^T * x)^-1 * x
    temp = matrix_multiply(temp, x)
    # to gets weights_b we multiply temp by transpose_y to get (x^T * x)^-1 * x * y^T
    weights_b = matrix_multiply(temp, transpose_y)
    # to gets weights_z we multiply temp by transpose_z to get (x^T * x)^-1 * x * z^T
    weights_d = matrix_multiply(temp, transpose_z)
    # Flatten weights_b and weights_z and convert them into lists 
    weights_b = [item for sublist in weights_b for item in sublist]
    weights_d = [item for sublist in weights_d for item in sublist]


    # The object function we are trying to minimize is c[0]*x1 + c[1]*x2 + c[2]*x3 + c[3]*x4.
    # So object coefficients are (objective_coeffs) c[0], c[1], c[2], c[3]
    objective_coeffs = [c[0], c[1], c[2], c[3]]

    # The constraints function we have to follow are the total safeguard value has to be no less than se_bound and 
    # The total maintenance load needs to be no greater than ml_bound. To put them in equations it is 
    # y + y_initial => se_bound and z + z_initial <= ml_bound. However to get it in proper form all the known variables (constants) e.g se_bound, ml_bound, y and z initial, b0
    # need to be on the side of <= so we need to change y => se_bound by multiplying it by -1 to get -y - y_initial <= -se_bound and then we need to add y_initial to both sides to get 
    # -y <= -se_bound + y_initial we also need to add b0 over to as y is calculated as y = b0 + b1x1 + b2x2 + b3x3 + b4x4 and b0 is a constant. To change z + z_initial <= ml_bound we need to subtract
    # z_initial to both sides to get z <= ml_bound - z_initial we also need to subtract z0 over to as z is calculated as z = z0 + z1x1 + z2x2 + z3x3 + z4x4 and z0 is a constant.
    # Leaving our equations to be -b1x1 - b2x2 - b3x3 - b4x4 <= -se_bound + b0 + initial_y and z1x1 + z1x2 + z1x3 + z4x4 <= ml_bound - initial_z - z0  
    # Now to get the constraints coefficients are the numbers in front of each x value e.g the first set of constraints coefficients -b1, -b2, -b3, -b4, and the next set of constraints are
    # z1, z2, z3, z4
    constraints_coeffs = [
        [-weights_b[1], -weights_b[2], -weights_b[3], -weights_b[4]], 
        [weights_d[1], weights_d[2], weights_d[3], weights_d[4]]      
    ]
    # The constraints_rhs represent the upper bounds for the safeguard effect and maintenance load.
    # e.g for the safeguard effect is has to be <= -se_bound + b0 + initial_y and for the maintenance load it is ml_bound - initial_z - z0.
    # To calculate initial_y it is x_initial[1]*b[1] + x_initial[2]*b[2] + x_initial[3]*b[3] + x_initial[4]*b[4] where x_initial represents the initial number for each security control
    # To calculate initial_z it is x_initial[1]*z[1] + x_initial[2]*z[2] + x_initial[3]*z[3] + x_initial[4]*z[4].
    constraints_rhs = [-se_bound + weights_b[0] + sum(weights_b[i+1] * x_initial[i] for i in range(4)),  
                       ml_bound - weights_d[0] - sum(weights_d[i+1] * x_initial[i] for i in range(4))] 
    print(constraints_coeffs)
    print(constraints_rhs)
    # Define bounds for the decision variables.
    # These bounds represent the maximum number of each additional security controls that can be deployed.
    bounds = [(0, x_bound[0]), (0, x_bound[1]), (0, x_bound[2]), (0, x_bound[3])]
    # Solve the linear programming problem using the linprog function from scipy.optimize.
    # The objective is to minimize the total cost, subject to the defined constraints.
    result = linprog(objective_coeffs, A_ub=constraints_coeffs, b_ub=constraints_rhs, bounds=bounds)
    # Extract the solution vector x_add from the result obtained by linprog.
    x_add = result.x
    print(x_add)
    # return weights_b, weights_d, x_add
    return (weights_b, weights_d, x_add)


x = [[5,4,8,8,2,5,5,7,8],[3,7,7,2,2,5,10,4,6],[8,3,6,7,9,10,6,2,2],[9,3,9,3,10,4,2,3,7]]
y = [82,53,94,58,80,74,69,47,71]
z = [57,58,82,53,46,66,81,50,67]
c = [11, 6, 8, 10]
se_bound = 100
ml_bound = 10000
x_bound = [300,500,200,450]
x_initial = [3,5,4,2]
print(Task3(x, y, z, x_initial, c, x_bound, se_bound, ml_bound))




