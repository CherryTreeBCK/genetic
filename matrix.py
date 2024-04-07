import random

class Matrix:
    def __init__(self, rows = None, cols = None, python_list = None):
        if rows != None and cols != None:
            self.rows = rows
            self.cols = cols
            self.data = [[0 for _ in range(cols)] for _ in range(rows)]
        elif python_list != None:
            self.rows = len(python_list)
            self.cols = len(python_list[0])
            self.data = [row[:] for row in python_list]
            print("Matrix successfully created from python list")
        else:
            raise ValueError("Either rows and cols or python_list must be provided")
    
    # Returns a matrix of random values with the given dimensions
    @classmethod
    def random(cls, rows, cols):
        m = cls(rows, cols)
        m.data = [[random.random() for _ in range(cols)] for _ in range(rows)]
        return m
        
    # Returns a matrix with the function applied
    def apply_function(self, func):
        m = Matrix(rows = self.rows, cols = self.cols)
        m.data = [[func(self.data[r][c]) for c in range(self.cols)] for r in range(self.rows)]
        return m
    
    # Overload the [] operator to access elements of the matrix
    def __getitem__(self, idx):
        return self.data[idx]
    
    # Overload the [] operator to set elements of the matrix
    def __setitem__(self, idx, value):
        self.data[idx] = value
    
    # Returns the transpose of the given matrix
    def transpose(self):
        t = Matrix(rows = self.cols, cols = self.rows)
        
        for r in range(self.rows):
            for c in range(self.cols):
                t.data[c][r] = self.data[r][c]
        
        return t
    
    # Returns true if each matrix is the same dimension, false otherwise
    def is_same_dimension(self, B):
        return self.cols == B.cols and self.rows == B.rows
        
    # Returns a single boolean based on if each element is equal
    def equals(self, B):
        if not self.is_same_dimension(B):
            return False

        for r in range(self.rows):
            for c in range(B.cols):
                if self.data[r][c] != B.data[r][c]:
                    return False

        return True

    # Returns a new matrix that is the dot product of A and B
    def dot(self, B):
        if self.is_same_dimension(B) and self.cols == 1:
            return self.dot_row(B)
        
        if self.cols != B.rows:
            raise ValueError("Dot product error: Dimensions not compatible")
        
        prod = Matrix(rows = self.rows, cols = B.cols)

        for i in range(self.rows):
            for j in range(B.cols):
                for k in range(self.cols):
                    prod.data[i][j] += self.data[i][k] * B.data[k][j]

        return prod
    
    # Returns a new matrix with the elements of A added to the elements of B
    def add(self, B):
        if self.cols != B.rows:
            raise ValueError("Add error: Dimensions not compatible")
        
        sum = Matrix(rows = self.rows, cols = B.cols)

        for r in range(self.rows):
            for c in range(self.cols):
                sum.data[r][c] = self.data[r][c] + B.data[r][c]

        return sum
    
    # Returns a new matrix with the elements of A added to the scalar
    def add_scalar(self, scaler):
        # No need for dimension check with scalar
        sum = Matrix(rows = self.rows, cols = self.cols)

        for r in range(self.rows):
            for c in range(self.cols):
                sum.data[r][c] = self.data[r][c] + scaler

        return sum
    
    # Returns a new matrix with the elements of B subtracted from the elements of A
    def subtract(self, B):
        if not self.is_same_dimension(B):
            raise ValueError("Subtract error: Dimensions not compatible")
        
        diff = Matrix(rows = self.rows, cols = B.cols)

        for r in range(self.rows):
            for c in range(B.cols):
                diff.data[r][c] = self.data[r][c] - B.data[r][c]

        return diff
    
    def subtract_scaler(self, scaler):
        # No need for dimension check with scalar        
        diff = Matrix(rows = self.rows, cols = self.cols)

        for r in range(self.rows):
            for c in range(self.cols):
                diff.data[r][c] = self.data[r][c] - scaler

        return diff
    
    # Returns a new matrix with the elements of A multiplied by the elements of B
    # elementwise multiplication
    def multiply(self, B):
        if not self.is_same_dimension(B):
            raise ValueError("Multiply elementwise error: Dimensions not compatible")
        
        prod = Matrix(rows = self.rows, cols = B.cols)

        for r in range(self.rows):
            for c in range(B.cols):
                prod.data[r][c] = self.data[r][c] * B.data[r][c]

        return prod
        
    def multiply_scalar(self, scalar):
        # No need for dimension check with scalar
        prod = Matrix(rows = self.rows, cols = self.cols)
        
        for r in range(self.rows):
            for c in range(self.cols):
                prod.data[r][c] = self.data[r][c] * scalar
        
        return prod


