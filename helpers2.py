import random

# Helper function to generate a Hilbert matrix
def hilbert_matrix(n):
    """Generates an n x n Hilbert matrix."""
    return [[1.0 / (i + j - 1) for j in range(1, n+1)] for i in range(1, n+1)]

# Helper function to generate a random matrix
def random_matrix(n, min_value=-10, max_value=10):
    """Generates an n x n random dense matrix with values between min_value and max_value."""
    return [[random.uniform(min_value, max_value) for _ in range(n)] for _ in range(n)]



def print_matrix(matrix, end_symbol="----------"):
    """
    Prints a matrix (list of lists) in a readable format, ensuring proper alignment of elements.
    Adds an optional end symbol (default is a line of dashes) to indicate where the matrix ends.
    Handles cases where the input is a vector (list) or a single number.

    Parameters:
    matrix (list of lists or list or number): A matrix represented as a list of lists, a vector as a list, or a single number.
    end_symbol (str): A symbol or message to indicate the matrix ended (default is "----------").

    Example:
    A = [[2.0, 3.0, -1.0],
         [4.0, 1.0, 2.0],
         [-2.0, 7.0, 2.0]]
    print_matrix(A)
    """
    # If the input is a single number, print it directly
    if isinstance(matrix, (int, float)):
        print(f"[ {matrix:.2f} ]")
        print(end_symbol)
        return

    # If the input is a vector (1D list), print each element in its own row
    if isinstance(matrix, list) and all(isinstance(elem, (int, float)) for elem in matrix):
        for elem in matrix:
            print(f"[ {elem:.2f} ]")
        print(end_symbol)
        return

    # Handle the case where the input is a matrix (list of lists)
    if isinstance(matrix, list) and all(isinstance(row, list) for row in matrix):
        # Determine the maximum width of each column for better formatting
        col_widths = [max(len(f"{elem:.2f}") for elem in col) for col in zip(*matrix)]

        for row in matrix:
            # Format each row with the determined column widths, rounded to 2 decimal places
            formatted_row = "  ".join(f"{elem:>{col_widths[i]}.2f}" for i, elem in enumerate(row))
            print(f"[ {formatted_row} ]")
    
        # Print the matrix end symbol
        print(end_symbol)
        return

    # If input doesn't match any valid types, print an error message
    print("Invalid matrix input. Please provide a list of lists, a list (vector), or a single number.")


if __name__ == "__main__":

    # Example usage
    A = [[2.0, 3.0, -1.0],
        [4.0, 1.0, 2.0],
        [-2.0, 7.0, 2.0]]

    B = [1.0, 2.0, 3.0]

    C = 42.0

    print("Matrix A:")
    print_matrix(A)

    print("\nVector B:")
    print_matrix(B)

    print("\nSingle number C:")
    print_matrix(C)
