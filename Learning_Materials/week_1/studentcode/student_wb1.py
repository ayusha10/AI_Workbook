from approvedimports import *

def exhaustive_search_4tumblers(puzzle: CombinationProblem) -> list: #creating the exhaustive search 4tumblers function
    """Simple brute-force search method that tries every combination until
    it finds the answer to a 4-digit combination lock puzzle.
    """  
    # check that the lock has the expected number of digits
    assert puzzle.numdecisions == 4, "this code only works for 4 digits"

    # creating an empty candidate solution
    my_attempt = CandidateSolution()

    # ===> insert your code below here
    for num1 in puzzle.value_set:#inserting the code by using nested for loop
        for num2 in puzzle.value_set:
            for num3 in puzzle.value_set:
                for num4 in puzzle.value_set:
                    my_attempt.variable_values = [num1, num2, num3, num4]
                    
                    try: #exception handeling
                        result = puzzle.evaluate(my_attempt.variable_values)
                        if result == 1:  # Assuming 1 means correct solution
                            return [num1, num2, num3, num4]
                    except Exception:
                        continue
    # <==== insert your code above here

    # should never get here if puzzle is solvable
    return [-1, -1, -1, -1]
#completed of activity 1

import numpy as np

def get_names(namearray: np.ndarray) -> list: #creating the function for an array
    family_names = []
    # ====> insert your code below here
    for i in range(namearray.shape[0]):  #using for loop 
        family_name = "".join(namearray[i, -6:]).strip()  
        family_names.append(family_name)  
    # <==== insert your code above here
    return family_names
    return family_names
#completed of activity 3

def check_sudoku_array(attempt: np.ndarray) -> int: #creating the first function for the sudoku array
    tests_passed = 0
    slices = []  # this is the list of numpy arrays
    
    # ====> insert your code below here
    # using assertions to check that the array has 2 dimensions each of size 9
    assert attempt.ndim == 2, "Array must be 2-dimensional"
    assert attempt.shape[0] == 9 and attempt.shape[1] == 9, "Array must be 9*9 "
    
    #checking all the row
    #using nested for loop for multiple times to make it similer and understandable 
    for row in attempt:
        if len(np.unique(row)) == 9 and np.all((row >= 1) & (row <= 9)):
            tests_passed += 1
            
    #checking column
    for col in attempt.T:
        if len(np.unique(col)) == 9 and np.all((col >= 1) & (col <= 9)):
            tests_passed += 1
    
    #checking all 3*3 
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            slices.append(attempt[i:i+3, j:j+3])
            
    for slice in slices:
        pass
        unique_values = np.unique(slice)
        if len(unique_values) == 9 and np.all(unique_values >= 1) and np.all(unique_values <= 9):
            tests_passed += 1
    
    # <==== insert your code above here
    return tests_passed
#end of activity 4

