from problem import Problem

class CombinationProblem(Problem):
    def __init__(self, tumblers=4, num_options=10):
        self.tumblers = tumblers
        self.num_options = num_options
        # This would be the correct combination
        self.goal = [1, 2, 3, 4]  # Example goal
        
    def get_goal(self):
        return self.goal
        
    def evaluate(self, attempt):
        if len(attempt) != self.tumblers:
            raise ValueError(f"Attempt must have {self.tumblers} tumblers")
        
        if not all(isinstance(x, int) for x in attempt):
            raise ValueError("All tumblers must be integers")
            
        # Score is number of correct positions
        score = sum(1 for a, g in zip(attempt, self.goal) if a == g)
        return score