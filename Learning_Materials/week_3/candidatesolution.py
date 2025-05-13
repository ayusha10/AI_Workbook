class CandidateSolution:
    """
    Represents a candidate solution in a search problem.
    """
    
    def __init__(self, values=None, quality=0):
        """
        Initialize a candidate solution.
        
        Args:
            values: The sequence of values representing the solution
            quality: The quality/fitness of the solution
        """
        self.values = values if values is not None else []
        self.quality = quality
    
    def __str__(self):
        return f"Solution(values={self.values}, quality={self.quality})"
    
    def __repr__(self):
        return self.__str__()
    
    def copy(self):
        """
        Create a deep copy of this solution.
        """
        return CandidateSolution(list(self.values), self.quality)