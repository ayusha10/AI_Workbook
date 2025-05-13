class Problem:
    def evaluate(self, attempt):
        raise NotImplementedError("Evaluate method must be implemented by subclass")
    
    def get_goal(self):
        raise NotImplementedError("Get_goal method must be implemented by subclass")