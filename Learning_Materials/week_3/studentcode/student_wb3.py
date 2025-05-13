
from approvedimports import *

class DepthFirstSearch(SingleMemberSearch):
    """your implementation of depth first search to extend
    the superclass SingleMemberSearch search.
    Adds a __str__method
    Over-rides the method select_and_move_from_openlist
    to implement the algorithm
    """

    def __str__(self):
        return "depth-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """void in superclass
        In sub-classes should implement different algorithms
        depending on what item it picks from self.open_list
        and what it then does to the openlist

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        if not self.open_list:
            return None 
        # selecting the last item from openlist
        next_soln = self.open_list[-1]
        # remove it from openlist
        self.open_list.pop()

        return next_soln

class BreadthFirstSearch(SingleMemberSearch):
    """your implementation of breadth first search to extend
    the superclass SingleMemberSearch search.
    Adds a __str__method
    Over-rides the method select_and_move_from_openlist
    to implement the algorithm
    """

    def __str__(self):
        return "breadth-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements the breadth-first search algorithm

        Returns
        -------
        next working candidate (solution) taken from openlist
        """
        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # if openlist is empty, return None 
        if not self.open_list:
            return None
        # select the first item from openlist
        next_soln = self.open_list[0]
        # remove it from openlist
        self.open_list.pop(0)

        return next_soln

class BestFirstSearch(SingleMemberSearch):
    """Implementation of Best-First search."""

    def __str__(self):
        return "best-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements Best First by finding, popping and returning member from openlist
        with best quality.

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # IF IsEmpty (open_list) THEN RETURN None
        if not self.open_list:
            return None
        # ELSE
        best_index = 0
        best_quality = self.open_list[0].quality
        for i in range(len(self.open_list)):
            if self.open_list[i].quality < best_quality:
                best_index = i
                best_quality = self.open_list[i].quality

        next_soln = self.open_list[best_index]
        self.open_list.pop(best_index)

        return next_soln

class AStarSearch(SingleMemberSearch):
    """Implementation of A-Star search."""

    def __str__(self):
        return "A Star"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements A-Star by finding, popping and returning member from openlist
        with lowest combined length+quality.

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # IF IsEmpty (open_list) THEN RETURN None
        if not self.open_list:
            return None
        # ELSE
        best_index = 0
        best_score = self.open_list[0].quality + len(self.open_list[0].variable_values)
        for i in range(len(self.open_list)):
            score = self.open_list[i].quality + len(self.open_list[i].variable_values)
            if score < best_score:
                best_index = i
                best_score = score

        next_soln = self.open_list[best_index]
        self.open_list.pop(best_index)

        return next_soln

wall_colour = 0.0
hole_colour = 1.0

def create_maze_breaks_depthfirst():
    # Create a maze where DFS gets stuck in a long path while BFS finds a shorter path
    maze_breaks_depth = Maze(mazefile="maze.txt")

    # Create a long vertical tunnel to trap DFS
    for row in range(2, 15):  # From row 2 to row 14
        maze_breaks_depth.contents[row][10] = hole_colour  # Clear path
    # Add a dead-end at the bottom
    maze_breaks_depth.contents[15][10] = wall_colour

    # Save the maze
    maze_breaks_depth.save_to_txt("maze-breaks-depth.txt")

    # Comment out show_maze() for submission
    # maze_breaks_depth.show_maze()

def create_maze_depth_better():
    # Load original maze
    maze_depth_better = Maze(mazefile="maze.txt")

    # Create a direct horizontal path from start (row 2, col 9) towards the goal
    for col in range(10, 15):  # Clear path from col 10 to 14
        maze_depth_better.contents[2][col] = hole_colour

    # Add a few short branches to increase BFS trials
    # Branch 1: Down from row 2, col 11
    maze_depth_better.contents[3][11] = hole_colour
    maze_depth_better.contents[4][11] = wall_colour  # Dead-end
    # Branch 2: Down from row 2, col 12
    maze_depth_better.contents[3][12] = hole_colour
    maze_depth_better.contents[4][12] = wall_colour  # Dead-end

    # Save the maze
    maze_depth_better.save_to_txt("maze-depth-better.txt")

    # Comment out show_maze() for submission
    # maze_depth_better.show_maze()
