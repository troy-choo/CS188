# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    return generalizedSearch(problem, 'dfs')

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    return generalizedSearch(problem, 'bfs')

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    return generalizedSearch(problem, 'ucs')

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    return generalizedSearch(problem, 'astar', heuristic)

def generalizedSearch(problem, searchType, heuristic=None):
    """
    A generalized search function that can perform various search strategies 
    including DFS, BFS, UCS, and A* based on the specified searchType and heuristic.
    """
    # Set the heuristic to a default lambda function if none is provided
    heuristic = heuristic or (lambda state, problem=None: 0)
    
    # Determine the type of data structure to use for the frontier
    if searchType == 'dfs':
        frontier = util.Stack()
    elif searchType == 'bfs':
        frontier = util.Queue()
    elif searchType in ['ucs', 'astar']:
        frontier = util.PriorityQueue()
    else:
        raise ValueError("Unsupported search strategy specified.")
    
    # Initialize the set of explored states
    explored = set()
    # Initialize the frontier with the start state
    initial_state = problem.getStartState()
    initial_entry = (initial_state, [], 0)  # State, path, and cost
    if searchType in ['ucs', 'astar']:
        frontier.push(initial_entry, 0)
    else:
        frontier.push(initial_entry)
    
    # Main search loop
    while not frontier.isEmpty():
        current_state, path_so_far, current_cost = frontier.pop()
        
        if problem.isGoalState(current_state):
            return path_so_far
        
        if current_state not in explored:
            explored.add(current_state)
            successors = problem.getSuccessors(current_state)
            
            for successor, action, step_cost in successors:
                if successor not in explored:
                    new_path = path_so_far + [action]
                    new_cost = current_cost + step_cost if searchType in ['ucs', 'astar'] else 0
                    if searchType == 'dfs' or searchType == 'bfs':
                        frontier.push((successor, new_path, new_cost))
                    elif searchType == 'ucs':
                        frontier.push((successor, new_path, new_cost), new_cost)
                    elif searchType == 'astar':
                        priority = new_cost + heuristic(successor, problem)
                        frontier.push((successor, new_path, new_cost), priority)
    
    # Return an empty list if no solution is found
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
