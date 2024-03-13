# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foods = newFood.asList()

        # Initialize variables for the nearest food and ghost calculations.
        foodDistances = [manhattanDistance(newPos, food) for food in foods]
        nearest_food_distance = min(foodDistances) if foodDistances else 0
        
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates if ghost.scaredTimer == 0]
        nearest_ghost_distance = min(ghostDistances) if ghostDistances else float('inf')
        
        scaredGhostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates if ghost.scaredTimer > 0]
        nearest_scared_ghost_distance = min(scaredGhostDistances) if scaredGhostDistances else float('inf')
        
        # Adjustments for scared ghosts - encouraging chasing if close and scared.
        if nearest_scared_ghost_distance != float('inf'):
            scared_ghost_score = 2.0 / nearest_scared_ghost_distance
        else:
            scared_ghost_score = 0
        
        # Basic score from the successor state.
        score = successorGameState.getScore()
        
        # Penalize or reward based on ghost distance - avoiding non-scared ghosts.
        if nearest_ghost_distance < 2:  # Immediate danger threshold
            ghost_score = -200
        else:
            ghost_score = -2.0 / nearest_ghost_distance
        
        # Encourage eating closer food.
        food_score = 0 if nearest_food_distance == 0 else 1.0 / nearest_food_distance
        
        # Combine all the components to form a final score.
        final_score = score + food_score + ghost_score + scared_ghost_score
        
        return final_score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0:  # Pacman's turn, maximize
                return max(minimax(1, depth, gameState.generateSuccessor(agentIndex, action)) for action in gameState.getLegalActions(agentIndex))
            else:  # Ghosts' turn, minimize
                nextAgent = agentIndex + 1
                if nextAgent >= gameState.getNumAgents():
                    nextAgent = 0  # Loop back to Pacman, increase depth
                    depth += 1
                return min(minimax(nextAgent, depth, gameState.generateSuccessor(agentIndex, action)) for action in gameState.getLegalActions(agentIndex))

        # Start the minimax process from Pacman (agentIndex 0) at the top level (depth 0)
        # and find the action that leads to the best outcome according to minimax
        maximumValue = float("-inf")
        actionToTake = Directions.STOP  # Default action
        for action in gameState.getLegalActions(0):  # Loop through Pacman's actions
            value = minimax(1, 0, gameState.generateSuccessor(0, action))
            if value > maximumValue:
                maximumValue = value
                actionToTake = action

        return actionToTake

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth * gameState.getNumAgents():
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0:  # Pacman's turn, maximize
                value = float("-inf")
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = max(value, alphaBeta(1, depth + 1, successor, alpha, beta))
                    alpha = max(alpha, value)
                    if alpha > beta:
                        break  # Beta cut-off
                return value
            else:  # Ghosts' turn, minimize
                value = float("inf")
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = min(value, alphaBeta(nextAgent, depth + 1, successor, alpha, beta))
                    beta = min(beta, value)
                    if beta < alpha:
                        break  # Alpha cut-off
                return value

        alpha = float("-inf")
        beta = float("inf")
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):  # Pacman is agent 0
            value = alphaBeta(1, 1, gameState.generateSuccessor(0, action), alpha, beta)
            if value > alpha:
                alpha = value
                bestAction = action
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None

            if agentIndex == 0:  # Pacman's turn, maximize
                return max((expectimax(1, depth, gameState.generateSuccessor(agentIndex, action))[0], action)
                           for action in gameState.getLegalActions(agentIndex))
            else:  # Ghosts' turn, expectimax
                nextAgent = agentIndex + 1
                if nextAgent >= gameState.getNumAgents():
                    nextAgent = 0
                    depth += 1
                actions = gameState.getLegalActions(agentIndex)
                return sum(expectimax(nextAgent, depth, gameState.generateSuccessor(agentIndex, action))[0]
                           for action in actions) / len(actions), None

        # Start the expectimax process from Pacman (agentIndex 0) at the top level (depth 0)
        return expectimax(0, 0, gameState)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    foodList = newFood.asList()

    # Calculate the score as the starting point of the evaluation
    score = currentGameState.getScore()

    # Adjust score based on the distance to the closest food
    if foodList:
        minFoodDist = min([util.manhattanDistance(newPos, food) for food in foodList])
        score += 1.0 / (minFoodDist + 1)

    # Adjust score based on the distance to ghosts and their scared state
    for ghost, scaredTime in zip(newGhostStates, newScaredTimes):
        distToGhost = util.manhattanDistance(newPos, ghost.getPosition())
        if scaredTime > 0:  # If the ghost is scared
            score += max(scaredTime - distToGhost, 0) * 2
        else:
            if distToGhost <= 1:  # Very close ghosts are dangerous
                score -= 200

    # Consider the number of remaining food pellets
    score -= len(foodList) * 2

    # Consider the number of remaining power pellets
    score -= len(currentGameState.getCapsules()) * 10

    return score

# Abbreviation
better = betterEvaluationFunction
