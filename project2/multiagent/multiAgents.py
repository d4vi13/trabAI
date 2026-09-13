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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()

        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)
            if ghostDist <= 1 and ghostState.scaredTimer == 0:
                return -float('inf')  

        foodList = newFood.asList()
        if foodList:
            minFoodDist = min([manhattanDistance(newPos, food) for food in foodList])
            score += 1.0 / minFoodDist

        return score

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
    def getAction(self, gameState):
        def minimax(state, depth, agent_idx):
            if agent_idx == state.getNumAgents():
                agent_idx = 0
                depth += 1

            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            actions = state.getLegalActions(agent_idx)
            if not actions:
                return self.evaluationFunction(state), None

            if agent_idx == 0:
                best_score = float('-inf')
                best_action = None
                for action in actions:
                    successor = state.generateSuccessor(agent_idx, action)
                    score, _ = minimax(successor, depth, agent_idx + 1)
                    if score > best_score:
                        best_score, best_action = score, action
                return best_score, best_action

            else:
                best_score = float('inf')
                best_action = None
                for action in actions:
                    successor = state.generateSuccessor(agent_idx, action)
                    score, _ = minimax(successor, depth, agent_idx + 1)
                    if score < best_score:
                        best_score, best_action = score, action
                return best_score, best_action

        _, action = minimax(gameState, 0, 0)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        """
        Returns the minimax action using alpha-beta pruning.
        """
        def alpha_beta(state, depth, agent_idx, alpha, beta):
            if agent_idx == state.getNumAgents():
                agent_idx = 0
                depth += 1

            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            actions = state.getLegalActions(agent_idx)
            if not actions:
                return self.evaluationFunction(state), None

            if agent_idx == 0:
                value = float('-inf')
                best_action = None
                for action in actions:
                    successor = state.generateSuccessor(agent_idx, action)
                    score, _ = alpha_beta(successor, depth, agent_idx + 1, alpha, beta)
                    if score > value:
                        value, best_action = score, action
                    if value > beta:  # Prune
                        return value, best_action
                    alpha = max(alpha, value)
                return value, best_action

            else:
                value = float('inf')
                best_action = None
                for action in actions:
                    successor = state.generateSuccessor(agent_idx, action)
                    score, _ = alpha_beta(successor, depth, agent_idx + 1, alpha, beta)
                    if score < value:
                        value, best_action = score, action
                    if value < alpha:  # Prune
                        return value, best_action
                    beta = min(beta, value)
                return value, best_action

        _, action = alpha_beta(gameState, 0, 0, float('-inf'), float('inf'))
        return action

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
