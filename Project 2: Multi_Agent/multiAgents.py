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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        foodList = currentGameState.getFood().asList()
        minGhostDistance = min(manhattanDistance(newPos, ghost.configuration.pos) for ghost in newGhostStates)
        minFoodDistance = min(manhattanDistance(newPos, nextFood) for nextFood in foodList) if foodList else 0
        scaredTime = min(newScaredTimes)

        if scaredTime == 0:
            F1 = -1 / (minGhostDistance + 1)
        else:
            F1 = 0.5 / (minGhostDistance + 1)
            
        F2 = -len(foodList)
        F3 = 0.5 / (minFoodDistance + 1)
        F4 = scaredTime * 0.5
        F5 = currentGameState.getScore()

        return F1 + F2 + F3 + F4 + F5
        
def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        agentIndex = 0
        legalActions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]
        v = -float('inf')
        i = 0
        for x in range(0, len(successors)):
            scores = self.Minimax(successors[x], 1, 0)
            if scores > v:
                v = scores
                i = x
        return legalActions[i]
                
                
    def Minimax(self, gameState, agentIndex, depth):
        if depth == self.depth or gameState.getLegalActions(agentIndex) == 0 or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState))
                
        if agentIndex == 0:
            legalActions = gameState.getLegalActions(agentIndex)
            successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]
            v = -float('inf')
            for successor in successors:
                v = max(v, self.Minimax(successor, 1, depth))
            return v
            
        if agentIndex > 0:
            legalActions = gameState.getLegalActions(agentIndex)
            successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]
            v = float('inf')
            for successor in successors:
                if (agentIndex + 1) %  gameState.getNumAgents() == 0:
                    v = min(v, self.Minimax(successor, 0, depth + 1))
                else:
                    v = min(v, self.Minimax(successor, agentIndex + 1, depth))
            return v



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agentIndex = 0
        alpha = -float('inf')
        beta = float('inf')
        legalActions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]
        v = -float('inf')
        i = 0
        for x in range(0, len(successors)):
            scores = self.alpha_beta(successors[x], 1, 0, alpha, beta)
            if scores > v:
                v = scores
                i = x
                alpha = scores
        return legalActions[i]
        
    def alpha_beta(self, gameState, agentIndex, depth, alpha, beta):
        if depth == self.depth or gameState.getLegalActions(agentIndex) == 0 or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState))
                
        if agentIndex == 0:
            legalActions = gameState.getLegalActions(agentIndex)
            v = -float('inf')
            for action in legalActions:
                successors = gameState.generateSuccessor(agentIndex, action)
                v = max(v, self.alpha_beta(successors, 1, depth, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
            
        if agentIndex > 0:
            legalActions = gameState.getLegalActions(agentIndex)
            v = float('inf')
            for action in legalActions:
                successors = gameState.generateSuccessor(agentIndex, action)
                if (agentIndex + 1) %  gameState.getNumAgents() == 0:
                    v = min(v, self.alpha_beta(successors, 0, depth + 1, alpha, beta))
                else:
                    v = min(v, self.alpha_beta(successors, agentIndex + 1, depth, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        agentIndex = 0
        legalActions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]
        v = -float('inf')
        i = 0
        for x in range(0, len(successors)):
            scores = self.Expectimax(successors[x], 1, 0)
            if scores > v:
                v = scores
                i = x
        return legalActions[i]
      
      
    def Expectimax(self, gameState, agentIndex, depth):
        if depth == self.depth or gameState.getLegalActions(agentIndex) == 0 or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState))
                
        if agentIndex == 0:
            legalActions = gameState.getLegalActions(agentIndex)
            successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]
            v = -float('inf')
            for successor in successors:
                v = max(v, self.Expectimax(successor, 1, depth))
            return v
            
        if agentIndex > 0:
            legalActions = gameState.getLegalActions(agentIndex)
            successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]
            v = 0
            for successor in successors:
                if (agentIndex + 1) %  gameState.getNumAgents() == 0:
                    v += self.Expectimax(successor, 0, depth + 1)
                else:
                    v += self.Expectimax(successor, agentIndex + 1, depth)
            return float(v)/len(successors)

def betterEvaluationFunction(currentGameState):
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

    foodList = currentGameState.getFood().asList()
    minGhostDistance = min(manhattanDistance(newPos, ghost.configuration.pos) for ghost in newGhostStates)
    minFoodDistance = min(manhattanDistance(newPos, nextFood) for nextFood in foodList) if foodList else 0
    scaredTime = min(newScaredTimes)

    if scaredTime == 0:
        F1 = -1 / (minGhostDistance + 1)
    else:
        F1 = 0.5 / (minGhostDistance + 1)
            
    F2 = -len(foodList)
    F3 = 0.5 / (minFoodDistance + 1)
    F4 = scaredTime * 0.5
    F5 = currentGameState.getScore()

    return F1 + F2 + F3 + F4 + F5


# Abbreviation
better = betterEvaluationFunction
