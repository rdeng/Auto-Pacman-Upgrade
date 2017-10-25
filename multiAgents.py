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
from sys import maxsize
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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        foodDis = 100000

        for x in range(newFood.width):
            for y in range(newFood.height):
                if newFood[x][y] is True:
                    if(util.manhattanDistance(newPos, (x, y)) < foodDis):
                        foodDis = util.manhattanDistance(newPos, (x, y))

        ghostDis = 100000
        for ghost in newGhostStates:
            if ghostDis > util.manhattanDistance(newPos, ghost.getPosition()):
                ghostDis = util.manhattanDistance(newPos, ghost.getPosition())

        if ghostDis < 2:
            return -100000
        if foodDis < 1:
            return 1000000
        return ghostDis/foodDis + successorGameState.getScore()


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
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(self.index)
        val = []
        for action in actions:
            val.append(self.minimaxFunction(gameState.generateSuccessor(self.index, action),
                                    (self.depth * gameState.getNumAgents()) - 1, 1,))

        bestVal = max(val)
        indices = [index for index in range(len(val)) if val[index] == bestVal]
        return actions[random.choice(indices)]

    def minimaxFunction(self, state, depth, pacOrGhost):

        if pacOrGhost == state.getNumAgents():
            pacOrGhost = 0
            #do we need to decrement depth???

        if depth is 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        possibleAct = state.getLegalActions(pacOrGhost)

        if pacOrGhost is 0:
            bestVal = maxsize * -1
            for action in possibleAct:
                upperL = state.generateSuccessor(pacOrGhost, action)
                v = self.minimaxFunction(upperL, depth - 1, pacOrGhost + 1)
                bestVal = max(bestVal, v)
            return bestVal

        else:
            bestVal = maxsize
            for action in possibleAct:
                upperL = state.generateSuccessor(pacOrGhost, action)
                v = self.minimaxFunction(upperL, depth - 1, pacOrGhost + 1)
                bestVal = min(bestVal, v)
            return bestVal

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.val(gameState, -float("inf"), float("inf"), self.index, 0)[1]

    def val(self, gameState, a, b, agentIndex, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        if agentIndex % gameState.getNumAgents() == 0 and agentIndex != 0:
            agentIndex = 0
            depth += 1
        if depth == self.depth:
            return self.evaluationFunction(gameState), None
        if agentIndex % gameState.getNumAgents() == 0:
            return self.maxValue(gameState, a, b, agentIndex, depth)
        else:
            return self.minValue(gameState, a, b, agentIndex, depth)

    def maxValue(self, gameState, a, b, agentIndex, depth):
        v = (-float("inf"), None)
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            newV = self.val(successor, a, b, agentIndex+1, depth)
            vIns = max(newV[0], v[0])
            if vIns == newV[0]:
                v = (newV[0], action)
            if v[0] > b:
                return v
            a = max(a, v[0])
        return v

    def minValue(self, gameState, a, b, agentIndex, depth):
        v = (float("inf"), None)
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            newV = self.val(successor, a, b, agentIndex+1, depth)
            vIns = min(newV[0], v[0])
            if vIns == newV[0]:
                v = (newV[0], action)
            if v[0] < a:
                return v
            b = min(b, v[0])
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
        return self.val(gameState, self.index, 0)[1]

    def val(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        if agentIndex % gameState.getNumAgents() == 0 and agentIndex != 0:
            agentIndex = 0
            depth += 1
        if depth == self.depth:
            return self.evaluationFunction(gameState), None
        if agentIndex % gameState.getNumAgents() == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.expValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        v = (-float("inf"), None)
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            newV = self.val(successor, agentIndex+1, depth)
            vIns = max(newV[0], v[0])
            if vIns == newV[0]:
                v = (newV[0], action)
        return v

    def expValue(self, gameState, agentIndex, depth):
        values = 0.0
        num = 0.0
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            newV = self.val(successor, agentIndex+1, depth)
            vIns = newV[0]
            values += vIns
            num += 1.0
        # since totally random, weights are the same
        toReturn = values/num
        return toReturn, None

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

    "*** YOUR CODE HERE ***"
    foodDis = 100000
    foodTot = 0
    for x in range(newFood.width):
        for y in range(newFood.height):
            if newFood[x][y] is True:
                foodTot += util.manhattanDistance(newPos, (x, y))
                if(util.manhattanDistance(newPos, (x, y)) < foodDis):
                    foodDis = util.manhattanDistance(newPos, (x, y))

    ghostDis = 100000
    for ghost in newGhostStates:
        if ghostDis > util.manhattanDistance(newPos, ghost.getPosition()):
            ghostDis = util.manhattanDistance(newPos, ghost.getPosition())

    if ghostDis < 2:
        return -100000
    if foodDis < 1:
        return 1000000
    return ghostDis/foodDis + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction

