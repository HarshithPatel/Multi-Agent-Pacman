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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # print "legalMoves %s" %legalMoves
        # Choose one of the best actions

        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        # print scores
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
        # Get the food list
        foodAsList = newFood.asList()
        # Return a max score if no more food available
        if len(foodAsList)==0:
            return 100000
        stopScore = 0
        # Get the nearest food
        manhattanFood = min([util.manhattanDistance(newPos,food) for food in foodAsList])
        # Find the nearest ghost
        manhattanGhost = min([util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
        # Discourage stop action
        if action == 'Stop':
            stopScore = 5
        # Ignore ghost when scared
        scaredTime = min(newScaredTimes)
        if scaredTime > 0:
            manhattanGhost = 1
        # Reducing ghost preference by 50% and full preference to food
        totalScore = successorGameState.getScore() + (manhattanGhost * 0.5 /(manhattanFood+stopScore))
        return totalScore


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

        # Defining a max agent to choose the best maximum action after every ghost actions
        def max_agent(gameState, depth, actionFlag):
            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()
            maxAgentValues=[-100000,None]
            pacmanAgent = 0
            firstGhost = 1
            for pacmanAction in gameState.getLegalActions(0):
                pacmanGameState = gameState.generateSuccessor(pacmanAgent, pacmanAction)
                score = min_agent(pacmanGameState, depth, firstGhost)
                # Pick the max score from the returned min values
                if score > maxAgentValues[0]:
                    maxAgentValues=[score,pacmanAction]
            # Return the action when the iteration is over
            if actionFlag == 1:
                return maxAgentValues[1]
            return maxAgentValues[0]

        # Defining a min agent that chooses a best ghost min action that could minimise the score
        def min_agent(gameState, depth, ghost):
            if gameState.isLose() or gameState.isWin():
                return gameState.getScore()
            lastGhost = gameState.getNumAgents() - 1
            min_score = 100000
            for ghostAction in gameState.getLegalActions(ghost):
                if ghost != lastGhost:
                    ghostState = gameState.generateSuccessor(ghost, ghostAction)
                    score = min_agent(ghostState, depth, ghost + 1)
                elif ghost == lastGhost and depth == self.depth - 1:
                    lastGhostAtDepthState = gameState.generateSuccessor(lastGhost, ghostAction)
                    score = self.evaluationFunction(lastGhostAtDepthState)
                else:
                    lastGhostState = gameState.generateSuccessor(lastGhost, ghostAction)
                    score = max_agent(lastGhostState, depth + 1, 0)
                # Pick the min value from the max scores
                if score < min_score:
                    min_score = score
            return min_score
        return max_agent(gameState, 0, 1)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Defining a max agent to choose the best maximum action after every ghost actions
        def max_agent(gameState, depth, actionFlag, alphaVal, betaVal):
            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()
            maxAgentValues=[-100000,None]
            pacmanAgent = 0
            firstGhost = 1
            for pacmanAction in gameState.getLegalActions(0):
                pacmanGameState = gameState.generateSuccessor(pacmanAgent, pacmanAction)
                score = min_agent(pacmanGameState, depth, firstGhost, alphaVal, betaVal)
                # Pick the max score from the returned min values
                if score > maxAgentValues[0]:
                    maxAgentValues=[score,pacmanAction]
                alphaVal = max(maxAgentValues[0],alphaVal)
                # Performing beta pruning. Stop the iteration if the score is bigger then beta value
                if maxAgentValues[0] > betaVal:
                    return maxAgentValues[0]
            # Retun action when iteration is over
            if actionFlag == 1:
                return maxAgentValues[1]
            return maxAgentValues[0]

        # Defining a min agent that chooses a best ghost min action that could minimise the score
        def min_agent(gameState, depth, ghost, alphaVal, betaVal):
            if gameState.isLose() or gameState.isWin():
                return gameState.getScore()
            lastGhost = gameState.getNumAgents() - 1
            min_score = 100000
            for ghostAction in gameState.getLegalActions(ghost):
                if ghost != lastGhost:
                    ghostState = gameState.generateSuccessor(ghost, ghostAction)
                    score = min_agent(ghostState, depth, ghost + 1,  alphaVal, betaVal)
                elif ghost == lastGhost and depth == self.depth - 1:
                    lastGhostAtDepthState = gameState.generateSuccessor(lastGhost, ghostAction)
                    score = self.evaluationFunction(lastGhostAtDepthState)
                else:
                    lastGhostState = gameState.generateSuccessor(lastGhost, ghostAction)
                    score = max_agent(lastGhostState, depth + 1, 0,  alphaVal, betaVal)
                # Pick the min value from the max scores
                if score < min_score:
                    min_score = score
                betaVal = min(min_score,betaVal)
                # Performing alpha pruning. Stop the iteration if the score is smaller then alpha value
                if min_score<alphaVal:
                    return min_score

            return min_score
        return max_agent(gameState, 0, 1,float('-inf'),float('+inf'))



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

        # Defining a max agent to choose the best maximum action after every ghost actions
        def max_agent(gameState, depth, actionFlag):
            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()
            maxAgentValues=[-100000,None]
            pacmanAgent = 0
            firstGhost = 1
            for pacmanAction in gameState.getLegalActions(0):
                pacmanGameState = gameState.generateSuccessor(pacmanAgent, pacmanAction)
                score = min_agent(pacmanGameState, depth, firstGhost)
                if score > maxAgentValues[0]:
                    maxAgentValues=[score,pacmanAction]
            if actionFlag == 1:
                return maxAgentValues[1]
            return maxAgentValues[0]

        # Defining a min agent that chooses a best ghost min action that could minimise the score
        def min_agent(gameState, depth, ghost):
            if gameState.isLose() or gameState.isWin():
                return gameState.getScore()
            lastGhost = gameState.getNumAgents() - 1
            averageMinScore = 0
            lengthOfActions = len(gameState.getLegalActions(ghost))
            for ghostAction in gameState.getLegalActions(ghost):
                if ghost != lastGhost:
                    ghostState = gameState.generateSuccessor(ghost, ghostAction)
                    # Averaging out to get the expectimax score
                    averageMinScore += min_agent(ghostState, depth, ghost + 1) / lengthOfActions
                elif ghost == lastGhost and depth == self.depth - 1:
                    lastGhostAtDepthState = gameState.generateSuccessor(lastGhost, ghostAction)
                    # Averaging out to get the expectimax score
                    averageMinScore += self.evaluationFunction(lastGhostAtDepthState) / lengthOfActions
                else:
                    lastGhostState = gameState.generateSuccessor(lastGhost, ghostAction)
                    # Averaging out to get the expectimax score
                    averageMinScore += max_agent(lastGhostState, depth + 1, 0)/ lengthOfActions

            return averageMinScore
        return max_agent(gameState, 0, 1)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      Evaluated the floowing
      1. Nearest food
      2. Nearest ghost
      3. Nearest capsule
      Feature preference is given in the following way:
      1. Eat the Nearest Food
      2. Dont go long way for Capusles (50% preference)
      3. Dont run far away from Ghost (50% preference)
    """
    "*** YOUR CODE HERE ***"
    pacManPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    foodAsList = newFood.asList()
    capsuleNearest = 0
    if len(foodAsList) == 0:
        return 100000
    manhattanFood = min([util.manhattanDistance(pacManPos, food) for food in foodAsList])
    manhattanGhost = min([util.manhattanDistance(pacManPos, ghost.getPosition()) for ghost in newGhostStates])
    capsules = [util.manhattanDistance(pacManPos, capsules) for capsules in currentGameState.getCapsules()]
    if len(capsules) > 0:
        capsuleNearest = min(capsules)
    scaredTime = min(newScaredTimes)
    if scaredTime > 0:
        manhattanGhost = 1

      # 1. Eat the Nearest Food (100% preference), Eat Capusles (50% preference), Keep safe from Ghost (50% preference)
    totalScore = currentGameState.getScore() + (manhattanGhost * 0.5 / manhattanFood) - 0.5*capsuleNearest
    return totalScore

# Abbreviation
better = betterEvaluationFunction

