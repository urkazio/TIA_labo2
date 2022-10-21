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
import sys

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
        # Useful information you can extract from a gameState (pacman.py)
        successorgameState = currentGameState.generatePacmanSuccessor(action)  # estado sucesor tras una accion
        oldFoodNum = currentGameState.getFood().count()  # numero de comidas disponibles antes de la accion

        """ --- DATOS DESPUES DE EJECUTAR LA ACCION --- """
        newPos = successorgameState.getPacmanPosition()  # coordenada tras la accion
        newFood = successorgameState.getFood()  # matriz bool con comidas
        newGhostStates = successorgameState.getGhostStates() # posicion de los fantamas
        newListFood = newFood.asList()  # coord de comidas DESPUES de la accion
        newFoodNum = len(newListFood)
        minDistComida = sys.maxsize
        minDistFantasma = sys.maxsize

        # cuanto menor el score --> mayor prirodiad del estado (pensar en cola proritaria)
        # cuanto menor distancia a la comida mas cercana --> MEJOR
        # cuanto mayor distancia al fantasma mas cercano --> MEJOR

        if newFoodNum == oldFoodNum:  # si la accion no come punto
            for food in newListFood:
                if manhattanDistance(food, newPos) < minDistComida:
                    minDistComida = manhattanDistance(food, newPos)  # quedarse con la distancia minima a una comida
        else:
            minDistComida = 0

        for ghost in newGhostStates:  # the impact of ghost surges as distance get close
            if manhattanDistance(ghost.getPosition(), newPos) < minDistFantasma:
                minDistFantasma = manhattanDistance(ghost.getPosition(), newPos)  # quedarse con la distancia minima a un fantasma

        # primero hago el inverso porque quiero que cuanto mas grande la distancia mejor sea el resultado.
        # multiplico por 100 para darle algo mas de peso a la distancia respecto al fantasma en relacion a la distancia con la comida
        minDistFanstasmaEscalada = 100*4**-minDistFantasma

        return -(minDistComida+minDistFanstasmaEscalada)

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
        # ---------------------- COSAS A TENER EN CUENTA ---------------------
        # hacer el arbol de profundidad depth
        # evaluar las hojas con scoreEvaluationFunction
        # el pacman es el agente 0, el resto se supone que son fantasmas
        # PUEDE HABER VARIOS FANTASMAS !!!!
        # Una capa de profundidad implica el movimiento del pacman y de todos los fantasmas
        # ---------------------------------------------------------------------

        maxScore = -sys.maxsize  #-inf
        bestAction = "Stop"  # empezar MANUALMENTE con el pacman haciendo algun movimiento (el de menor riesgo es estar parado)
        actions = gameState.getLegalActions(0) # se empieza con el pacman

        for action in actions:
            successor = gameState.generateSuccessor(0, action)  # se arranca el algoritmo con el primer fantasma (el primer movimiento ha sido manual)
            utility = self.minValue(0, 1, successor)  # empezar el algoritmo con el primer fantasma
            if utility > maxScore:
                maxScore = utility
                bestAction = action  # guardar la accion cuya utility es la mayor hasta la fecha

        return bestAction


    def maxValue(self, depth, agentid, state):

        v = -sys.maxsize #-inf

        # --- parte no recursiva ---
        if depth == self.depth:  # comprobar que si se ha llegado al depth arbitrario
            return self.evaluationFunction(state)
        else:
            # se deben obtener los estados sucesores a partir de las acciones legales:
            actions = state.getLegalActions(agentid)
            if len(actions) == 0:  # si el nodo es hoja
                return self.evaluationFunction(state)

            # --- parte recursiva ---
            for action in actions:
                successor = state.generateSuccessor(agentid, action)
                v = max(v, self.minValue(depth, agentid+1, successor))
            return v


    def minValue(self, depth, agentid, state):

        v = sys.maxsize #inf

        # --- parte no recursiva ---
        if depth == self.depth:  # comprobar que si se ha llegado al depth arbitrario
            return self.evaluationFunction(state)
        else:
            # se deben obtener los estados sucesores a partir de las acciones legales:
            actions = state.getLegalActions(agentid)
            if len(actions) == 0:  # si el nodo es hoja
                return self.evaluationFunction(state)

            # --- parte recursiva ---
            for action in actions:
                if agentid == state.getNumAgents()-1:  # si ya se han recorrido todos los fantasmas del turno...
                    # --> le vuelve a tocar al pacman y se avanza una profundidad
                    successor = state.generateSuccessor(agentid, action)
                    v = min(v, self.maxValue(depth+1, 0, successor))
                else:  # si el siguiente turno es de otro fantasma...
                    successor = state.generateSuccessor(agentid, action)
                    v = min(v, self.minValue(depth, agentid+1, successor))
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
        maxScore = -sys.maxsize  # -inf
        alfa = -sys.maxsize  # -inf
        beta = sys.maxsize  # inf
        bestAction = "Stop"  # empezar MANUALMENTE con el pacman haciendo algun movimiento (el de menor riesgo es estar parado)
        actions = gameState.getLegalActions(0)  # se empieza con el pacman

        for action in actions:
            successor = gameState.generateSuccessor(0, action)  # se arranca el algoritmo con el primer fantasma (el primer movimiento ha sido manual)
            utility = self.minValue(0, 1, successor, alfa, beta)  # empezar el algoritmo con el primer fantasma
            if utility > maxScore:
                maxScore = utility
                bestAction = action  # guardar la accion cuya utility es la mayor hasta la fecha

        return bestAction

    def maxValue(self, depth, agentid, state, alfa, beta):

        v = -sys.maxsize  # -inf

        # --- parte no recursiva ---
        if depth == self.depth:  # comprobar que si se ha llegado al depth arbitrario
            return self.evaluationFunction(state)
        else:
            # se deben obtener los estados sucesores a partir de las acciones legales:
            actions = state.getLegalActions(agentid)
            if len(actions) == 0:  # si el nodo es hoja
                return self.evaluationFunction(state)

            # --- parte recursiva ---
            for action in actions:
                successor = state.generateSuccessor(agentid, action)
                v = max(v, self.minValue(depth, agentid + 1, successor, alfa, beta))
                if v >= beta:
                    return v
                alfa = max(alfa, v)
            return v

    def minValue(self, depth, agentid, state, alfa, beta):

        v = sys.maxsize  # inf

        # --- parte no recursiva ---
        if depth == self.depth:  # comprobar que si se ha llegado al depth arbitrario
            return self.evaluationFunction(state)
        else:
            # se deben obtener los estados sucesores a partir de las acciones legales:
            actions = state.getLegalActions(agentid)
            if len(actions) == 0:  # si el nodo es hoja
                return self.evaluationFunction(state)

            # --- parte recursiva ---
            for action in actions:
                if agentid == state.getNumAgents() - 1:  # si ya se han recorrido todos los fantasmas del turno...
                    # --> le vuelve a tocar al pacman y se avanza una profundidad
                    successor = state.generateSuccessor(agentid, action)
                    v = min(v, self.maxValue(depth + 1, 0, successor, alfa, beta))
                    if v <= alfa:
                        return v
                    beta = min(beta, v)
                else:  # si el siguiente turno es de otro fantasma...
                    successor = state.generateSuccessor(agentid, action)
                    v = min(v, self.minValue(depth, agentid + 1, successor, alfa, beta))
                    if v <= alfa:
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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
