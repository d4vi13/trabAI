# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections
class ValueIterationAgent(ValueEstimationAgent):

    def __init__(self, mdp, discount = 0.9, iterations = 100):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()
        self.runValueIteration()

    def runValueIteration(self):
        for step in range(self.iterations):
            next_values = util.Counter()
            for s in self.mdp.getStates():
                if self.mdp.isTerminal(s):
                    continue
                actions = self.mdp.getPossibleActions(s)
                if len(actions) == 0:
                    continue
                best_q = -999999999
                for a in actions:
                    current_q = self.computeQValueFromValues(s, a)
                    if current_q > best_q:
                        best_q = current_q
                next_values[s] = best_q
            self.values = next_values


    def getValue(self, state):
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        total = 0.0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in transitions:
            r = self.mdp.getReward(state, action, nextState)
            val = self.values[nextState]
            total = total + prob * (r + self.discount * val)
        return total

    def computeActionFromValues(self, state):
        if self.mdp.isTerminal(state):
            return None

        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None

        chosen_action = None
        highest_q = -999999999

        for a in actions:
            q = self.computeQValueFromValues(state, a)
            if q > highest_q:
                highest_q = q
                chosen_action = a

        return chosen_action
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        preds = {}
        all_states = self.mdp.getStates()
        for s in all_states:
            preds[s] = set()

        for s in all_states:
            if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)
                for a in actions:
                    transitions = self.mdp.getTransitionStatesAndProbs(s, a)
                    for nextState, prob in transitions:
                        if prob > 0:
                            preds[nextState].add(s)

        pq = util.PriorityQueue()

        for s in all_states:
            if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)
                best_q = -999999999
                for a in actions:
                    q = self.computeQValueFromValues(s, a)
                    if q > best_q:
                        best_q = q
                diff = abs(self.values[s] - best_q)
                pq.push(s, -diff)

        for step in range(self.iterations):
            if pq.isEmpty():
                break

            s = pq.pop()

            if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)
                best_q = -999999999
                for a in actions:
                    q = self.computeQValueFromValues(s, a)
                    if q > best_q:
                        best_q = q
                self.values[s] = best_q

            for p in preds[s]:
                if not self.mdp.isTerminal(p):
                    p_actions = self.mdp.getPossibleActions(p)
                    best_q = -999999999
                    for a in p_actions:
                        q = self.computeQValueFromValues(p, a)
                        if q > best_q:
                            best_q = q
                    diff = abs(self.values[p] - best_q)
                    if diff > self.theta:
                        pq.update(p, -diff)
