from utils import *
from search import *


class Fruits(Problem):

    def __init__(self):
        # Complete the implementation of initial function by calling
        # the parent constructor and initialize with the initial state
        # Write your code below this line!
        super().__init__((13, 46, 59))
        # Write your code above this line! Delete the 'pass' keyword!

    def actions(self, state):
        # Complete the implementation of possible operators.
        # The function should return all next possible actions from the input state.
        acts = []
        # Write your code below this line!
        for i in range(len(state)):
            for j in range(len(state)):
                if i != j and state[i] > 0 and state[j] > 0:
                    action = (i+1, j+1)
                    acts.append(action)
        # Write your code above this line!
        return acts

    def result(self, state, action):
        # Fill in the missing parts of the transition function
        i, j = action
        # Write your code below this line!
        new_state = list(state)
        i -= 1
        j -= 1
        for k in range(len(state)):
            if (k == i or k == j) and state[k] > 0:
                new_state[k] = state[k] - 1
            else:
                new_state[k] = state[k] + 2
        # Write your code above this line!
        return tuple(new_state)

    def goal_test(self, state):
        # Write a logic here to test if the state is a goal state. Return True if it is a goal state, False if not
        # Write your code below this line!
        if state[0] + state[1] == 0 or state[0] + state[2] == 0 or state[1] + state[2] == 0:
            return True
        return False
        # Write your code above this line! Delete the 'pass' keyword.


def depth_first_graph_search(problem):
    # Write your code below this line!
    frontier = [(Node(problem.initial))]
    res_path = []
    explored = set()
    while frontier:
        node = frontier.pop()
        res_path.append(node)
        if problem.goal_test(node.state):
            print(res_path)
            return node
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and child not in frontier)
    return None
    # Write your code above this line! Delete the 'pass' keyword.


def main():

    # 1. Exercise: Fill in the missing parts of init function and print out the initial state result (1 point)
    # Write your code below this line!
    fruit_trader = Fruits()
    print(fruit_trader.initial)
    # Write your code above this line! Delete the 'pass' keyword.

    # 2. Exercise: Fill out the goal_test function and test if it works correctly (test for (49, 0, 0) and (5, 6, 6)
    # Write your code below this line!
    print(fruit_trader.goal_test((49, 0, 0)))
    print(fruit_trader.goal_test((5, 6, 6)))
    # Write your code above this line!
    # 3. Exercise: Fill out the actions function and test if it works correctly (test using the initial state)
    # Write your code below this line!
    print(fruit_trader.actions(fruit_trader.initial))
    # Write your code above this line!
    # 4. Exercise: Fill out the result function and test if it works correctly
    # (use the initial state and (2, 3) as the action when testing)
    # Write your code below this line!
    print(fruit_trader.result(fruit_trader.initial, (2, 3)))
    # Write your code above this line!
    # 5. Fill out the depth_first_graph_search function and solve the problem using it
    # Write your code below this line!
    print(depth_first_graph_search(fruit_trader))
    # Write your code above this line!


if __name__ == '__main__':
    main()
