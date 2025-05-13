from search import Problem, Trial_Error, Node
from collections import deque


def convert_state_to_list(state_tuple):
    return [list(x) for x in state_tuple]


def convert_state_to_tuple(state_list):
    return tuple([tuple(x) for x in state_list])


def print_matrix(matrix):
    for i in range(len(matrix)):
        print(matrix[i])


class FourQueensProblem(Problem):
    def __init__(self):
        super().__init__(((0, 0, 0, 0),
                          (0, 0, 0, 0),
                          (0, 0, 0, 0),
                          (0, 0, 0, 0)))

    def actions(self, state):
        acts = []
        for i in range(4):
            for j in range(4):
                if state[i][j] == 0:
                    acts.append("o {} {}".format(i + 1, j + 1))
        return acts

    def result(self, state, action):
        i, j = int(action.split(' ')[1]) - 1, int(action.split(' ')[2]) - 1
        new_state = convert_state_to_list(state)

        for l in range(4):
            for k in range(4):
                if l == i and k == j:
                    new_state[l][k] = 1
                elif (l == i or k == j or abs(i - l) == abs(j - k)) and not (l == i and k == j):
                    new_state[l][k] = 2

        return convert_state_to_tuple(new_state)

    def goal_test(self, state):
        bool_state = convert_state_to_list(state)

        for i in range(4):
            for j in range(4):
                bool_state[i][j] = state[i][j] == 1

        return all([any(bool_state[i]) for i in range(4)])


def breadth_first_tree_search(problem):
    frontier = deque([Node(problem.initial)])  # FIFO queue
    res_path = []  # initialize the path/expansion list
    while frontier:  # run until we have something in the frontier
        node = frontier.popleft()  # get the next node from the frontier (using the queue FIFO logic)
        res_path.append(node)   # add it to the path since we are planning on expanding it
        if problem.goal_test(node.state):  # check if it is a goal state
            print(res_path)  # print out the result path because we found a goal state
            return node  # return the result node
        frontier.extend(node.expand(problem))  # if we are not in a goal state, expand the current node

    return None


def breadth_first_graph_search(problem):
    frontier = deque([Node(problem.initial)])  # FIFO queue
    res_path = []  # initialize the path/expansion list
    explored = set()  # initialize a set where we will store all already explored nodes (since this is a graph search)
    while frontier:  # run until we have something in the frontier
        node = frontier.popleft()  # get the next node from the frontier (using the queue FIFO logic)
        res_path.append(node)  # add it to the path since we are planning on expanding it
        explored.add(node.state)  # also add the state to the set of already explored nodes
        for child in node.expand(problem):  # start expanding the current node
            if child.state not in explored and child not in frontier:  # check if the child of the node was already explored
                if problem.goal_test(child.state):  # check if it is a goal state
                    print(res_path)  # print out the result path because we found a goal state
                    return child  # return the result
                frontier.append(child)  # append all the nodes that were not already explored to the frontier
    return None


def depth_first_tree_search(problem):
    # same as breadth first tree search, but using a stack for the frontier (a list in Python is suitable)
    frontier = [Node(problem.initial)]  # LIFO Stack
    res_path = []
    while frontier:
        node = frontier.pop()
        res_path.append(node)
        if problem.goal_test(node.state):
            print(res_path)
            return node
        frontier.extend(node.expand(problem))
    return None


def depth_first_graph_search(problem):
    # same as breadth first graph search, but using a stack for the frontier (a list in Python is suitable)
    frontier = [(Node(problem.initial))]  # LIFO Stack
    res_path = []
    explored = set()
    while frontier:
        node = frontier.pop()
        res_path.append(node)
        if problem.goal_test(node.state):
            print(res_path)
            return node
        explored.add(node.state)
        # same for loop as in breadth first graph search just with different syntax
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and child not in frontier)
    return None


def main():
    f_queen = FourQueensProblem()

    my_state = ((0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0))

    print(len(f_queen.actions(my_state)))

    print_matrix(f_queen.result(my_state, "o 4 4"))

    my_goal_state = ((0, 1, 0, 0),
                     (0, 0, 0, 1),
                     (1, 0, 0, 0),
                     (0, 0, 1, 0))
    print()
    print_matrix(my_goal_state)
    print(f_queen.goal_test(my_goal_state))

    print()
    print("Trying to solve with Trial Error method")
    print(Trial_Error(f_queen))

    print()
    print("Solving with breadth first tree search")
    print(breadth_first_tree_search(f_queen))

    print()
    print("Solving with breadth first graph search")
    print(breadth_first_graph_search(f_queen))

    print()
    print("Solving with depth first tree search")
    print(depth_first_tree_search(f_queen))

    print()
    print("Solving with depth first graph search")
    print(depth_first_graph_search(f_queen))


main()
