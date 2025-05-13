from search import Problem, Trial_Error


class Cup3(Problem):
    def __init__(self):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        super().__init__((5, 0, 0), [(4, 1, 0), (4, 0, 1)])

        self.H1 = [0, 1, 2, 3, 4, 5]
        self.H2 = [0, 1, 2, 3]
        self.H3 = [0, 1, 2]

        self.H = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3], [0, 1, 2]]

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        acts = []
        cup1, cup2, cup3 = state

        if cup1 > 0 and cup2 < max(self.H2):
            acts.append("o 1 2")
        if cup1 > 0 and cup3 < max(self.H3):
            acts.append("o 1 3")
        if cup2 > 0 and cup1 < max(self.H1):
            acts.append("o 2 1")
        if cup2 > 0 and cup3 < max(self.H3):
            acts.append("o 2 3")
        if cup3 > 0 and cup1 < max(self.H1):
            acts.append("o 3 1")
        if cup3 > 0 and cup2 < max(self.H2):
            acts.append("o 3 2")
        return acts

    def actions2(self, state):
        acts = []

        for i in range(1, 4):
            for j in range(1, 4):
                if i != j:
                    if state[i-1] > 0 and state[j-1] < max(self.H[j-1]):
                        acts.append("o {} {}".format(i, j))

        return acts

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        cup1, cup2, cup3 = state
        if action == "o 1 2":
            m = min(cup1, max(self.H2)-cup2)
            return cup1 - m, cup2 + m, cup3
        if action == "o 1 3":
            m = min(cup1, max(self.H3)-cup3)
            return cup1 - m, cup2, cup3 + m
        if action == "o 2 1":
            m = min(cup2, max(self.H1)-cup1)
            return cup1 + m, cup2 - m, cup3
        if action == "o 2 3":
            m = min(cup2, max(self.H3)-cup3)
            return cup1, cup2 - m, cup3 + m
        if action == "o 3 1":
            m = min(cup3, max(self.H1)-cup1)
            return cup1 + m, cup2, cup3 - m
        if action == "o 3 2":
            m = min(cup3, max(self.H2)-cup2)
            return cup1, cup2 + m, cup3 - m

    def result2(self, state, action):
        from_cup = int(action.split(' ')[1])
        to_cup = int(action.split(' ')[2])
        v = min(state[from_cup - 1], max(self.H[to_cup - 1]) - state[to_cup - 1])

        new_state = list(state)

        new_state[from_cup - 1] = state[from_cup - 1] - v
        new_state[to_cup - 1] = state[to_cup - 1] + v

        return tuple(new_state)


def main():
    c = Cup3()
    print(c.actions((5, 0, 0)))
    print(c.actions2((5, 0, 0)))

    # print(c.result((5, 0, 0), "o 1 2"))
    # print(c.result2((5, 0, 0), "o 1 2"))

    # print(c.result((1, 3, 1), "o 2 1"))
    # print(c.result2((1, 3, 1), "o 2 1"))

    # print(Trial_Error(c))


main()
