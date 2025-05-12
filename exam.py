from re import VERBOSE
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from collections import deque
from search import Problem, Trial_Error, Node
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

########################## NN #########################

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

X = dataset[:,0:8]
y = dataset[:,8]

model = Sequential()
model.add(Dense(15,input_dim = 8, activation="relu"))
model.add(Dense(8,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(1, 'sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X,y,epochs=150,batch_size=20)

loss, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
print('loss: %.2f' % (loss))


###############          CNN           #############################
num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train) , (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

for i in range(9):

  plt.subplot(330 + 1 + i)

  plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
plt.show()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
    keras.Input(shape = input_shape),
    layers.Conv2D(32,kernel_size=(3,3), activation = 'relu'), #convolutional layer
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(64,kernel_size=(3,3), activation = 'relu'), #convolutional layer
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
    ]
)

model.summary()

batch_size = 128
epochs = 1

model.compile(loss = "categorical_crossentropy", optimizer = "adam" , metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 1, validation_split = 0.1)
score = model.evaluate(x_test, y_test, verbose = 0)

print("Test loss:", score[0])
print("Test accuracy:", score[1])



####################Search##################################


def breadth_first_tree_search(problem):
    frontier = deque([Node(problem.initial)])
    res_path = []
    while frontier:
        node = frontier.popleft()
        res_path.append(node)
        if problem.goal_test(node.state):
            print(res_path)
            return node
        frontier.extend(node.expand(problem))

    return None


def breadth_first_graph_search(problem):
    frontier = deque([Node(problem.initial)])
    res_path = []
    explored = set()
    while frontier:
        node = frontier.popleft()
        res_path.append(node)
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    print(res_path)
                    return child
                frontier.append(child)
    return None


def depth_first_tree_search(problem):
    frontier = [Node(problem.initial)]
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
    frontier = [(Node(problem.initial))]
    res_path = []
    explored = set()
    while frontier:
        node = frontier.pop()
        res_path.append(node)
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    print(res_path)
                    return child
                frontier.append(child)
    return None




################Q-Learning######################


#work starts from here
class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate):
        self.n_state = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate

        #q-value table
        self.q_table = np.zeros((n_states, n_actions))

    def act(self, state, epsilon):
        random_int = random.uniform(0, 1)

        if random_int > epsilon:
            action = np.argmax(self.q_table[state])  # take the table with state, not the table itself
        else:
            action = random.randint(0, self.n_actions - 1)

        return action

    def learn(self, state, action, reward, new_state, gamma):
        old_value = self.q_table[state][action]
        new_estimate = reward + gamma * max(self.q_table[new_state])

        self.q_table[state][action] = old_value + self.learning_rate * (new_estimate - old_value)
