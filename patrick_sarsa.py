class LinearExpectedSarsaFunction():
    def __init__(self, n_features, action_space=4, weights=None, default=0.0, lr=0.001, gamma=0.99, epsilon=0.91):
        # In this case, features represents the states the agent see
        # n_features should be the number of states that the agent sees
        # action_space should be the number of actions the agent can take in the state

        self.n_features = n_features
        self.action_space = action_space
        self.n_actions = action_space  ##I will make sure that n_actions and action_space is different
        self.lr = lr  # learning rate
        self.gamma = gamma
        self.epsilon = epsilon

        if weights == None:
            self.weights = np.array(
                [
                    [default] * n_features
                    for _ in range(0, self.action_space)
                ]
            )

    def update(self, curent_stacked_feature, action, next_stacked_features, reward, done):
        # update the weights
        q = np.dot(self.weights[action], curent_stacked_feature)

        # Starting here for expected value of sarsa
        q_nexts = np.dot(self.weights, next_stacked_features)
        max_q = np.max(q_nexts)
        n_max_q = 0

        for q_next in q_nexts:  # determining how many max q_values
            if q_next == max_q:
                n_max_q += 1

        # probability distribution
        non_greedy_action_prob = self.epsilon / self.action_space
        greedy_action_prob = (1 - self.epsilon) / n_max_q + non_greedy_action_prob

        expected_q = 0
        sum_prop = 0
        for i in range(self.n_actions):
            if (q_nexts[i] == max_q):
                expected_q += greedy_action_prob * q_nexts[i]
                sum_prop += greedy_action_prob
            else:
                expected_q += non_greedy_action_prob * q_nexts[i]
                sum_prop += non_greedy_action_prob

        # TD based on expected sarsa
        td_error = reward + self.gamma * expected_q * (1 - done) - q
        self.weights[action] += self.lr * td_error * curent_stacked_feature[0]

        # annealing epsilon
        if self.epsilon > 0.09:
            self.epsilon *= 0.999999

    def take_action(self, curent_stacked_feature):
        if (np.random.random() < self.epsilon):
            return np.random.randint(self.action_space)
        else:
            q = np.dot(self.weights, curent_stacked_feature)
            return np.argmax(q)