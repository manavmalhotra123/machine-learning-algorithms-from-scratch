class GradientDescentMLR:
  def __init__(self, learning_rate=0.01, iters=2) -> None:
    self.learning_rate = learning_rate
    self.iters = iters

  def fit(self, X1, X2, Y):
    ones = np.ones(len(X1))
    features = np.c_[ones, X1, X2]
    known_labels = np.array(Y).reshape((len(Y),1))
    weights = np.zeros(features.shape[1])
    temp = weights
    for i in range(self.iters):
      temp[0] = weights[0] - (self.learning_rate/len(X1)) * np.sum((features @ weights) - known_labels.transpose())
      for j in range(1, len(weights)):
        temp[j] = weights[j] - (self.learning_rate/len(X1)) * ((features @ weights) - known_labels.transpose()) @ features.transpose()[j]
      for k in range(len(weights)):
        weights[k] = temp[k]
        print(f"For iteration {i+1}: \nb_{k} = {weights[k]}\n")
        
        
        
 def MLRusingGradientDescent():
  object = GradientDescentMLR()
  X1 = np.array([2.75, 2.5, 2.25, 2, 2, 2, 1.75, 1.75])
  X2 = np.array([5.3, 5.3, 5.5, 5.7, 5.9, 6, 5.9, 6.1])
  y = np.array([1464, 1394, 1159, 1130, 1075, 1047, 965, 719])
  object.fit(X1,X2,y)
