from node import Node
import sys

class Tree(object):
	def __init__(self, target):
		self.target = target
		self.root = None

	def fit(self, predictors, emotion_number, training_data):
	    # Create dataframe which has only 0/1 in target column indicating if training sample belongs to given emotion or not
	    # binary_training_data = training_data.copy()
	    # binary_training_data['emotion'] = binary_training_data['emotion'].apply(lambda emotion: 1 if emotion == emotion_number else 0)
	    root = self.helper(predictors, training_data)
	    self.root = root

	def predict(self, data_point):
	    return self.predict_helper(self.root, data_point)

	def predict_helper(self, node, data_point):
	    if node.c != -1:
	        return node.c
	    if data_point[node.op] == 0:
	        return self.predict_helper(node.kids[0], data_point)
		    
	    return self.predict_helper(node.kids[1], data_point)    
    
	def helper(self, remaining_predictors, remaining_training_data):  
	#     print("Entering helper: data has length {}".format(len(remaining_training_data)))
	    # If the remaining_training_data all belong to same class, then stop
	    if not remaining_predictors or self.all_equal(remaining_training_data['emotion']):
	        # class is the target value of any of the remaining training_examples 
	        # TODO: Fix: should pick most frequent
	#         print("Most Frequent value" + str(remaining_training_data['emotion'].value_counts().idxmax()))
	        return Node(c = remaining_training_data['emotion'].value_counts().idxmax())
	        
	    # Out of all predictors remaining_X_j and values s, we need to find j and s such that entropy is minimized
	    # In this case, since the predictors are binary, we don't need to find s since we only have one choice of branching
	    j, new_remaining_predictors = self.get_best_predictor(remaining_predictors, remaining_training_data)
	    
	    if j == -1:
	        return Node(c = remaining_training_data.iloc[0]['emotion'])
	    
	#     print("Helper: best predictor is {}".format(j))
	    
	    # Now all training examples with remaining_X_j = 0 will belong to the left subtree, and remaining_X_j = 1 to the right subtree
	    left_training_data, right_training_data = self.split_data_based_on_predictor(remaining_training_data, j)
	#     print("Helper: Split data: left has size {}, right has size {}".format(len(left_training_data), len(right_training_data)))
	    
	    left_node = self.helper(new_remaining_predictors, left_training_data)
	    right_node = self.helper(new_remaining_predictors, right_training_data)
	    
	    current_node = Node(kids=[left_node, right_node], op=j)
	    return current_node
	    
	""" 
	Return the predictor among predictors which maximizes the information gain (minimizes the entropy) for data X and y
	"""
	def get_best_predictor(self, predictors, training_data):
	    min_entropy = sys.maxsize
	    best_predictor = -1
	    remaining_predictors = predictors[:]
	    
	    for predictor in predictors:
	        # Split data based on predictor
	        left_training_data, right_training_data = self.split_data_based_on_predictor(training_data, predictor)
	        
	        if left_training_data.empty or right_training_data.empty:
	            remaining_predictors.remove(predictor)
	            continue
	        
	        # Compute entropy for this split
	        left_entropy = self.get_list_cross_entropy(left_training_data[self.target])
	        right_entropy = self.get_list_cross_entropy(right_training_data[self.target])
	        entropy = left_entropy + right_entropy
	        #print("get_best_predictor: predictor {} generates entropy of {}".format(predictor, entropy))
	        if entropy < min_entropy:
	            min_entropy = entropy
	            best_predictor = predictor
	       
	    if best_predictor != -1:
	        remaining_predictors.remove(best_predictor)
	    
	    return best_predictor, remaining_predictors
	        
	def get_list_cross_entropy(self, ys):
	    # An empty list has 0 entropy
	    if ys.empty:
	        return 0
	    
	    proportion_of_ones = sum(ys) / len(ys)
	    proportion_of_zeros = 1 - proportion_of_ones
	    
	    if proportion_of_ones == 0 or proportion_of_zeros == 0:
	        return 0
	    
	    entropy = - proportion_of_zeros * math.log2(proportion_of_zeros) - proportion_of_ones * math.log2(proportion_of_ones)

	    return entropy

	def split_data_based_on_predictor(self, data, j):
	    left_training_data = data[data[j] == 0]
	    right_training_data = data[data[j] == 1]
	    # print("Splitting data: left has size {}, right has size {}".format(len(left_training_data), len(right_training_data)))
	    return left_training_data, right_training_data

	def all_equal(self, series):
	    return len(set(series)) == len(series)