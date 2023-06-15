import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import matplotlib.colors as colors

import pickle 
import os
from sklearn.feature_selection import SelectKBest, mutual_info_regression

class SelfOrganizingMap:
    def __init__(self, width, height, input_dimensions,  data=None):
        '''
        Initialize a Self-Organizing Map (SOM) object
        
        Parameters:
            width (int): width of the SOM grid
            height (int): height of the SOM grid
            data (pandas.DataFrame, optional): data to train the SOM on, defaults to None
        '''
        self.width = width
        self.height = height
        self.input_dimensions = input_dimensions
        self.data = data

        self.initial_som = None
        self.online_som = None
        self.layer_one = None
        self.layer_two = None
        self.two_layer_som = None
    # TODO: set parameters

    def train_initial_som(self, patterns, parameters=None):
        """
        function to train the initial self-organizing map on the given data

        Parameters:
            data (pandas.DataFrame): data to be used for training the self-organizing map
            parameters (dict): dictionary of parameters to be used for training the self-organizing map
                - 'map_width' (int): width of the map in number of nodes
                - 'map_height' (int): height of the map in number of nodes
                - 'training_epochs' (int): number of training epochs to be run

        Returns:
            som (SelfOrganizingMap): trained self-organizing map
        """

        # Initialize the weights of the self-organizing map
        if(not self.initial_som):
            self.initialize_weights(parameters)
        self.layer_one = np.copy(self.initial_som)

        MAP = np.copy(self.initial_som)
        prev_MAP = np.zeros((self.height, self.width, self.input_dimensions))

        radius0 = max(self.height, self.width) / 2
        learning_rate0 = parameters['learning_rate0']

        coordinate_map = np.zeros([self.height, self.width, 2], dtype=np.int32)

        for i in range(0, self.height):
            for j in range(0, self.width):
                coordinate_map[i][j] = [i, j]

        epochs = parameters['epochs']
        radius = radius0
        learning_rate = learning_rate0
        max_iterations = len(patterns) + 1
        too_many_iterations = 10*max_iterations

        convergence = [1]

        timestep = 1
        e = 0.001
        flag = 0

        epoch = 0
        while epoch < epochs:
            shuffle = np.random.randint(len(patterns), size=len(patterns))

            for i in range(len(patterns)):
                J = np.linalg.norm(MAP - prev_MAP)
                # J = || euclidean distance between previous MAP and current MAP  ||

                if J < e:
                    flag = 1
                    break

                else:
                    pattern = patterns[shuffle[i]]
                    pattern_arr = np.tile(pattern, (self.height, self.width, 1))
                    Eucli_MAP = np.linalg.norm(pattern_arr - MAP, axis=2)

                    BMU = np.unravel_index(np.argmin(Eucli_MAP, axis=None), Eucli_MAP.shape)

                    prev_MAP = np.copy(MAP)

                    for j in range(self.height):
                        for k in range(self.width):
                            distance = np.linalg.norm([j - BMU[0], k - BMU[1]])
                            if distance <= radius:
                                # self.print_som()
                                MAP[j][k] = MAP[j][k] + learning_rate * (pattern - MAP[j][k])
                    
                    # learning_rate = learning_rate0 * np.exp(-timestep / (max_iterations / np.log(learning_rate0)))  # learning rate decay
                    learning_rate = learning_rate0 * (1 -(epoch/epochs))
                    # radius = radius0 * np.exp(-timestep / (max_iterations / np.log(radius0)))  # radius decay
                    radius = radius0 * math.exp(-epoch/epochs)
                    timestep += 1

            if J < min(convergence):
                # print('Lower error found: %s' %str(J) + ' at epoch: %s' % str(epoch))
                # print('\tLearning rate: ' + str(learning_rate))
                # print('\tNeighbourhood radius: ' + str(radius))
                self.layer_one = np.copy(MAP)
            convergence.append(J)

            if flag == 1:
                break

            epoch += 1

        return convergence

    def initialize_weights(self, parameters=None):
        # function to initialize the weights of the self-organizing map
        if parameters and parameters['load_from_file']:
            initial_som = self.load_saved_model(parameters['load_from_file'])
            if(initial_som.map_width != parameters['map_width'] or initial_som.map_height != parameters['map_height']):
                raise Exception('The map size does not match the size of the loaded model')
        elif parameters:
            self.initial_som = np.random.uniform(size=(parameters['map_height'], parameters['map_width'], parameters['input_dimensions']))
        else:
            return False
        
        return True
    
    def print_som(self):
        '''
        function to print the SOM
        '''
        print("Initial som\n", self.initial_som)
        print("\nSOM layer 1\n", self.layer_one)
        print("\nSOM layer 2\n", self.layer_two)
        print("\ninitial_som\n", self.two_layer_som)

    def init_predictor(self, patterns, classes, layer):
        '''
        function to initialize the predictor
        '''
        if layer == 1:
            MAP = np.copy(self.layer_one)
        elif layer == 2:
            MAP = np.copy(self.layer_two)
        elif layer == 3:
            MAP = np.copy(self.two_layer_som)
        else:
            print("Invalid layer number")
        
        MAP_class = np.zeros([self.height, self.width], dtype=np.int32)
        
        # for each neuron, find the class of the pattern closest to it
        for i in range(self.height):
            for j in range(self.width):
                neuron = MAP[i][j]
                closest_pattern_idx = None
                closest_pattern_distance = float('inf')
                for k in range(len(patterns)):
                    pattern = patterns[k]
                    distance = np.linalg.norm(pattern - neuron)
                    if distance < closest_pattern_distance:
                        closest_pattern_distance = distance
                        closest_pattern_idx = k
                MAP_class[i][j] = classes[closest_pattern_idx]

        return MAP_class

    def predict(self, predictor, X_test, layer):
        '''
        function to predict the class labels for the provided data
        '''
        if layer == 1:
            MAP = np.copy(self.layer_one)
        elif layer == 2:
            MAP = np.copy(self.layer_two)
        elif layer == 3:
            MAP = np.copy(self.two_layer_som)
        else:
            print("Invalid layer number")
            return

        pred = np.zeros([len(X_test)], dtype=np.int32)
        for i in range(len(X_test)):
            pattern = X_test[i]
            closest_neuron_idx = None
            closest_neuron_distance = float('inf')
            for x in range(self.height):
                for y in range(self.width):
                    neuron = MAP[x][y]
                    distance = np.linalg.norm(pattern - neuron)
                    if distance < closest_neuron_distance:
                        closest_neuron_distance = distance
                        closest_neuron_idx = (x, y)
            pred[i] = predictor[closest_neuron_idx[0]][closest_neuron_idx[1]]

        return pred

    def visualize_som(self, layer, patterns, classes, palette, parameters=None):
        dpi = 10
        BMU = np.zeros([2], dtype=np.int32)
        result_map = np.zeros([self.height, self.width, 3], dtype=np.float32)

        i = 0

        if layer == 1:
            MAP = np.copy(self.layer_one)
        elif layer == 2:
            MAP = np.copy(self.layer_two)
        elif layer == 3:
            MAP = np.copy(self.two_layer_som)
        else:
            print("Invalid layer number")
            return

        count_arr = np.zeros([self.height, self.width], dtype=np.int32)
        for pattern in patterns:
            pattern_arr = np.tile(pattern, (self.height, self.width, 1))
            Eucli_MAP = np.linalg.norm(pattern_arr - MAP, axis=2)

            BMU = np.unravel_index(np.argmin(Eucli_MAP, axis=None), Eucli_MAP.shape)

            x = BMU[0]
            y = BMU[1]

            result_map[x][y] += palette[int(classes[i])]
            count_arr[x][y] += 1
            i += 1

        for i in range(self.height):
            for j in range(self.width):
                if count_arr[i][j] != 0:
                    result_map[i][j] /= count_arr[i][j]

        result_map = result_map / 255.0
        result_map = np.flip(result_map, 0)

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(self.width*dpi, self.height*dpi), dpi=dpi)

        # Calculate the required font size based on the figure width
        title_length = len(parameters['title'])
        fontsize = int(self.width * dpi * 0.5 / title_length)

        # Display the array as an image with the color map
        # ax.set_title(parameters['title'])
        ax.set_title(parameters['title'], fontsize=fontsize*dpi*dpi)
        ax.imshow(result_map)
        ax.axis('on')

        # Save the plot and close the figure
        plt.savefig(parameters['savepath'], dpi=dpi)
        plt.close(fig)

        return result_map



    
    def train_Hikawa_SOM(self, patterns, alpha0, sigma0, epochs):
        '''
        function to train the Hikawa SOM

        Parameters:
            patterns (pandas.DataFrame): data to be used for training the self-organizing map
            alpha0 (float): initial learning rate
            sigma0 (float): initial neighbourhood radius
            epochs (int): number of epochs to train the self-organizing map for
        '''
        MAP = np.random.uniform(size=(self.height, self.width, self.input_dimensions))
        prev_MAP = np.zeros((self.height, self.width, self.input_dimensions))

        dc_bar = 0
        for epoch in range(epochs):
            shuffle = np.random.randint(len(patterns), size=len(patterns))
            for i in range(len(patterns)):
                pattern = patterns[shuffle[i]]

                # Get the best matching unit (BMU)
                pattern_ary = np.tile(pattern, (self.height, self.width, 1))
                Eucli_MAP = np.linalg.norm(pattern_ary - MAP, axis=2)
                BMU = np.unravel_index(np.argmin(Eucli_MAP, axis=None), Eucli_MAP.shape)

                dc = np.linalg.norm(MAP[BMU[0], BMU[1]] - pattern)
                dc_bar = 0.7 * dc_bar + 0.3 * dc.mean()
                alpha = alpha0 * dc / dc_bar
                sigma = sigma0 * dc_bar

                prev_MAP = np.copy(MAP)
                for i in range(self.height):
                    for j in range(self.width):
                        distance = np.linalg.norm([i - BMU[0], j - BMU[1]])
                        theta = 1 if distance <= sigma else 0
                        MAP[i][j] = MAP[i][j] + alpha * theta * (pattern - MAP[i][j])

        self.online_som = np.copy(MAP)

    def train_layer_two(self, new_data, alpha, sigma, dc_bar, k, epochs=1):
        '''
        function to train the second layer of the SOM

        Parameters:
            new_data (numpy.ndarray): data to be used for training the second layer of the SOM
            alpha (float): learning rate
            sigma (float): neighborhood radius
            dc_bar (float): average distance between the BMU and the input pattern
            k (float): constant used to calculate dc_bar
            epochs (int): number of epochs to train the SOM for

        Returns:
            None
        '''
        MAP = np.copy(self.layer_one)
        # MAP2 = np.copy(MAP)
        # create a data frame to store alpha, sigma and dc_bar for each epoch
        df = pd.DataFrame(columns=['alpha', 'sigma', 'dc_bar'])
        # print("hi")
        for epoch in range(epochs):
            for pattern in new_data:
                pattern_ary = np.tile(pattern, (self.height, self.width, 1))
                Eucli_MAP = np.linalg.norm(pattern_ary - MAP, axis=2)
                BMU = np.unravel_index(np.argmin(Eucli_MAP, axis=None), Eucli_MAP.shape)
                dc = np.linalg.norm(MAP[BMU[0], BMU[1]] - pattern)
                dc_bar = (1 - k) * dc_bar + k * dc
                alpha = alpha * dc / dc_bar
                sigma = sigma * dc_bar
                df.loc[len(df)] = [alpha, sigma, dc_bar]

                for i in range(self.height):
                    for j in range(self.width):
                        distance = np.linalg.norm([i - BMU[0], j - BMU[1]])
                        theta = 1 if distance <= sigma else 0
                        # MAP2[i][j] = MAP[i][j] + alpha*theta*(pattern-MAP[i][j])
                        MAP[i][j] = MAP[i][j] + alpha*theta*(pattern-MAP[i][j])
            # MAP = np.copy(MAP2)
        # save the data frame to a csv file
        df.to_csv('alpha_sigma_dcbar.csv', index=False)
        self.layer_two = np.zeros((self.height, self.width, self.input_dimensions))
        self.layer_two = np.copy(MAP)

       
    def error_calculation(self, layer, patterns):
        '''
        function to calculate the quantization and topographic error

        Parameters:
            layer (int): layer to calculate the error for
            patterns (numpy.ndarray): patterns to calculate the error
        Returns:
            quantization_error (float): quantization error
            topographic_error (float): topographic error
        '''
        if layer == 1:
            MAP_final = np.copy(self.layer_one)
        elif layer == 2:
            MAP_final = np.copy(self.layer_two)
        elif layer == 3:
            MAP_final = np.copy(self.two_layer_som)
        else:
            print("Invalid layer number")
            return

        q_error = quantization_error(patterns, MAP_final)
        t_error = topographic_error(patterns, MAP_final, (self.width, self.height))

        return q_error, t_error

    
    def get_alpha_sigma_dc_bar(self, patterns):
        '''
        function to return the values of alpha, sigma, and dc_bar for a trained SOM

        Parameters:
            MAP (numpy.ndarray): trained SOM
            patterns (numpy.ndarray): patterns used to train the SOM

        Returns:
            alpha (float): alpha value      : Learning rate
            sigma (float): sigma value      : Neighborhood radius
            dc_bar (float): dc_bar value    : Average distance between the BMU and the input pattern
        '''
        MAP = np.copy(self.layer_one)
        map_height, map_width, _ = MAP.shape
        dc_bar = 0
        sum_dc = 0
        count = 0
        for pattern in patterns:
            pattern_ary = np.tile(pattern, (map_height, map_width, 1))
            Eucli_MAP = np.linalg.norm(pattern_ary - MAP, axis=2)
            BMU = np.unravel_index(np.argmin(Eucli_MAP, axis=None), Eucli_MAP.shape)
            dc = np.linalg.norm(MAP[BMU[0], BMU[1]] - pattern)
            dc_bar = 0.99 * dc_bar + 0.01 * dc
            sum_dc += dc
            count += 1
        dc_bar = sum_dc / count
        alpha = 0.1 * dc / dc_bar
        sigma = 0.1 * dc_bar
        return alpha, sigma, dc_bar

    def get_change_alpha_sigma_dc_bar(self, patterns):
        '''
        function to returns the changes to values of alpha, sigma, and dc_bar for a trained SOM

        Parameters:
            MAP (numpy.ndarray): trained SOM
            patterns (numpy.ndarray): patterns used to train the SOM

        Returns:
            alphas (float): list of alpha values
            sigmas (float): list of sigma values
            dc_bars (float): list of dc_bar values
        '''
        MAP = np.copy(self.layer_one)
        map_height, map_width, _ = MAP.shape
        dc_bar = 0
        sum_dc = 0
        count = 0
        alphas = []
        sigmas = []
        dc_bars = []
        for pattern in patterns:
            pattern_ary = np.tile(pattern, (map_height, map_width, 1))
            Eucli_MAP = np.linalg.norm(pattern_ary - MAP, axis=2)
            BMU = np.unravel_index(np.argmin(Eucli_MAP, axis=None), Eucli_MAP.shape)
            dc = np.linalg.norm(MAP[BMU[0], BMU[1]] - pattern)
            dc_bar = 0.99 * dc_bar + 0.01 * dc
            sum_dc += dc
            count += 1
            dc_bar = sum_dc / count
            alpha = 0.1 * dc / dc_bar
            sigma = 0.1 * dc_bar
            alphas.append(alpha)
            sigmas.append(sigma)
            dc_bars.append(dc_bar)
        return alphas, sigmas, dc_bars

    def __del__(self):
        '''
        destructor to delete the SOM object
        '''
        # print('SOM object deleted')
        return
################################################################
####################### END OF SOM CLASS #######################
################################################################

def load_data(filepath, header=None):
    '''
    function to load the data from the given filepath
    
    Parameters:
        filepath (str): path to the data file

    Returns:
        data (pandas.DataFrame): data loaded from the given filepath
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, '..', '..', filepath)
    data = pd.read_csv(filepath, header=header)
    return data

#     def feature_selection(data, num_features, method='mutual_info'):
#         '''
#         Function to perform feature selection on the data
        
#         Parameters:
#             data (pandas.DataFrame): input data
#             num_features (int): number of features to select
#             method (str): method to perform feature selection (default: 'mutual_info')
            
#         Returns:
#             data (pandas.DataFrame): data after performing feature selection
#         '''
#         if method == 'mutual_info':
#             selector = SelectKBest(score_func=mutual_info_regression, k=num_features)
#             data = selector.fit_transform(data, y)
#         return data


def preprocess_data(data, parameters=None, selected_features=None):
    """
    function to preprocess the given data

    Parameters:
        data (pandas.DataFrame): data to preprocess
        parameters (dict, optional): preprocessing parameters. Defaults to None.

    Returns:
        processed_data (pandas.DataFrame): preprocessed data
    """
        # remove any missing values
    data.dropna(inplace=True)

    # select specified features, if provided
    if selected_features:
        data = data[selected_features]

    # Convert categorical variables into numeric variables
    if parameters and 'categorical_vars' in parameters:
        categorical_vars = parameters['categorical_vars']
        for var in categorical_vars:
            data[var] = pd.Categorical(data[var])
            data[var] = data[var].cat.codes
    
    # Normalize the data
    if parameters and 'normalize' in parameters:
        if parameters['normalize']:
            data = (data - data.min()) / (data.max() - data.min())

    # Convert the data into a numpy array
    processed_data = data.to_numpy()
    return processed_data

def get_patterns_and_classes(data, parameters):
    '''
    function to get the patterns and classes from the given data

    Parameters:
        data (pandas.DataFrame): data to get the patterns and classes from
        parameters (dict): dictionary of parameters to get the patterns and classes
            - 'patterns' (list): list of patterns to get from the data
            - 'classes' (list): list of classes to get from the data

    Returns:
        patterns (numpy.ndarray): patterns from the given data
        classes (numpy.ndarray): classes from the given data
    '''
    X = data.iloc[:, parameters['patterns']].values

    classes = data.iloc[:, parameters['classes']].values
    patterns = np.asarray(X, dtype=np.float32)

    return patterns, classes

def plot_convergence(convergence, parameters):
    '''
    function to plot the convergence of the self-organizing map

    Parameters:
        convergence (list): list of convergence values
        parameters (dict): dictionary of parameters to plot the convergence
            - 'save_plot' (bool): whether to save the plot or not
            - 'title' (str): title of the plot
            - 'xlabel' (str): label of the x-axis
            - 'ylabel' (str): label of the y-axis
            - 'filepath' (str): path to save the plot
    '''
    # assign minimum value of convergence to epsilon
    epsilon = min(convergence)    
    convergence = [value + epsilon for value in convergence]

    plt.plot(convergence)
    plt.title(parameters['title'])
    plt.xlabel(parameters['xlabel'])
    plt.ylabel(parameters['ylabel'])
    plt.yscale('log')
    plt.grid(True)
    if parameters['save_plot']:
        plt.savefig(parameters['filepath'])
        plt.close()
    # plt.show()

def generate_palette(num_colors):
    '''
    function to generate a palette of colors

    Parameters:
        num_colors (int): number of colors to generate

    Returns:
        palette (list): list of colors
    '''
    palette = sns.color_palette("bright", num_colors)

    color_arr = np.array(palette.as_hex())
    color_arr_rgb = [0]*num_colors

    for i in range(num_colors):
        color_arr_rgb[i] = tuple(int(color_arr[i][1:][j:j+2], 16) for j in (0, 2, 4))

    return color_arr_rgb

def quantization_error(data, weights):
    '''
    function to calculate the quantization error of the self-organizing map

    Parameters:
        data (numpy.ndarray): data to calculate the quantization error on
        weights (numpy.ndarray): weights of the self-organizing map

    Returns:
        quantization_error (float): quantization error of the self-organizing map
    '''
    N = data.shape[0]
    quantization_error = 0
    
    weights_reshaped = weights.reshape(weights.shape[0] * weights.shape[1], -1)

    for pattern in data:
        diff = np.linalg.norm(pattern - weights_reshaped, axis=1)
        min_diff = np.min(diff)
        quantization_error += min_diff

    quantization_error /= N
    return quantization_error

def topology_distance(x1, y1, x2, y2):
    '''
    function to calculate the topology distance between two nodes in the self-organizing map

    Parameters:
        x1 (int): x-coordinate of the first node
        y1 (int): y-coordinate of the first node
        x2 (int): x-coordinate of the second node
        y2 (int): y-coordinate of the second node

    Returns:
        topology_distance (float): topology distance between the two nodes
    '''
    topology_distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return topology_distance

def topographic_error(data, weights, topology):
    '''
    function to calculate the topographic error of the self-organizing map

    Parameters:
        data (numpy.ndarray): data to calculate the topographic error on
        weights (numpy.ndarray): weights of the self-organizing map
        topology (tuple): topology of the self-organizing map

    Returns:
        topographic_error (float): topographic error of the self-organizing map
    '''
    N = data.shape[0]
    topographic_error = 0

    weights_reshaped = weights.reshape(weights.shape[0] * weights.shape[1], -1)

    for pattern in data:
        diff = np.linalg.norm(pattern - weights_reshaped, axis=1)
        min_diff = np.min(diff)
        min_index = np.argmin(diff)
        min_index_x = min_index // topology[0]
        min_index_y = min_index % topology[0]
        min_index_topology = (min_index_x, min_index_y)
        topology_diff = topology_distance(min_index_topology[0], min_index_topology[1], topology[0] // 2, topology[1] // 2)
        if topology_diff > 1:
            topographic_error += 1

    topographic_error /= N
    return topographic_error

# def merge_soms(initial_som, online_som):
#     # function to merge the two self-organizing maps into a two-layer self-organizing map
#     pass

# def predict_class(data, som):
#     # function to predict the class of new data points based on the self-organizing map
#     pass

# def evaluate_model(data, som, metrics):
#     # function to evaluate the performance of the self-organizing map on the given data
#     pass

# def update_weights(data):
#     # function to update the weights of the self-organizing map
#     return

# def distance(data_point, weight_vector):
#     # function to calculate the distance between a data point and a weight vector
#     return

# def map_coordinate(data_point):
#     # function to map a data point to the coordinate of the winning node in the self-organizing map
#     return

# def get_neighborhood(coordinate, radius):
#     # function to return the nodes in the self-organizing map within a specified radius of a given coordinate
#     return

# def neighborhood_update(coordinate, data_point, learning_rate, radius):
#     # function to update the weights of the nodes in the self-organizing map within a specified radius of a given coordinate
#     return



# def calculate_error(data, prediction):
#     # function to calculate the error between the actual data and the predicted data
#     return

# def visualize_som(som):
#     # function to visualize the self-organizing map
#     pass

# def partition_data(data, partition_ratio):
#     # function to partition the data into training and test sets
#     return

# def save_model(som, filepath):
#     '''
#     function to save a trained SOM model to a file.

#     Parameters:
#         som (SelfOrganizingMap): trained SOM model
#         filepath (str): path to save the model file
    
#     Returns:
#         None
#     '''

#     with open(filepath, 'wb') as file:
#         pickle.dump(som, file)

