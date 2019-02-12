import pickle
import random
import time

import numpy as np
from sklearn.svm import SVC

import hlt
import parse
from hlt import constants
from hlt import positionals
import pandas as pd
import sys

# f = open("time.txt", "w")
# f.write("lol wtf is my life1")

import os
stderr = sys.stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stderr = open(os.devnull, 'w')
# f.write("lol wtf is my life1.5")
# import tensorflow as tf
# f.write("lol wtf is my life2")
from keras import Sequential
# f.write("lol wtf is my life3")
from keras.layers import Conv1D,MaxPooling2D,Flatten,Dense
# f.write("lol wtf is my life4")
from keras.models import load_model
# f.write("lol wtf is my life5")
# from keras.backend import set_session
# f.write("lol wtf is my life6")
# config = tf.ConfigProto()
# f.write("lol wtf is my life7")
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# f.write("lol wtf is my life8")
# set_session(tf.Session(config=config))
# f.write("lol wtf is my life9")
sys.stderr = stderr

import warnings
warnings.filterwarnings("ignore")
import time

# f =  open("time.txt", "w")

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '99'
# tf.logging.set_verbosity(tf.logging.ERROR)

class HaliteModel:
    MAX_FILES = 100
    DIRECTION_ORDER = [positionals.Direction.West,
                       positionals.Direction.North,
                       positionals.Direction.East,
                       positionals.Direction.South]
    MOVE_TO_DIRECTION = {
        "o": positionals.Direction.Still,
        "w": positionals.Direction.West,
        "n": positionals.Direction.North,
        "e": positionals.Direction.East,
        "s": positionals.Direction.South}
    OUTPUT_TO_MOVE = {
        0: "o",
        1: "w",
        2: "n",
        3: "e",
        4: "s"}
    MOVE_TO_OUTPUT = {v: k for k, v in OUTPUT_TO_MOVE.items()}

    def __init__(self, model_main ):
        self.model = model_main


    # Generate the feature vector
    def input_for_ship(self, game_map, ship, my_other_ships, other_ships, my_dropoffs, other_dropoffs, turn_number,
                       rotation=0):
        result = []

        # game turn
        percent_done = turn_number / constants.MAX_TURNS
        result.append(percent_done)
        #print("ship",ship)
        #print("other_ships",other_ships)
        # Local area stats
       # print(len(result))
        for objs in [my_other_ships, other_ships, my_dropoffs, other_dropoffs]:
            #print('objs',objs)
            objs_directions = []
            for d in self.DIRECTION_ORDER:
                objs_directions.append(int(game_map.normalize(ship.position.directional_offset(d)) in objs))
            #print("objs_directions",objs_directions)
            result += self.rotate_direction_vector(objs_directions, rotation)
           # print("result2",result)
        #print(len(result))
        # directions to highest halite cells at certain distances
        for distance in range(1, 13):
            max_halite_cell = self.max_halite_within_distance(game_map, ship.position, distance)
            halite_directions = self.generate_direction_vector(game_map, ship.position, max_halite_cell)
            result += self.rotate_direction_vector(halite_directions, rotation)
        #print(len(result))
        # directions to closest drop off
        closest_dropoff = my_dropoffs[0]
        for dropoff in my_dropoffs:
            if game_map.calculate_distance(ship.position, dropoff) < game_map.calculate_distance(ship.position,
                                                                                                 closest_dropoff):
                closest_dropoff = dropoff
        dropoff_directions = self.generate_direction_vector(game_map, ship.position, closest_dropoff)
        result += self.rotate_direction_vector(dropoff_directions, rotation)
        #print(len(result))
        # local area halite
        local_halite = []
        for d in self.DIRECTION_ORDER:
            local_halite.append(game_map[game_map.normalize(ship.position.directional_offset(d))].halite_amount / 1000)
        result += self.rotate_direction_vector(local_halite, rotation)
        #print(local_halite)
        # current cell halite indicators
        for i in range(0, 200, 50):
            #print(i)
            result.append(int(game_map[ship.position].halite_amount <= i))
            #print(len(result))
        result.append(game_map[ship.position].halite_amount / 1000)
        
        #sys.exit()
        return result

    def predict_move(self, ship, game_map, me, other_players, turn_number):
        other_ships = [s.position for s in me.get_ships() if s.id != ship.id]
        opp_ships = [s.position for p in other_players for s in p.get_ships()]
        my_dropoffs = [d.position for d in list(me.get_dropoffs()) + [me.shipyard]]
        opp_dropoffs = [d.position for p in other_players for d in p.get_dropoffs()] + \
                       [p.shipyard.position for p in other_players]
        data = np.array(self.input_for_ship(game_map,
                                            ship,
                                            other_ships,
                                            opp_ships,
                                            my_dropoffs,
                                            opp_dropoffs,
                                            turn_number))
        data = data.reshape(1, -1)
        # data = data.reshape(1, 78, 1)
        
        model_output = np.argmax(self.model.predict(data))
        # f.write("-----------" + str(model_output) + " " +str(self.MOVE_TO_DIRECTION[self.OUTPUT_TO_MOVE[model_output]]) +" \n")
        return self.MOVE_TO_DIRECTION[self.OUTPUT_TO_MOVE[model_output]]

    def save(self, file_name=None):
        if file_name is None:
            file_name = "model_weights_%f.svc" % time.time()
        with open(file_name, 'wb') as f:
            pickle.dump(self.model, f)

    # finds cell with max halite within certain distance of location
    def max_halite_within_distance(self, game_map, location, distance):
        max_halite_cell = location
        max_halite = 0
        for dx in range(-distance, distance + 1):
            for dy in range(-distance, distance + 1):
                loc = game_map.normalize(location + hlt.Position(dx, dy))
                if game_map.calculate_distance(location, loc) > distance:
                    continue

                # pick cell with max halite
                cell_halite = game_map[loc].halite_amount
                if cell_halite > max_halite:
                    max_halite_cell = loc
                    max_halite = cell_halite
        return max_halite_cell

    # generate vector that tells which directions to go to get from ship_location to target
    def generate_direction_vector(self, game_map, ship_location, target):
        directions = []
        for d in self.DIRECTION_ORDER:
            directions.append(
                int(game_map.calculate_distance(game_map.normalize(ship_location.directional_offset(d)), target) <
                    game_map.calculate_distance(ship_location, target)))
        return directions

    def rotate_direction_vector(self, direction_vector, rotations):
        for i in range(rotations):
            direction_vector = [direction_vector[-1]] + direction_vector[:-1]
        return direction_vector



