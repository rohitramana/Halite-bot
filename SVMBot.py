#!/usr/bin/env python3

import os
from collections import defaultdict

import hlt
import model
from hlt import constants
from hlt import positionals

import time

import logging

import random


import os
import sys
stderr = sys.stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stderr = open(os.devnull, 'w')
from keras.models import load_model
sys.stderr = stderr
MODEL = load_model("Training models/halite_bot_teccles_diamond.h5")

class SVMBot:
    def __init__(self):
    # Get the initial game state
        game = hlt.Game()
        game.ready("NNBot")


        # During init phase: initialize the model and compile it
        my_model = model.HaliteModel(MODEL)


        self.my_model = my_model
        self.game = game

        

    def run(self):
        # Some minimal state to say when to go home
        step = 0
        div = 5
        limit = 5000
        
        go_home = defaultdict(lambda: False)
        f = open("textfile.txt", "w")
        x = positionals.Direction.get_all_cardinals()
        while True:
            self.game.update_frame()
            me = self.game.me  # Here we extract our player metadata from the game state
            game_map = self.game.game_map  # And here we extract the map metadata
            other_players = [p for pid, p in self.game.players.items() if pid != self.game.my_id]

            command_queue = []

            if game_map.width >= 40:
                converge_point = 450
            else:
                converge_point = 385

            if self.game.turn_number > converge_point:
                for ship in me.get_ships():
                    if game_map.calculate_distance(ship.position,me.shipyard.position) == 1:
                        movement = game_map.get_unsafe_moves(ship.position, me.shipyard.position)[0]
                    else:
                        movement = game_map.get_safe_move(game_map[ship.position], game_map[me.shipyard.position])
                    if movement is not None:
                        game_map[ship.position.directional_offset(movement)].mark_unsafe(ship)
                        # f.write(str(movement) +" -- " + str(ship.move(movement)) + "\n")
                        command_queue.append(ship.move(movement))
                    else:
                        ship.stay_still()
                self.game.end_turn(command_queue)
                continue

            for ship in me.get_ships():  # For each of our ships
                # Did not machine learn going back to base. Manually tell ships to return home
                if ship.position == me.shipyard.position:
                    go_home[ship.id] = False
                    x = x[1:] + x[:1]
                    for di in x:
                        if not game_map[ship.position.directional_offset(di)].is_occupied:
                            # f.write(str(ship.move(game_map.get_safe_move(game_map[ship.position], game_map[di]))) +" -- "+ str( game_map[di]) + " -- " +  str(game_map.get_safe_move(game_map[ship.position], game_map[di])) + " -- " + str(type(game_map[ship.position])) + "\n")
                            game_map[ship.position.directional_offset(di)].mark_unsafe(ship)
                            command_queue.append(ship.move(di))
                            break
                    continue
                elif go_home[ship.id] or ship.halite_amount >= constants.MAX_HALITE / 2 + 50:
                    go_home[ship.id] = True
                    movement = game_map.get_safe_move(game_map[ship.position], game_map[me.shipyard.position])
                    if movement is not None:
                        game_map[ship.position.directional_offset(movement)].mark_unsafe(ship)
                        # f.write(str(movement) +" -- " + str(ship.move(movement)) + "\n") 
                        command_queue.append(ship.move(movement))
                    else:
                        ship.stay_still()
                    continue

                if (ship.halite_amount < (1 / 10) * game_map[ship.position].halite_amount or game_map[ship.position].halite_amount > 100) and ship.position != me.shipyard.position:
                    ml_move = (0,0)
                else:
                    # Use machine learning to get a move
                    ml_move = self.my_model.predict_move(ship, game_map, me, other_players, self.game.turn_number)

                    
                if ml_move is not None:

                    if ship.halite_amount == 0 and game_map[ship.position.directional_offset(ml_move)] == game_map[me.shipyard.position]:
                        ml_move = positionals.Direction.invert(ml_move)

                    movement = game_map.get_safe_move(game_map[ship.position],
                                                        game_map[ship.position.directional_offset(ml_move)])
                    if movement is not None:
                        game_map[ship.position.directional_offset(movement)].mark_unsafe(ship)
                        # f.write(str(movement) +" -- " + str(ship.move(movement)) + "\n")
                        command_queue.append(ship.move(movement))
                        continue
                    
                ship.stay_still()

            if me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied and len(me.get_ships()) < 60 :
                command_queue.append(self.game.me.shipyard.spawn())

            self.game.end_turn(command_queue)  # Send our moves back to the game environment
