#!/usr/bin/env python3

import os
from collections import defaultdict

import hlt
import model
from hlt import constants
from hlt import positionals

import time

import logging


import os
import sys
stderr = sys.stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stderr = open(os.devnull, 'w')
from keras.models import load_model
sys.stderr = stderr
MODEL = load_model("Training models\\halite_bot_estoev_final.h5")

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

        go_home = defaultdict(lambda: False)

        while True:
            self.game.update_frame()
            me = self.game.me  # Here we extract our player metadata from the game state
            game_map = self.game.game_map  # And here we extract the map metadata
            other_players = [p for pid, p in self.game.players.items() if pid != self.game.my_id]

            command_queue = []

            if self.game.turn_number > 385:
                for ship in me.get_ships():
                    movement = game_map.get_unsafe_moves(game_map[ship.position], game_map[me.shipyard.position])
                    if movement is not None:
                        command_queue.append(ship.move(movement))
                    else:
                        ship.stay_still()
                self.game.end_turn(command_queue)
                continue

            for ship in me.get_ships():  # For each of our ships
                # Did not machine learn going back to base. Manually tell ships to return home
                if ship.position == me.shipyard.position:
                    go_home[ship.id] = False
                elif go_home[ship.id] or ship.halite_amount == constants.MAX_HALITE:
                    go_home[ship.id] = True
                    movement = game_map.get_safe_move(game_map[ship.position], game_map[me.shipyard.position])
                    if movement is not None:
                        game_map[ship.position.directional_offset(movement)].mark_unsafe(ship)
                        command_queue.append(ship.move(movement))
                    else:
                        ship.stay_still()
                    continue

                if game_map[ship.position].halite_amount > 100 and ship.position != me.shipyard.position:
                    ml_move = (0,0)
                else:
                    # Use machine learning to get a move
                    ml_move = self.my_model.predict_move(ship, game_map, me, other_players, self.game.turn_number)

                if ml_move is not None:
                    movement = game_map.get_safe_move(game_map[ship.position],
                                                      game_map[ship.position.directional_offset(ml_move)])
                    if movement is not None:
                        game_map[ship.position.directional_offset(movement)].mark_unsafe(ship)
                        command_queue.append(ship.move(movement))
                        continue
                ship.stay_still()

            # Spawn some more ships
            if me.halite_amount >= constants.SHIP_COST and \
                    not game_map[me.shipyard].is_occupied and len(me.get_ships()) <= 4:
                command_queue.append(self.game.me.shipyard.spawn())

            self.game.end_turn(command_queue)  # Send our moves back to the game environment
