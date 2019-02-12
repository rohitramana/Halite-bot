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

MODEL = load_model("halite_bot_teccles_diamond.h5")


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
        f = open("time.txt", "w")
        go_home = defaultdict(lambda: False)

        while True:
            # t1 = time.time()
            self.game.update_frame()
            me = self.game.me  # Here we extract our player metadata from the game state
            game_map = self.game.game_map  # And here we extract the map metadata
            other_players = [p for pid, p in self.game.players.items() if pid != self.game.my_id]

            command_queue = []

            if self.game.turn_number > 385:
                for ship in me.get_ships():
                    movement = game_map.get_safe_move(game_map[ship.position], game_map[me.shipyard.position])
                    if movement is not None:
                        command_queue.append(ship.move(movement))
                    else:
                        ship.stay_still()
                self.game.end_turn(command_queue)
                continue

            elif self.game.turn_number < 10:
                for ship in me.get_ships():
                    
                    move =  game_map.get_safe_move(game_map[ship.position], game_map[ship.position.directional_offset(random.choice([positionals.Direction.West, positionals.Direction.North, positionals.Direction.East, positionals.Direction.South]))])
                    count = 0
                    while move == None:
                        move =  game_map.get_safe_move(game_map[ship.position], game_map[ship.position.directional_offset(random.choice([positionals.Direction.West, positionals.Direction.North, positionals.Direction.East, positionals.Direction.South]))])
                        count += 1
                        if count > 3: 
                            break
                    if move is not None:
                        command_queue.append(ship.move(move))
                    else:
                        command_queue.append(ship.stay_still())
                self.game.end_turn(command_queue)
                continue

            else:
                
                for ship in me.get_ships():  # For each of our ships
                        # Did not machine learn going back to base. Manually tell ships to return home
                        decision = -1
                        if ship.position == me.shipyard.position:
                            go_home[ship.id] = False

                        elif go_home[ship.id] or ship.halite_amount > constants.MAX_HALITE / 2:
                            go_home[ship.id] = True
                            movement = game_map.get_safe_move(game_map[ship.position], game_map[me.shipyard.position])
                            if movement is not None:
                                game_map[ship.position.directional_offset(movement)].mark_unsafe(ship)
                                command_queue.append(ship.move(movement))
                                decision = 1
                                
                            else:
                                move =  game_map.get_safe_move(game_map[ship.position], game_map[ship.position.directional_offset(random.choice([positionals.Direction.West, positionals.Direction.North, positionals.Direction.East, positionals.Direction.South]))])
                                count = 0
                                while move == None:
                                    move =  game_map.get_safe_move(game_map[ship.position], game_map[ship.position.directional_offset(random.choice([positionals.Direction.West, positionals.Direction.North, positionals.Direction.East, positionals.Direction.South]))])
                                    count += 1
                                    if count > 3: 
                                        break
                                if move is not None:
                                    command_queue.append(ship.move(move))
                                    decision = 2
                                else:
                                    command_queue.append(ship.stay_still())
                                    decision = 3
                                    
                            f.write("move ==== {}, ship ==== {}, game_round ==== {}, decision ==== {}\n".format(movement, ship.id, self.game.turn_number, decision))        
                            continue

                        
                    # Use machine learning to get a move
                        ml_move = self.my_model.predict_move(ship, game_map, me, other_players, self.game.turn_number)

                        
                        if ml_move is not None:
                            movement = game_map.get_safe_move(game_map[ship.position],
                                                              game_map[ship.position.directional_offset(ml_move)])
                           
                            if movement is not None:
                                game_map[ship.position.directional_offset(movement)].mark_unsafe(ship)
                                command_queue.append(ship.move(movement))
                                decision = 4
                                f.write("move ==== {}, ship ==== {}, game_round ==== {}, decision ==== {} \n".format(ml_move, ship.id, self.game.turn_number, decision))
                                continue
                            else:

                                if (ship.halite_amount < constants.MAX_HALITE / 2  and game_map[ship.position].halite_amount > 0) and game_map[ship.position] != game_map[me.shipyard.position]:
                                    ship.stay_still()
                                    decision = 5

                                elif game_map[ship.position] == game_map[me.shipyard.position]:
                                    # f.write(type(game_map[ship.position.directional_offset(random.choice([positionals.Direction.West]))]))
                                    move =  game_map.get_safe_move(game_map[ship.position], game_map[ship.position.directional_offset(random.choice([positionals.Direction.West, positionals.Direction.North, positionals.Direction.East, positionals.Direction.South]))])
                                    count = 0
                                    while move == None:
                                        move =  game_map.get_safe_move(game_map[ship.position], game_map[ship.position.directional_offset(random.choice([positionals.Direction.West, positionals.Direction.North, positionals.Direction.East, positionals.Direction.South]))])
                                        count += 1
                                        if count > 3: 
                                            break
                                    if move is not None:
                                        command_queue.append(ship.move(move))
                                        decision = 6
                                    else:
                                        command_queue.append(ship.stay_still())
                                        decision = 7

                                else:
                                    movement = game_map.get_safe_move(game_map[ship.position], game_map[me.shipyard.position])
                                    decision = 8
                                    if movement is not None:
                                        game_map[ship.position.directional_offset(movement)].mark_unsafe(ship)
                                        command_queue.append(ship.move(movement))
                                        decision = 9
                        else:
                            ship.stay_still()
                            decision = 10
                        f.write("move ==== {}, ship ==== {}, game_round ==== {}, decision ==== {} \n".format(ml_move, ship.id, self.game.turn_number, decision))
                            

                
            # Spawn some more ships
            if me.halite_amount/2 >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied :
                command_queue.append(self.game.me.shipyard.spawn())

            f.write("move ==== {} \n".format(len(me.get_ships())))

            self.game.end_turn(command_queue)  # Send our moves back to the game environment
            # t2 = time.time()
            # f.write(t2 - t1)    


        