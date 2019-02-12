#!/usr/bin/env python3

import SVMBot

import time

class SVMBotAggressive(SVMBot.SVMBot):
	def __init__(self):
		super().__init__()
	

if __name__ == '__main__':
    bot = SVMBotAggressive()
    bot.run()
