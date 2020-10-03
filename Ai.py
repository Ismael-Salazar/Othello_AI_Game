#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
An AI player for Othello.

@author: Ismael Salazar
"""

import random
import sys
import time

# You can use the functions in othello_shared to write your AI 
from othello_shared import find_lines, get_possible_moves, get_score, play_move

#The global dictionary used for caching, maps board to
#a utility.
seenBoards = {}

#Computes utility of given board by counting the
#number of color on given board.
def compute_utility(board, color):
    
    score = get_score(board)
    
    if color == 1:
        utility = score[0] - score[1]
    else:
        utility = score[1] - score[0]
    
    return utility
        


############ MINIMAX ###############################

#Gets all possible moves given a board and the current color, and
#finds move that produces the lowest utility value using the
#MINIMAX algorithm.
def minimax_min_node(board, color):

    #Finds opponents color.
    if color == 1:
        oppColor = 2
    else:
        oppColor = 1
            
    #Gets all possible moves and finds the move 
    #that produces the lowest utility value.
    allMoves = get_possible_moves(board, color)
    
    #Checks if there are no longer any moves left.
    if not allMoves:
        return compute_utility(board, color)
        
    currSmallest = float("inf")
        
    for move in allMoves:
        MinimaxValue = minimax_max_node(play_move(board, oppColor, move[0], move[1]), oppColor)
            
        if MinimaxValue < currSmallest:
            currSmallest = MinimaxValue
                
    return currSmallest

#Gets all possible moves given a board and current color and
#finds move that produces the highest utility value.
def minimax_max_node(board, color):
    
    allMoves = get_possible_moves(board, color)
    
    #Checks if there are no longer any moves left.
    if not allMoves:
        return compute_utility(board, color)
    
    currBiggest = float("-inf") 
        
    for move in allMoves:
        value = minimax_min_node(play_move(board, color, move[0], move[1]), color)
            
        if value > currBiggest:
            currBiggest = value
    return currBiggest

  
#Begins the Minimax Algorithm given a board and color.
#Gets all possible moves given a board and current color and
#finds move coordinate that produces the highest utility value
def select_move_minimax(board, color):
    
    allMoves = get_possible_moves(board, color)
    
    if not allMoves:
        return [0,0]
    
    #Used to keep track of the position for the move that
    #produces the highest utility.
    currBiggest = float("-inf")
    currBigIVal = 0
    currBigJVal = 0
    
    for move in allMoves:
        
        value = minimax_min_node(play_move(board, color, move[0], move[1]), color)
        
        if value > currBiggest:
            currBiggest = value 
            currBigIVal = move[0]
            currBigJVal = move[1]
        
    return [currBigIVal, currBigJVal]

############ ALPHA-BETA PRUNING #####################

#Implements alpha-beta pruning by passing along alpha and
#beta values to possibly ignore exploring states.
#seenBoards is used to as cache memory to not repeat calculations.
#Nodes are placed in ascending order to explore nodes with higher
#utility values. Level provides a mode by which to limit the search space.
def alphabeta_min_node(board, color, alpha, beta, level, limit): 

    global seenBoards

    #Checks if board utility has already been calculated.
    if board in seenBoards:
        return seenBoards.get(board)
    
    if level == limit:
        return compute_utility(board,color)

    #Finds opponents color.
    if color == 1:
        oppColor = 2
    else:
        oppColor = 1

    allMoves = get_possible_moves(board, color)
    
    #Leaf nodes, so compute utility.
    if not allMoves:
        boardUtility = compute_utility(board, color)
        seenBoards[board] = boardUtility
        return boardUtility
    
    #Sorts allMoves by utility so boards with higher utility and
    #will be explored first using Python's sort method
    sortedMoves = []
    
    for move in allMoves:
        sortedMoves.append([compute_utility(play_move(board, color, move[0], move[1]), color), [move[0], move[1]]])
        
    sortedMoves.sort(reverse = True)
    
    newMoves = []
    for move in sortedMoves:
        newMoves.append(move[1])
        
    allMoves = tuple(newMoves)

    currUtility = float("inf")
    
    #Goes through and looks for nodes to either process or ignore.
    for move in allMoves:
        
        newBoard = play_move(board, oppColor, move[0], move[1])
        utility = alphabeta_max_node(newBoard, oppColor, alpha, beta, level + 1, limit)
            
        seenBoards[newBoard] = utility
        
        if utility < currUtility:
            currUtility = utility
        
        if currUtility <= alpha:
            return currUtility
        
        if currUtility < beta:
            beta = currUtility
            
    return currUtility


#Implements alpha-beta pruning by passng along alpha and
#beta values to possibly ignore exploring states. seenBoards is
#used to as cache memory to not repeat calculations.
#Nodes are placed in ascending order to explore nodes with higher
#utility values.
def alphabeta_max_node(board, color, alpha, beta, level, limit):

    global seenBoards

    #Checks if board utility has already been calculated
    if board in seenBoards:
        return seenBoards.get(board)
    
    #Checks if limit has been reach, in which case only
    #the utility is provided.
    if level == limit:
        return compute_utility(board,color)

    allMoves = get_possible_moves(board, color)
    
    #Leaf node, so just compute utility.
    if not allMoves:
        boardUtility = compute_utility(board, color)
        seenBoards[board] = boardUtility
        return boardUtility
    
    #Sorts all moves by utility so boards with higher utility
    #will be explored first using the sort method in python.
    sortedMoves = []
    
    for move in allMoves:
        sortedMoves.append([compute_utility(play_move(board, color, move[0], move[1]), color), [move[0], move[1]]])
        
    sortedMoves.sort(reverse = True)
    
    newMoves = []
    for move in sortedMoves:
        newMoves.append(move[1])
        
    allMoves = tuple(newMoves)
    
    currUtility = float("-inf")
    
    for move in allMoves:
        
        newBoard = play_move(board, color, move[0], move[1])
        utility = alphabeta_min_node(newBoard, color, alpha, beta, level + 1, limit)
            
        seenBoards[newBoard] = utility
        
        if utility > currUtility:
            currUtility = utility
        
        if currUtility >= beta:
            return currUtility
        
        if currUtility > alpha:
            alpha = currUtility
            
    return currUtility

#Initalizes alpha-beta pruning by processing the intial board
#as a max mode and keeps track of move with highest utility.
#Limit can be varied depending on size of board and time
#constraint.
def select_move_alphabeta(board, color): 
    
    global seenBoards
    
    allMoves = get_possible_moves(board, color)
    
    #Sorts allMoves by utility so boards with higher utility
    #will be explored first using sort method in python.
    sortedMoves = []
    
    for move in allMoves:
        sortedMoves.append([compute_utility(play_move(board, color, move[0], move[1]), color), [move[0], move[1]]])
        
    sortedMoves.sort(reverse = True)
    
    newMoves = []
    for move in sortedMoves:
        newMoves.append(move[1])
        
    allMoves = tuple(newMoves)
    
    #Used to keep track of the position for the move that
    #produces the highest utility.
    currUtility = float("-inf")
    currUtiIVal = 0
    currUtiJVal = 0
    
    #Limit is set arbitrarily to 4 for 8x8 board.
    limit = 5
    
    for move in allMoves:
        
        newBoard = play_move(board, color, move[0], move[1])
            
        utility = alphabeta_min_node(newBoard, color, float("-inf"), float("inf"), 0, limit)
        seenBoards[newBoard] = utility
        
        if utility > currUtility:
            currUtility = utility
            currUtiIVal = move[0]
            currUtiJVal = move[1]
    
    return [currUtiIVal, currUtiJVal]

####################################################
def run_ai():
    """
    This function establishes communication with the game manager. 
    It first introduces itself and receives its color. 
    Then it repeatedly receives the current score and current board state
    until the game is over. 
    """
    print("Minimax AI") # First line is the name of this AI  
    color = int(input()) # Then we read the color: 1 for dark (goes first), 
                         # 2 for light. 

    while True: # This is the main loop 
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input() 
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL": # Game is over. 
            print 
        else: 
            board = eval(input()) # Read in the input and turn it into a Python
                                  # object. The format is a list of rows. The 
                                  # squares in each row are represented by 
                                  # 0 : empty square
                                  # 1 : dark disk (player 1)
                                  # 2 : light disk (player 2)
                    
            # Select the move and send it to the manager 
            movei, movej = select_move_alphabeta(board, color)
            #movei, movej = select_move_alphabeta(board, color)
            print("{} {}".format(movei, movej)) 


if __name__ == "__main__":
    run_ai()
