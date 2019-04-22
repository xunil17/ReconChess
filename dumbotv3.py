import chess
import random
import numpy as np
import os
from player import Player
from typing import List, Tuple, Optional
from enum import Enum
from evaluate_network import fen_to_bin
from knight_movement import minStepToReachTarget
from keras_chess import kerasChessNetwork
import copy
import time

Square = int
Color = bool
PieceType = int

#given board state, want to sense stuff that gives you most information
# places where good information are?

# where last piece was captured
# most places where your pieces are attacking
# never sense in the corners
# based on where my pieces are
# maybe where evaluation funtion of stockfish changed the most?? before and after sense
# 


# how to update board based on sensing to get rid of multiple pieces showing
# save which sensing time that piece was last seen

# always correct assumption
# king - always remove old
# bishop - always remove old on same color diagonal

# most of the time correct assumption
# pawn - if right after capture, leave there, else, remove the one in same column below
# queen - always remove old

# maybe correct assumption??
# knight - if more than 2 - remove one of the old ones
# rook - if more than 2 - remove one of the old ones
# ideally - you should know which knight or rook it is to remove

# move sequences from white's perspective, flipped at runtime if playing as black
QUICK_ATTACKS = [
    # queen-side knight attacks
    [chess.Move(chess.B1, chess.C3), chess.Move(chess.C3, chess.B5), chess.Move(chess.B5, chess.D6),
     chess.Move(chess.D6, chess.E8)],

    [chess.Move(chess.B1, chess.C3), chess.Move(chess.C3, chess.E4), chess.Move(chess.E4, chess.F6),
     chess.Move(chess.F6, chess.E8)],

    # king-side knight attacks
    [chess.Move(chess.G1, chess.H3), chess.Move(chess.H3, chess.F4), chess.Move(chess.F4, chess.H5),
     chess.Move(chess.H5, chess.F6), chess.Move(chess.F6, chess.E8)],

    # four move mates
    [chess.Move(chess.E2, chess.E4), chess.Move(chess.F1, chess.C4), chess.Move(chess.D1, chess.H5), chess.Move(
        chess.C4, chess.F7), chess.Move(chess.F7, chess.E8), chess.Move(chess.H5, chess.E8)],

    # no weird moves
    [],
]

# QUICK_ATTACKS = [[]]

def flipped_move(move):
    def flipped(square):
        return chess.square(chess.square_file(square), 7 - chess.square_rank(square))

    return chess.Move(from_square=flipped(move.from_square), to_square=flipped(move.to_square), promotion=move.promotion, drop=move.drop)



class WinReason(Enum):
    """The reason the game ended"""

    KING_CAPTURE = 1
    """The game ended because one player captured the other's king."""

    TIMEOUT = 2
"""The game ended because one player ran out of time"""

class dumbotV3(Player):

    def __init__(self):
        self.board = None
        self.color = None
        self.my_piece_captured_square = None
        self.nextSenseLocation = None

        #SET TO EMPTY LIST TO DISABLE OPENING ATTACKS
        self.move_sequence = random.choice(QUICK_ATTACKS)
        # self.move_sequence = []

        self.board_edges = [
        chess.A1, chess.B1, chess.C1, chess.D1, chess.E1, chess.F1, chess.G1, chess.H1, 
        chess.H2, chess.H3, chess.H4, chess.H5, chess.H6, chess.H7, chess.H8, 
        chess.A2, chess.A3, chess.A4, chess.A5, chess.A6, chess.A7, chess.A8, 
        chess.B8, chess.C8, chess.D8, chess.E8, chess.F8, chess.G8]


        self.chessNet = kerasChessNetwork('dumbot_weights.hdf5')


    def handle_game_start(self, color: Color, board: chess.Board):
        self.board = board
        self.color = color
        if color == chess.BLACK:
            self.move_sequence = list(map(flipped_move, self.move_sequence))

        self.int_set = 2
        # if self.color == chess.BLACK: #black need to set to 2
        #     self.int_set = 2
        # elif self.color == chess.WHITE: # white, need to set to 0
        #     self.int_set = 0

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        # if the opponent captured our piece, remove it from our board.
        self.my_piece_captured_square = capture_square
        if captured_my_piece:
            self.board.remove_piece_at(capture_square)

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> Square:
        #if last turn there was a piece where king used to be - look for where king is
        if self.nextSenseLocation:
            print ("Sensing Last Weird Location")
            print (self.nextSenseLocation)
            nextSenseSquare = self.nextSenseLocation[0]
            nextSenseKing = self.nextSenseLocation[1]
            if nextSenseKing:
                #choose  randomly somewhere close to nextSenseLocation
                # up down left right
                dy = [8, -8, 0, 0]
                dx = [0, 0, -1, 1]
                senseList = []
                # only choose right, up, down
                if nextSenseSquare % 8 == 0:
                    return nextSenseSquare + 1
                #     senseList = [self.nextSenseLocation + 8, self.nextSenseLocation - 8, ]
                # # only choose left, up, down
                elif (nextSenseSquare + 1) % 8 == 0:
                    return nextSenseSquare - 1

                else:
                    senseList = [nextSenseSquare + 1, nextSenseSquare - 1]
                    return random.choice(senseList)
            else:
                # return where our last move failed
                return nextSenseSquare

        # if our piece was just captured, sense where it was captured
        if self.my_piece_captured_square:
            print ("Sensing where last piece was captured")
            return self.my_piece_captured_square

        # if we might capture a piece when we move, sense where the capture will occur
        future_move = self.choose_move(move_actions, seconds_left, save_bool = False, sensing = True)
        if future_move is not None and self.board.piece_at(future_move.to_square) is not None:
            print ("Sensing future want to capture square")
            return future_move.to_square

        # print (sense_actions)
        # otherwise, just randomly choose a sense action, but don't sense on a square where our pieces are located
        good_sense_actions = []
        for square in sense_actions:
            piece = self.board.piece_at(square)
            if square not in self.board_edges:
                good_sense_actions.append(square)
                if piece:
                    if piece.color == self.color:
                        good_sense_actions.remove(square)


        print ("Sensing randomly")
        print (good_sense_actions)
        return random.choice(good_sense_actions)

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        # add the pieces in the sense result to our board

        self.nextSenseLocation = None
        for square, piece in sense_result:
            if (piece):
                if piece.color != self.color:
                    letter = piece.symbol()
                    #print (chess.SQUARE_NAMES[square], letter)
                    if letter.lower() == "k":
                        self.handle_sensed_king(square, piece)
                    elif letter.lower() == "b":
                        self.handle_sensed_bishop(square, piece)
                    elif letter.lower() == "q":
                        self.handle_sensed_queen(square, piece)
                    elif letter.lower() == "p":
                        self.handle_sensed_pawn(square, piece)
                    elif letter.lower() == "n":
                        self.handle_sensed_knight(square, piece)
                    elif letter.lower() == "r":
                        self.handle_sensed_rook(square, piece)
                    else:
                        self.set_board_piece(square, piece)
            else:
                #empty square sensed
                self.set_board_piece(square, piece)
        # #print ("handle sense result")
        #print(self.board)

    def set_board_piece(self, square, piece):
        piece_at_square = self.board.piece_at(square)

        if (piece_at_square):
            if piece_at_square.symbol().lower() == "k":
                #trying to set new thing to where old king was
                #print ("trying to set new piece where king used to be")
                #don't place new piece here just yet - look for king
                self.nextSenseLocation = (square, True)
            else:
                self.board.set_piece_at(square, piece)
        else:
            #old square is empty - place new piece there
            self.board.set_piece_at(square, piece)

    def handle_sensed_knight(self, new_knight_square, piece):
        old_knight = np.array(self.board.pieces(chess.KNIGHT, not self.color).tolist())
        old_knight_locations = np.where(old_knight == True)[0].tolist()

        #print ("new_knight_square")
        #print (new_knight_square)

        # if there were old knight locations, look for them and remove one
        if old_knight_locations:
            # print (self.board)
            new_loc = [chess.square_rank(new_knight_square)+1, chess.square_file(new_knight_square)+1]
            # print ("old_knight_locations")
            # print (old_knight_locations)
            # print ('new location')
            # print (new_loc)
            if new_knight_square not in old_knight_locations:
                knight_moves = []
                for s in old_knight_locations:
                    old_loc = [chess.square_rank(s)+1, chess.square_file(s)+1]
                    # print("old location - ")
                    # print (old_loc)
                    knight_moves.append(minStepToReachTarget(old_loc, new_loc, 8))
 
                # print ("knight_moves")
                # print (knight_moves)
                ind = np.argmin(knight_moves)
                self.board.remove_piece_at(old_knight_locations[ind])

        self.set_board_piece(new_knight_square, piece)


    def handle_sensed_rook(self,new_rook_square, piece):
        self.set_board_piece(new_rook_square, piece)

    def handle_sensed_king(self, new_king_square, piece):
        ''' updates self.board if sensed king'''

        old_king = np.array(self.board.pieces(chess.KING, not self.color).tolist())
        old_king_locations = np.where(old_king == True)[0]

        if len(old_king_locations) == 1:
            self.board.remove_piece_at(old_king_locations[0])

        self.set_board_piece(new_king_square, piece)

        # #print (old_king_locations)
        # #print (self.board)

    def handle_sensed_bishop(self, new_bishop_square, piece):
        ''' updates self.board if sensed bishop'''
        # new_bishop = chess.SquareSet([new_bishop_square])
        dark_squares = chess.SquareSet(chess.BB_DARK_SQUARES)
        light_squares = chess.SquareSet(chess.BB_LIGHT_SQUARES)


        old_bishop = np.array(self.board.pieces(chess.BISHOP, not self.color).tolist())
        old_bishop_locations = np.where(old_bishop == True)[0]

        if new_bishop_square in dark_squares:
            for bishop in old_bishop_locations:
                if bishop in dark_squares:
                    self.board.remove_piece_at(bishop)
        else:
            for bishop in old_bishop_locations:
                if bishop in light_squares:
                    self.board.remove_piece_at(bishop)
        
        self.set_board_piece(new_bishop_square, piece)

        # #print (self.board)

    def handle_sensed_pawn(self, new_pawn_square, piece):
        ''' updates self.board if sensed pawn'''
        # only get rid of old pawn in column if this pawn didn't just capture
        # print (chess.SQUARE_NAMES[new_pawn_square])
        # print (new_pawn_square)
        # print (self.my_piece_captured_square)
        if not self.my_piece_captured_square == new_pawn_square:
            # remove furthest pawn in column
            pawn_file = chess.square_file(new_pawn_square)
            pawn_file_list = np.array(chess.SquareSet(chess.BB_FILE_MASKS[pawn_file]).tolist())
            pawn_file_locations = np.where(pawn_file_list == True)[0]

            #if playing as black, must search from top down instead of bottom up
            if self.color == chess.BLACK:
                np.flip(pawn_file_locations)

            for square in pawn_file_locations:
                if self.board.piece_at(square) == piece:
                    self.board.remove_piece_at(square)
                    break
            self.set_board_piece(new_pawn_square, piece)

        # else:
            # #print ("JUST GOT PIECE CAPTURED")


        self.set_board_piece(new_pawn_square, piece)

    def handle_sensed_queen(self, new_queen_square, piece):

        old_queen = np.array(self.board.pieces(chess.QUEEN, not self.color).tolist())
        old_queen_locations = np.where(old_queen == True)[0]

        if len(old_queen_locations) == 1:
            self.board.remove_piece_at(old_queen_locations[0])

        self.set_board_piece(new_queen_square, piece)

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float, save_bool = False, sensing = False) -> Optional[chess.Move]:
        
        # move_sequence is opening attack, pop and return
        if not sensing:
            while len(self.move_sequence) > 0 and self.move_sequence[0] not in move_actions:
                # print (self.move_sequence)
                self.move_sequence.pop(0)

        self.board.turn = self.color
        # after opening attack
        if len(self.move_sequence) == 0 or self.board.is_check():
            
            # try to take king if we can and we are not in check ourselves
            if not self.board.is_check():
                enemy_king_square = self.board.king(not self.color)
                if enemy_king_square:
                    # if there are any ally pieces that can take king, execute one of those moves
                    enemy_king_attackers = self.board.attackers(self.color, enemy_king_square)
                    if enemy_king_attackers:
                        attacker_square = enemy_king_attackers.pop()
                        return chess.Move(attacker_square, enemy_king_square)


            # otherwise, move with neural network
            print ("OGBOARD")
            # print (self.board.fen())
            print (self.board)
            print ('\n')

            scores = []
            for move in move_actions:
                self.board.turn = self.color
                newChessBoard = copy.deepcopy(self.board)
                

                from_square = move.from_square
                to_square = move.to_square

                piece_at_square = newChessBoard.piece_at(from_square)

                newChessBoard.push(move)
                # if piece_at_square.symbol().lower() == "k":
                #     newChessBoard.push(move)
                # else:
                #     newChessBoard.remove_piece_at(from_square)
                #     newChessBoard.set_piece_at(to_square, piece_at_square)

                # if we make this move, it will be opponent turn, look for min score from opponent side
                newChessBoard.turn = not self.color

                #if we are trying to move pawn diagonally and there is no piece there -> don't move
                if self.board.piece_at(to_square) == None and piece_at_square.symbol().lower() == "p" and chess.square_file(from_square) != chess.square_file(to_square):
                    newBoardScore = self.int_set

                #if there is a piece in front and our pawn is trying to move there -> can't take that piece
                elif self.board.piece_at(to_square) and piece_at_square.symbol().lower() == "p" and chess.square_file(from_square) == chess.square_file(to_square):
                    newBoardScore = self.int_set
                
                # if we are in check - try to get out
                elif newChessBoard.was_into_check():
                    newBoardScore = self.int_set
                else:
                    nn_board = fen_to_bin(newChessBoard.fen())
                    newBoardScore = self.chessNet.retrieveBoardValueNeuralNet(nn_board)
                # print ('\n')
                # print (newChessBoard.fen())
                # print (newChessBoard)
                # print (newBoardScore)
                # print (move)

                scores.append(newBoardScore)

            # if self.color == chess.BLACK: #black
            ind = np.argmin(scores)
            # elif self.color == chess.WHITE:
                # ind = np.argmax(scores)
            return move_actions[ind]
        else:
            # do opening attack
            desired_move = self.move_sequence[0]
            if sensing:
                return desired_move
            else:
                return self.move_sequence.pop(0)

    def handle_move_result(self, requested_move, taken_move, reason, captured_piece, captured_square):
        # if a move was executed, apply it to our board
        if taken_move is not None:
            #print (self.board)
            #print ("PUSH MOVE")
            # #print (taken_move)

            # avoid using board.push because pieces turned to other color
            from_square = taken_move.from_square
            to_square = taken_move.to_square
            piece_at_square = self.board.piece_at(from_square)

            # handle case where the piece moved was king - could be castling
            if piece_at_square.symbol().lower() == "k":
                self.board.push(taken_move)
            else:

                self.board.remove_piece_at(from_square)
                
                opponent_piece_at_to_square = self.board.piece_at(to_square)
                # if piece at the square was king, look for it next turn
                if opponent_piece_at_to_square:
                    if opponent_piece_at_to_square.symbol().lower() == "k":
                        self.nextSenseLocation = (to_square, False)

                # regardless of what the piece at the to_square is, place our piece there now
                self.board.set_piece_at(to_square, piece_at_square)
                

            
            #print (from_square, to_square)
            #print (self.board)
        else:
            self.nextSenseLocation = (requested_move.to_square, False)
        # #print ("handle move result")
        # #print (self.board)

    def handle_game_end(self, winner_color, win_reason):
        if winner_color == self.color:
            print ("Dumbot won!!")
        else:
            print ("Dumbot lost :(")


def createBoardFromMove(chessBoard, chessMove):
    newChessBoard = copy.deepcopy(chessBoard)

    from_square = chessMove.from_square
    to_square = chessMove.to_square

    piece_at_square = newChessBoard.piece_at(from_square)
    newChessBoard.remove_piece_at(from_square)
    newChessBoard.set_piece_at(to_square, piece_at_square)
    return newChessBoard