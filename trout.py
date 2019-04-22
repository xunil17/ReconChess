import chess.engine
import random
import numpy as np
# from reconchess import *
import os
from player import Player
from typing import List, Tuple, Optional
from enum import Enum
from evaluate_network import fen_to_bin
from knight_movement import minStepToReachTarget

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



class WinReason(Enum):
    """The reason the game ended"""

    KING_CAPTURE = 1
    """The game ended because one player captured the other's king."""

    TIMEOUT = 2
"""The game ended because one player ran out of time"""


STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'

class TroutBot(Player):
    """
    TroutBot uses the Stockfish chess engine to choose moves. In order to run TroutBot you'll need to download
    Stockfish from https://stockfishchess.org/download/ and create an environment variable called STOCKFISH_EXECUTABLE
    that is the path to the downloaded Stockfish executable.
    """

    def __init__(self):
        self.board = None
        self.color = None
        self.my_piece_captured_square = None
        self.nextSenseLocation = None

        self.board_edges = [
        chess.A1, chess.B1, chess.C1, chess.D1, chess.E1, chess.F1, chess.G1, chess.H1, 
        chess.H2, chess.H3, chess.H4, chess.H5, chess.H6, chess.H7, chess.H8, 
        chess.A2, chess.A3, chess.A4, chess.A5, chess.A6, chess.A7, chess.A8, 
        chess.B8, chess.C8, chess.D8, chess.E8, chess.F8, chess.G8]

        self.save_training_name = 'training/training21.csv'
        self.saveFile = open(self.save_training_name, "a")

        # make sure stockfish environment variable exists
        # if STOCKFISH_ENV_VAR not in os.environ:
        #     raise KeyError(
        #         'TroutBot requires an environment variable called "{}" pointing to the Stockfish executable'.format(
        #             STOCKFISH_ENV_VAR))

        # make sure there is actually a file
        # stockfish_path = os.environ[STOCKFISH_ENV_VAR]
        # stockfish_path = 'C:\\Users\\seanc\\Google Drive\\documents\\Robot Intelligence Planning\\ReconChess\\stockfish-10-win\\Windows\\stockfish_10_x64.exe'
        # stockfish_path = 'D:\\Desktop\\stockfish-10-win\\Windows\\stockfish_10_x64.exe'
        # stockfish_path = 'C:\\Users\\seanc\\Desktop\\stockfish-10-win\\Windows\\stockfish_10_x64.exe'

        stockfish_path = '/home/sean/Desktop/stockfish-10-linux/Linux/stockfish_10_x64'

        # stockfish_path = 'stockfish_10_x64.exe'
        if not os.path.exists(stockfish_path):
            raise ValueError('No stockfish executable found at "{}"'.format(stockfish_path))

        # initialize the stockfish engine
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def handle_game_start(self, color: Color, board: chess.Board):
        self.board = board
        self.color = color

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        # if the opponent captured our piece, remove it from our board.
        self.my_piece_captured_square = capture_square
        if captured_my_piece:
            self.board.remove_piece_at(capture_square)

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> Square:
        #if last turn there was a piece where king used to be - look for where king is
        if self.nextSenseLocation:
            # #choose  randomly somewhere close to nextSenseLocation
            # # up down left right
            # dy = [8, -8, 0, 0]
            # dx = [0, 0, -1, 1]
            # senseList = []
            # # only choose right, up, down
            if self.nextSenseLocation % 8 == 0:
                return self.nextSenseLocation + 1
            #     senseList = [self.nextSenseLocation + 8, self.nextSenseLocation - 8, ]
            # # only choose left, up, down
            elif (self.nextSenseLocation + 1) % 8 == 0:
                return self.nextSenseLocation - 1

            else:
                senseList = [self.nextSenseLocation + 1, self.nextSenseLocation - 1]
                return random.choice(senseList)

        # if our piece was just captured, sense where it was captured
        if self.my_piece_captured_square:
            return self.my_piece_captured_square

        # if we might capture a piece when we move, sense where the capture will occur
        future_move = self.choose_move(move_actions, seconds_left, save_bool = False)
        if future_move is not None and self.board.piece_at(future_move.to_square) is not None:
            return future_move.to_square

        # otherwise, just randomly choose a sense action, but don't sense on a square where our pieces are located
        for square, piece in self.board.piece_map().items():
            if (piece.color == self.color) or (square in self.board_edges):
                sense_actions.remove(square)

        # return chess.B2
        return random.choice(sense_actions)

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
                self.nextSenseLocation = square
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

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float, save_bool = True) -> Optional[chess.Move]:
        # if we might be able to take the king, try to
        enemy_king_square = self.board.king(not self.color)
        if enemy_king_square:
            # if there are any ally pieces that can take king, execute one of those moves
            enemy_king_attackers = self.board.attackers(self.color, enemy_king_square)
            if enemy_king_attackers:
                attacker_square = enemy_king_attackers.pop()
                return chess.Move(attacker_square, enemy_king_square)

        # otherwise, try to move with the stockfish chess engine
        try:
            self.board.turn = self.color
            self.board.clear_stack()

            result = self.engine.play(self.board, chess.engine.Limit(time=0.5))
            info = self.engine.analyse(self.board, chess.engine.Limit(depth=15))
            # #print (self.board)
            score = info["score"].relative.score(mate_score = 2000)
            if (score and save_bool):
                relative_score = int(score)
                nn_board = fen_to_bin(self.board.fen())
                nn_board = list(map(str,nn_board))
                # #print(nn_board, info["score"])
                # print ("TROUT SAVE")
                self.saveFile = open(self.save_training_name, "a")
                self.saveFile.write(str(relative_score) + ',' + ','.join(nn_board) + '\n')
                self.saveFile.close()
                # #print("Score:", info["score"])
            return result.move
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:
            print('Engine bad state at "{}"'.format(self.board.fen()))
            return None

        print ("BAD ENGINE")
        # if all else fails, pass
        return None

    # def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
    #                        captured_opponent_piece: bool, capture_square: Optional[Square]):
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

                # what to do when king is the piece that got replaced???
                self.board.set_piece_at(to_square, piece_at_square)
            
            #print (from_square, to_square)
            #print (self.board)
        else:
            if requested_move:
                self.nextSenseLocation = requested_move.to_square
        # #print ("handle move result")
        # #print (self.board)

    def handle_game_end(self, winner_color, win_reason):
        try:
            self.engine.quit()
        except:
            print ('engine wont quit')