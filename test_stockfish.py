import chess
import chess.engine
# import chess.engine.Score

stockfish_path = '/home/sean/Desktop/stockfish-10-linux/Linux/stockfish_10_x64'
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)


#original position
# newChessBoard = chess.Board(fen="rnbqk2r/pppp1ppp/8/4p3/3b4/P6P/1PP1BKP1/RNBQ2NR w kq - 0 1")
# result = engine.play(newChessBoard, chess.engine.Limit(time=1.0))

#king in the open
newChessBoard = chess.Board(fen="rnbqk2r/pppp1ppp/8/4p3/3b4/P5KP/1PP1B1P1/RNBQ2NR w kq -")
info = engine.analyse(newChessBoard, chess.engine.Limit(depth=10))
print (info)
print (info["score"].relative.score(mate_score = 3000))
# +36

print ('\n')

#recommended king move
newChessBoard = chess.Board(fen="rnbqk2r/pppp1ppp/8/4p3/3b4/P6P/1PP1B1P1/RNBQ1KNR w kq -")
info = engine.analyse(newChessBoard, chess.engine.Limit(depth=10))
print (info["score"].relative.score(mate_score = 3000))
# +143

# print (result.move)
# newChessBoard.turn = False
# print (newChessBoard.is_check())
# print (newChessBoard.was_into_check())
# print ()
# info = engine.analyse(newChessBoard, chess.engine.Limit(depth=18))

# print (info)
# print (info["score"])
# print (info)
# print(info["score"].black())
# print (info["score"].relative.score(mate_score = 3000))
# score = info["score"].relative.score()
# relative_score = int(score)

# print (relative_score)
engine.quit()