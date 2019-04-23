import chess

# returns list of squares dist away from input_square
# input_square: int representing desired square
# dist: how many squares 
def getSquaresMinDistAway(input_square, dist):
    all_squares = list(chess.SQUARES)
    return [square for square in all_squares if chess.square_distance(square, input_square) <= dist ]


class cell: 
      
    def __init__(self, x = 0, y = 0, dist = 0): 
        self.x = x 
        self.y = y 
        self.dist = dist 

def isInside(x, y, N): 
    if (x >= 1 and x <= N and 
        y >= 1 and y <= N):  
        return True
    return False

def minStepToReachTarget(knightpos, targetpos, N): 
      
    #all possible movments for the knight 
    dx = [2, 2, -2, -2, 1, 1, -1, -1] 
    dy = [1, -1, 1, -1, 2, -2, 2, -2] 
      
    queue = [] 
      
    # push starting position of knight 
    # with 0 distance 
    queue.append(cell(knightpos[0], knightpos[1], 0)) 
      
    # make all cell unvisited  
    visited = [[False for i in range(N + 1)]  
                      for j in range(N + 1)] 
      
    # visit starting state 
    visited[knightpos[0]][knightpos[1]] = True
      
    # loop until we have one element in queue  
    while(len(queue) > 0): 
          
        t = queue[0] 
        queue.pop(0) 
          
        # if current cell is equal to target  
        # cell, return its distance  
        if(t.x == targetpos[0] and 
           t.y == targetpos[1]): 
            return t.dist 
              
        # iterate for all reachable states  
        for i in range(8): 
              
            x = t.x + dx[i] 
            y = t.y + dy[i] 
              
            if(isInside(x, y, N) and not visited[x][y]): 
                visited[x][y] = True
                queue.append(cell(x, y, t.dist + 1)) 

# Driver Code      
if __name__=='__main__':  
    N = 8
    # knightpos = [1, 1] 
    # targetpos = [30, 30] 

    knightpos = [chess.square_rank(4), chess.square_file(4)]
    targetpos = [chess.square_rank(21), chess.square_file(21)]


    # r . b . k b . r
    # p p p . p p p p
    # . . . . . . . .
    # . . . B . . . .
    # . . . n . . . .
    # . . . . B . . .
    # P P P . . n P P
    # R . K . . . N .
    # old_knight_locations
    # [13, 27]
    # new location
    # [0, 7]
    # old location - 
    # [1, 5]
    # old location - 
    # [3, 3]
    # knight_moves
    # [None, None]


    # print(minStepToReachTarget(knightpos, targetpos, N)) 

    # print(minStepToReachTarget([1+1,5+1], [0+1,7+1], N))
    # print(minStepToReachTarget([3+1,3+1], [0+1,7+1], N))

    board_edges = [
    chess.A1, chess.B1, chess.C1, chess.D1, chess.E1, chess.F1, chess.G1, chess.H1, 
    chess.H2, chess.H3, chess.H4, chess.H5, chess.H6, chess.H7, chess.H8, 
    chess.A2, chess.A3, chess.A4, chess.A5, chess.A6, chess.A7, chess.A8, 
    chess.B8, chess.C8, chess.D8, chess.E8, chess.F8, chess.G8]

    surround_squares =  (getSquaresMinDistAway(chess.G1, 1))
    good_surrounding_squares = list(set(surround_squares) - set(board_edges))
    print (good_surrounding_squares)