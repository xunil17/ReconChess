def fen_to_bin(fen):
    ''' input a fen string representation and return 768 bit representation for NN training'''

    # Each rank is described, starting with rank 8 and ending with rank 1 (top down)
    # upper case - white, lower case - black

    # each square is 12 bits long
    ref_string = "prnbqkPRNBQK"

    fen_list = fen.split(" ")
    board_fen = fen_list[0]
    color_player = fen_list[1]

    rank_list = board_fen.split('/')
    # print(rank_list)

    total_representation = []

    for rank in rank_list: #rank
        for item in rank: #each alphanumeric character
            if item.isalpha(): #square has piece on it
                if color_player == 'w' and item.isupper(): #item is my piece
                    square_bits = [1 if a == item else 0 for a in ref_string]
                elif color_player == 'w' and item.islower(): #item is their piece
                    square_bits = [-1 if a == item else 0 for a in ref_string]
                elif color_player == 'b' and item.isupper(): #item is their piece
                    square_bits = [-1 if a == item else 0 for a in ref_string]
                elif color_player == 'b' and item.islower(): #item is my piece
                    square_bits = [1 if a == item else 0 for a in ref_string]
                else:
                    raise("Color player is wrong")
                total_representation.extend(square_bits)
            else: #number - indicates blank spots
                total_representation.extend([0]*int(item)*12)

    if (len(total_representation) != 768):
        raise("Wrong number of bits in representation")

    return total_representation


if __name__ == '__main__':
    # fen_to_bin("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    fen_to_bin("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
