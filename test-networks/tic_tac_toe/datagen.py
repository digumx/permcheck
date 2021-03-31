"""

File to generate data for the tic-tac-toe game for training and testing the network. Classification
of a generated valid config and move is done via z3. The board state is encoded as a 27-size bit
vector, three bits for each of the nine positions, the bits represent the board being empty, marked
by 1, or by 2 respectively. A move is represented by a 9-size bit vector, representing which board
position is being marked.

"""

import os.path
import random

import z3

from game_props import *


def gen_data(num, cache_file='data.val'):
    """
    Generates `num` counts of moves encoded as 36-size bit vectors and corresponding labels as a
    good or bad move. This only generates valid board positions, with valid as defined in the
    assignment question. It returns a list of tuples, with each tuple represing an encoding of the
    move and corresponding label, the label being True for a good move, and false otherwise. The
    function catches the generated data at `cache_file` and if the file exists and matches or
    exceeds the length of the data requested, the cached data is returned. Else, the missing data is
    generated
    """
    data = []
    
    # Read catched data
    if os.path.isfile(cache_file):
        with open(cache_file) as f:
            data = eval(f.read())
            print("Read catched data of length", len(data))
            random.shuffle(data)
            print("Shuffled read data")
            if num <= len(data):
                print("More cached data than requested")
                return data[:num]
            num -= len(data)
            print("Generating", num, "more data points")
   
    # Generate extra data
    for n in range(num):
        print("Generating point", n, "of", num, end="\r")

        # Loop until a proper board is generated
        raw_board = []
        while True:
            # Pick number of moves
            m = random.randrange(2, 4)
            
            # Permute markings into board positions
            raw_board = [0]*(9 - 2*m)
            raw_board.extend([1]*m)
            raw_board.extend([2]*m)
            random.shuffle(raw_board)

            # Check if board is a state where any player has already wone, If not, stop looking.
            is_win = False
            for i in range(3):
                is_win |= raw_board[3*i] == raw_board[3*i+1] and raw_board[3*i+1] == raw_board[3*i+2]
                is_win |= raw_board[i] == raw_board[i+3] and raw_board[i+3] == raw_board[i+6]
            is_win |= raw_board[0] == raw_board[4] and raw_board[4] == raw_board[8]
            is_win |= raw_board[2] == raw_board[4] and raw_board[4] == raw_board[6]
            if not is_win:
                break

        # Chose move
        raw_pick = random.choice(list(filter(lambda i: raw_board[i] == 0, range(9))))
       
        # Encode board and move
        move = []
        pick = []
        for b, i in zip(raw_board, range(9)):
            move.extend([k==b for k in range(3)])
            pick.append(i==raw_pick)
        move.extend(pick)

        # Check if any player has won

        # Check if move is good and append to dataset
        data.append((move, check_move(move)))

    # Cache data
    with open(cache_file, 'w') as f:
        f.write(str(data))

    return data
