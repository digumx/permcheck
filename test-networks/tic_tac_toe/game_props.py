"""

Method to check define in z3 properties of game states. Encodings are as described in problem statement"

"""

import z3


def encode_has_winning_move(z3_board, player):
    """
    Given a list of 27 z3 BoolSort constants encoding a board state as described above, return a z3
    boolean expression encoding the condition that the given player has a winning move from this
    board state.
    """

    ret = False

    # Check if there is a possible victory for player by completing a row or a column
    for i in range(3):
        for j in range(3):
            # Check if ith row can be completed in jth position
            ret = z3.Or(ret, z3.And([z3_board[9*i + 3*k + (0 if j == k else player)] for k in range(3)]))
            # Check if ith column can be completed in jth postion
            ret = z3.Or(ret, z3.And([z3_board[3*i + 9*k + (0 if j == k else player)] for k in range(3)]))

    # Check diagonals
    for i in range(3):
        # Can the primary diagonal be completed in ith position
        ret = z3.Or(ret, z3.And([z3_board[3*k + 9*k + (0 if i == k else player)] for k in range(3)]))
        # Can the secondary diagonal be completed in the ith position
        ret = z3.Or(ret, z3.And([z3_board[3*(2-k) + 9*k + (0 if i == k else player)] for k in range(3)]))

    return ret


def encode_has_won(z3_board, player):
    """
    Given a list of 27 z3 BoolSort constants encoding a board state as described above, return a z3
    boolean expression encoding the condition that the given player has won the game
    """

    ret = False

    # Check if a row or a column is complete
    for i in range(3):
        # Check ith row
        ret = z3.Or(ret, z3.And([z3_board[9*i + 3*j + player] for j in range(3)]))
        # Check column
        ret = z3.Or(ret, z3.And([z3_board[3*i + 9*j + player] for j in range(3)]))

    # Check primary diagonal
    ret = z3.Or(ret, z3.And([z3_board[3*i + 9*i + player] for i in range(3)]))
    # Check other diagonal
    ret = z3.Or(ret, z3.And([z3_board[3*(2-i) + 9*i + player] for i in range(3)]))

    return ret


def encode_move(z3_board_from, z3_board_to, z3_move, player):
    """
    Return a boolean z3 expression that encodes the condition that `z3_board_to` is a board
    representation that is obtained from the board representation `z3_board_from` by performing the
    move represented by `z3_move` for the player `player`.
    """

    ret = True

    for i in range(9):
        ret = z3.And(ret, z3.If(z3_move[i], 
                        z3.And([z3_board_to[3*i + j] if j==player else z3.Not(z3_board_to[3*i + j])
                                    for j in range(3)]),
                        z3.And([z3_board_from[3*i + j] == z3_board_to[3*i + j] 
                                    for j in range(3)])))
    return ret 


def check_move(move):
    """
    Given a move, use Z3 to check if it is good or bad. The move is encodeed as a 36-size bit vector
    as described above. Assumes no player has already won.
    """

    # Initialize and introduce z3 constants
    solver = z3.Solver()
    z3_board        = [z3.Const("board_state_" + str(i), z3.BoolSort()) for i in range(27)]
    z3_board_res    = [z3.Const("res_board_state_" + str(i), z3.BoolSort()) for i in range(27)]

    # Add constraints for input
    for z3_const, cond in zip(z3_board, move[:27]):
        solver.add(z3_const if cond else z3.Not(z3_const))

    # Add constraints for the fact that z3_board_res is obtained via move on z3_board
    solver.add(encode_move(z3_board, z3_board_res, move[27:], 1))

    # If player one can win, move must be winnig
    solver.add(z3.Implies(encode_has_winning_move(z3_board, 1), encode_has_won(z3_board_res, 1)))
    
    # If player one cannot win, player one should not win in next round
    solver.add(z3.Implies(z3.Not(encode_has_winning_move(z3_board, 1)),
                            z3.Not(encode_has_winning_move(z3_board_res, 2))))
    # Finally, check sat
    return solver.check() == z3.sat


def assert_valid_move(solver, z3_move):
    """
    Given a z3 representation of a move as a 36-size bit vector, add assertions to the solver
    encoding the fact that the board position is valid as defined in the problem statement
    """
    
    # Should not be a winning state for iether player
    solver.add(z3.Not(encode_has_won(z3_move[:27], 1)))
    solver.add(z3.Not(encode_has_won(z3_move[:27], 2)))

    # Each cell can be 0, 1 or 2
    for i in range(9):
        solver.add(z3.Not(z3.Or([z3.And(z3_move[3*i + j], z3_move[3*i + k] )
                                    for j,k in [(0,1), (0,2), (1,2)]])))
        solver.add(z3.Or(z3_move[3*i:3*(i+1)]))

    # Atleast two moves
    atleast_2_moves_p1 = False
    atleast_2_moves_p2 = False
    for i in range(9):
        for j in range(i):
            atleast_2_moves_p1 = z3.Or(atleast_2_moves_p1, 
                                        z3.And(z3_move[i*3 + 1], z3_move[j*3 + 1]))
            atleast_2_moves_p2 = z3.Or(atleast_2_moves_p2, 
                                        z3.And(z3_move[i*3 + 2], z3_move[j*3 + 2]))

    # Atleast three moves
    atleast_3_moves_p1 = False
    atleast_3_moves_p2 = False
    for i in range(9):
        for j in range(i):
            for k in range(j):
                atleast_3_moves_p1 = z3.Or(atleast_3_moves_p1, 
                                    z3.And(z3_move[i*3 + 1], z3_move[j*3 + 1], z3_move[k*3 + 1]))
                atleast_3_moves_p2 = z3.Or(atleast_3_moves_p2, 
                                    z3.And(z3_move[i*3 + 2], z3_move[j*3 + 2], z3_move[k*3 + 2]))
    # Atleast four moves
    atleast_4_moves_p1 = False
    atleast_4_moves_p2 = False
    for i in range(9):
        for j in range(i):
            for k in range(j):
                for l in range(k):
                    atleast_4_moves_p1 = z3.Or(atleast_4_moves_p1, 
                                z3.And(z3_move[i*3 + 1], z3_move[j*3 + 1], z3_move[k*3 + 1], z3_move[l*3 + 1]))
                    atleast_4_moves_p2 = z3.Or(atleast_4_moves_p2, 
                                z3.And(z3_move[i*3 + 2], z3_move[j*3 + 2], z3_move[k*3 + 2], z3_move[l*3 + 2]))

    # Both players have played exacly 2 or 3 moves
    solver.add(z3.Or(z3.And(    atleast_2_moves_p1, z3.Not(atleast_3_moves_p1), 
                                atleast_2_moves_p2, z3.Not(atleast_3_moves_p2)),
                     z3.And(    atleast_3_moves_p1, z3.Not(atleast_4_moves_p1), 
                                atleast_3_moves_p2, z3.Not(atleast_4_moves_p2))))

    # There is only one cell being marked
    atleast_2_cells = False
    for i in range(9):
       for j in range(i):
           atleast_2_cells = z3.Or(atleast_2_cells, z3.And(z3_move[27+i], z3_move[27+j]))
    solver.add(z3.Not(atleast_2_cells))
    solver.add(z3.Or(z3_move[27:]))

    # The cell being marked is not already marked
    solver.add(z3.And([z3.Implies(z3_move[27 + i], z3_move[i*3]) for i in range(9)]))

