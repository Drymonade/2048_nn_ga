# We will train nn with GA on 2048 game

# 2048 has 4x4 map and we can shift it to 4 directions - up, down, left, right

# We will have 16 inputs and 4 outputs - to up, down, left and right move

import random
import math
from itertools import combinations
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import numpy as np

N = 4  # size of one axis of board

nodes_by_layers = [4, 4]
# without input nodes which is always N*N

weights_precision = 3  # numbers of digits after comma

crossover_rate = 0.5  # how many weights will be taken from 1st parent

crossover_smoothness = 1  # if less than 1, we take weighted sum of genes

mutation_rate = 0.1  # how many weights will be changed randomly

mutation_count_rate = 0.5  # how many nns of epoch we mutate

max_selection_count = 50  # how many best nn will be chosen for selection

epochs = 100

directions_str = {
    0: 'up',
    1: 'down',
    2: 'left',
    3: 'right'
}

move_choosing_strategy = 'soft2'
# hard for choosing only max value of nn output,
# soft for trying max then 2nd max etc.
# soft2 for trying max then 2nd max and then nothing
# also there is soft3


def nn_length():
    # calculates total length of nn weights array

    buf_nodes = [N*N] + nodes_by_layers

    buf = zip(
        buf_nodes[:-1],
        buf_nodes[1:]
    )

    return sum([i*j for i, j in buf])


def get_random_nn():
    # generates random nn

    result = []

    for _ in range(nn_length()):
        result.append(
            random.randint(
                -10**weights_precision,
                10**weights_precision
            )/10**weights_precision
        )

    return result


def relu(score):
    # calculate activation

    return max(0, score)


def board_print(board):
    # prints board

    for row in board:
        print(row)
    print('_'*N*2)


def calc_left_board_move(board):
    # calcs board's left move
    # other cases will be computed through this one

    result_board = board

    for row_index, row in enumerate(board):
        if set(row) != {0}:
            buf_row = strafe_array_zeros_left(row)
            for cell_index in range(N-1):
                buf_set = set(buf_row[cell_index+1:])
                if buf_set not in [{0}, set()]:
                    buf_row = buf_row[:cell_index+1] + strafe_array_zeros_left(
                        buf_row[cell_index+1:]
                    )
                if buf_row[cell_index] == buf_row[cell_index+1]:
                    buf_row[cell_index] *= 2
                    buf_row[cell_index+1:] = buf_row[cell_index+2:] + [0]
            result_board[row_index] = buf_row
    return result_board


def next_board(board, direction):
    # board is NxN array
    # direction is integer, where 0 is up, 1 is down,
    # 2 is left, 3 is right
    # next_board returns next board configuration or None if there are
    # no moves

    # up
    if direction == 0:
        buf_board = transpose(board)
        buf_board = calc_left_board_move(buf_board)
        return transpose(buf_board)

    # down
    if direction == 1:
        buf_board = transpose_mirror(board)
        buf_board = calc_left_board_move(buf_board)
        return transpose_mirror(buf_board)

    # left
    if direction == 2:
        return calc_left_board_move(board)

    # right
    if direction == 3:
        buf_board = [row[::-1] for row in board]
        buf_board = calc_left_board_move(buf_board)
        return [row[::-1] for row in buf_board]


def transpose(board):
    # transpose board by main diag
    return [list(row) for row in zip(*board)]


def transpose_mirror(board):
    # ranspose board by another diag
    buf_board = board[::-1]
    buf_board = [row[::-1] for row in buf_board]
    return [list(row) for row in zip(*buf_board)]


def is_it_zero_on_board(board):
    # checks is there any zeros on the board

    for row in board:
        for cell in row:
            if cell == 0:
                return True
    return False


def is_there_moves(board):
    # checks is there any moves

    if not board:
        return False

    if is_it_zero_on_board(board):
        return True

    board_up = next_board(board, 0)
    board_down = next_board(board, 1)
    board_left = next_board(board, 2)
    board_right = next_board(board, 3)

    if board_up != board or board_down != board or \
       board_left != board or board_right != board:
        return True

    return False


def strafe_array_zeros_left(array):
    # helper function to calc [0, 0, 2, 0] -> [2, 0, 0, 0] for example

    result_array = array

    while result_array[0] == 0:
        result_array = result_array[1:] + [0]

    return result_array


def fill_random_2(board):
    # fill random empty cell with 2

    zero_indices = []

    buf_board = board

    for row_index, row in enumerate(board):
        for column_index, value in enumerate(row):
            if value == 0:
                zero_indices.append((row_index, column_index))

    if not zero_indices:
        return None

    target = random.choice(zero_indices)

    buf_board[target[0]][target[1]] = 2

    return buf_board


def generate_start_board():
    # produce start board with two 2s

    result = []

    for _ in range(N):
        buf_row = []
        for _ in range(N):
            buf_row.append(0)
        result.append(buf_row)

    result = fill_random_2(result)
    result = fill_random_2(result)

    return result


def play():
    # play the game!

    board = generate_start_board()

    moves = 0

    while is_there_moves(board):
        if is_there_2048(board):
            print('Hooray!')
            break
        board_print(board)
        print('moves =', moves)
        direction = int(input())
        board = next_board(board, direction)
        moves += 1
        board = fill_random_2(board)


def board_to_vec(board):
    # perform 2D to 1D array transform

    result = []
    for row in board:
        result += row
    return result


def scalar_mul(vec1, vec2):
    # perform scalar multiplication of two vectors

    return sum([item[0]*item[1] for item in zip(vec1, vec2)])


def board_to_log2(board):
    # convert non-zero values of board to log2(values) / 10

    result = []

    for row in board:
        buf_row = []
        for value in row:
            if value == 0:
                buf_row.append(value)
            else:
                buf_row.append(int(math.log2(value)))
        result.append(buf_row)

    return result


def compute_nn_one_step(board, nn):
    # compute result of nn on given board

    _board = board_to_log2(board)

    weights_count_cum_sum = [0, N*N*nodes_by_layers[0]]

    for layer_index in range(len(nodes_by_layers)-1):
        weights_count_cum_sum.append(
            nodes_by_layers[layer_index]*nodes_by_layers[layer_index+1]
            + weights_count_cum_sum[-1]
        )

    buf = [board_to_vec(_board)]

    for layer_index, layer_len in enumerate(nodes_by_layers):
        len_buf = len(buf[layer_index])

        layer_buf = []
        for node in range(layer_len):
            layer_buf.append(
                relu(
                    scalar_mul(
                        nn[
                            node*len_buf +
                            weights_count_cum_sum[layer_index]:
                            (node + 1)*len_buf +
                            weights_count_cum_sum[layer_index]
                        ],
                        buf[layer_index]
                    )
                )
            )
        buf.append(layer_buf)

    return buf


def get_moves_from_nn(nn_output):
    # returns array of indices of max value, second max etc. from nn output

    last_layer = nn_output[-1]

    sorted_last_layer = sorted(last_layer, reverse=True)

    result = []

    for value in sorted_last_layer:
        result.append(last_layer.index(value))

    return result


def direction_to_one_hot(direction):
    result = [0, 0, 0, 0]
    result[direction] = 1
    return result


def record_results(boards_n_moves_array):
    f = open('nn_ga_results.txt', 'a')

    for result in boards_n_moves_array:
        line = ' '.join(map(lambda x: str(x), result))
        f.write(line + '\n')

    f.close()


def is_there_2048(board):
    # checks if there is 2048 on board

    if not board:
        return False

    for row in board:
        for value in row:
            if value == 2048:
                return True

    return False


def experiment_workflow():
    # main workflow of experiment

    nns = [get_random_nn() for _ in range(1275)]

    result_max = []
    result_avg = []

    for epoch in range(epochs):
        result = []

        for index, nn in enumerate(
            tqdm(nns, ncols=100, desc='Epoch #'+str(epoch+1))
        ):

            board = generate_start_board()

            moves = 0
            boards_n_moves = []

            while is_there_moves(board):
                if is_there_2048(board):
                    print('>>>>!!!!!!<<<<')
                    record_results(boards_n_moves)
                    break

                directions = get_moves_from_nn(
                    compute_nn_one_step(board, nn)
                )

                move_completed = False

                if move_choosing_strategy == 'soft':
                    for direction in directions:
                        pre_board = next_board(board, direction)
                        if pre_board == board:
                            continue
                        else:
                            move_completed = True
                            boards_n_moves.append(
                                board_to_vec(board_to_log2(board)) +
                                direction_to_one_hot(direction)
                            )
                            break
                    if not move_completed:
                        break
                elif move_choosing_strategy == 'hard':
                    pre_board = next_board(board, directions[0])
                    if pre_board == board:
                        break
                elif move_choosing_strategy == 'soft2':
                    pre_board = next_board(board, directions[0])
                    if pre_board == board:
                        pre_board = next_board(board, directions[1])
                        if pre_board == board:
                            break
                elif move_choosing_strategy == 'soft3':
                    pre_board = next_board(board, directions[0])
                    if pre_board == board:
                        pre_board = next_board(board, directions[1])
                        if pre_board == board:
                            pre_board = next_board(board, directions[2])
                            if pre_board == board:
                                break

                board = pre_board
                moves += 1
                board = fill_random_2(board)

            result.append((nn, moves))

        all_moves = list(map(lambda x: x[1], result))

        avg_moves = round(sum(all_moves)/len(all_moves), 2)

        max_moves = max(all_moves)

        print('AVG moves:', avg_moves)
        print('MAX moves:', max_moves)

        result_avg.append(avg_moves)
        result_max.append(max_moves)

        result = sorted(result, key=lambda x: x[1], reverse=True)

        result_first_part = result[:max_selection_count]

        result_first_part = list(
            map(lambda x: x[0], result_first_part)
        )

        nns = [
            crossover(t[0], t[1])
            for t in combinations(
                result_first_part, 2
            )
        ]

        mutation_indices = sorted(random.sample(
            range(len(nns)),
            k=int(len(nns)*mutation_count_rate)
        ))

        for index in mutation_indices:
            nns[index] = mutation(nns[index])

        nns = result_first_part + nns

        nns += [get_random_nn() for _ in range(25)]

    print('AVGs')
    print(result_avg)
    print('MAXs')
    print(result_max)

    print('MAX of MAXs is', max(result_max))
    print('MAX of AVGs is', max(result_avg))


def nn_train_exp():

    f = open('nn_ga_results.txt', 'r')

    data = []

    for line in f.readlines():
        data.append(list(map(int, line.split(' '))))

    print(len(data))

    data = np.array(data)

    x_train = data[:8000000, :16]
    y_train = data[:8000000, 16:]

    x_test = data[8000000:, :16]
    y_test = data[8000000:, 16:]

    model = Sequential()
    model.add(Dense(12, activation='relu', input_dim=16))
    model.add(Dropout(0.15))
    model.add(Dense(4, activation='softmax'))

    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=5,
              batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)
    print(score)

    # start trained nn play

    board = generate_start_board()

    moves = 0

    while is_there_moves(board):
        if is_there_2048(board):
            print('>>>>!!!!!!<<<<')
            break

        board_print(board)

        prediction = model.predict(
            np.array([
                board_to_vec(
                    board_to_log2(board)
                )
            ])
        )

        direction = prediction.argmax()

        pre_board = next_board(board, direction)

        if pre_board == board:
            break

        board = pre_board
        print(prediction, directions_str[direction])
        moves += 1
        board = fill_random_2(board)

    print('Moves:', moves)


def crossover(nn1, nn2):
    # performs crossover of two nns and mutation of a child

    length = nn_length()

    # crossover

    crossover_indices = sorted(random.sample(
        range(length),
        k=int(length*crossover_rate)
    ))

    crossover_result = []

    for index, gene in enumerate(zip(nn1, nn2)):
        if index in crossover_indices:
            crossover_result.append(
                crossover_smoothness*gene[0] +
                (1-crossover_smoothness)*gene[1]
            )
        else:
            crossover_result.append(
                crossover_smoothness*gene[1] +
                (1-crossover_smoothness)*gene[0]
            )

    return crossover_result


def mutation(nn):
    # mutation

    length = nn_length()

    result_nn = nn

    mutation_indices = sorted(random.sample(
        range(length),
        k=int(length*mutation_rate)
    ))

    for index in mutation_indices:
        result_nn[index] = random.randint(
            -10**weights_precision,
            10**weights_precision
        )/10**weights_precision

    return result_nn


if __name__ == '__main__':
    experiment_workflow()
    # nn_train_exp()
