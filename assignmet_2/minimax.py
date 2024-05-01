# %%
import random
import time
import copy
inf = 1e5

# %%


class ConnectFour:
    def __init__(self, rows, columns, winning_points=3):
        self.rows = rows
        self.columns = columns
        self.board = [[0 for i in range(columns)] for j in range(rows)]
        self.number_of_states = 0
        self.winning_points = winning_points
        self.current_player = 1

    def print_board(self):
        print('Printing board')
        for i in range(self.rows):
            for j in range(self.columns):
                print("|", end='')
                if self.board[i][j] == 0:
                    print(" ", end='')
                else:
                    print(self.board[i][j], end='')
            print("|", end='')
            print('')
        for i in range(self.columns):
            print('--', end='')
        print('-')

    # Drop a piece in the column
    def make_move(self, column):
        if (self.board[0][column] != 0):
            # print("Columns is full")
            return -1
        i = 0
        while i < self.rows - 1 and self.board[i + 1][column] == 0:
            i = i + 1
        self.board[i][column] = self.current_player
        self.current_player = 3 - self.current_player
        return i, column

    def get_winner(self):

        # check horizontal
        for i in range(self.rows):
            for j in range(self.columns):
                ti = i
                count = 0
                while ti < self.rows and self.board[ti][j] == self.current_player:
                    count += 1
                    ti += 1
                if count == self.winning_points:
                    return self.current_player
        for i in range(self.rows):
            for j in range(self.columns):
                tj = j
                count = 0
                while tj < self.columns and self.board[i][tj] == self.current_player:
                    count += 1
                    tj += 1
                if count == self.winning_points:
                    return self.current_player
        for i in range(self.rows):
            for j in range(self.columns):
                ti = i
                tj = j
                count = 0
                while ti < self.rows and tj < self.columns and self.board[ti][tj] == self.current_player:
                    count += 1
                    ti += 1
                    tj += 1
                if count == self.winning_points:
                    return self.current_player
        for i in range(self.rows):
            for j in range(self.columns):
                ti = i
                tj = j
                count = 0
                while ti < self.rows and tj >= 0 and self.board[ti][tj] == self.current_player:
                    count += 1
                    ti += 1
                    tj -= 1
                if count == self.winning_points:
                    return self.current_player

        return 0

    def is_full(self):
        for i in range(self.columns):
            if self.board[0][i] == 0:
                return False
        return True

    def possible_moves(self):
        return [col for col in range(self.columns) if self.board[0][col] == 0]

    def next_find_move(self, column):
        for i in range(self.rows - 1, -1, -1):
            if self.board[i][column] == 0:
                return i
        return -1

    def play(self):
        turn = 1
        player = 1
        while True:
            if len(self.available_columns) == 0:
                print("Game over")
                break
            column = random.choice(self.available_columns)
            i, j = self.drop_piece(column, player)
            print('turn: ' + str(turn), end=' ')
            print('player: ' + str(player), end=' ')
            print('drop piece at: ', i, j)
            turn += 1
            self.print_board()
            if self.check_winner(player) != 0:
                print("Player ", player, " wins")
                break
            player = 3 - player

    def simulate(self, moves):
        for move in moves:
            if (move[0] == 1):
                print("Player 1 drops at ", move[1])
            else:
                print("Player 2 drops at ", move[1])
            self.board[move[1]][move[2]] = move[0]
            self.print_board()

    def clone(self):
        game = ConnectFour(self.rows, self.columns, self.winning_points)
        game.board = copy.deepcopy(self.board)
        game.current_player = self.current_player

        return game


# %%

def MiniMax(game, is_alpha):
    game.number_of_states += 1
    if game.get_winner() != 0:
        return game.get_winner(), []
    if game.is_full():
        return 0, []
    if is_alpha:
        best = -inf
        available_moves = []
        best_strategy = []
        available_moves = game.possible_moves()
        for col in available_moves:
            row = game.next_find_move(col)
            game.board[row][col] = 1
            curr, turn = MiniMax(game, 0)
            game.board[row][col] = 0
            if (curr > best):
                best = curr
                best_strategy = [(1, row, col)] + turn
        return (best, best_strategy)
    else:
        best = inf
        available_moves = []
        best_strategy = []
        available_moves = game.possible_moves()

        for col in available_moves:
            row = game.next_find_move(col)
            game.board[row][col] = 2
            curr, turn = MiniMax(game, 1)
            game.board[row][col] = 0
            if (curr < best):
                best = curr
                best_strategy = [(2, row, col)] + turn

        return (best, best_strategy)

# %%


def alpha_beta_prunning(game, is_alpha, alpha, beta):
    game.number_of_states += 1
    if game.get_winner() != 0:
        return game.get_winner(), []
    if game.is_full():
        return 0, []
    if is_alpha:
        best = -inf
        available_moves = []
        best_strategy = []
        available_moves = game.possible_moves()
        for col in available_moves:
            row = game.next_find_move(col)
            game.board[row][col] = 1
            curr, turn = alpha_beta_prunning(game, 0, alpha, beta)
            game.board[row][col] = 0
            if (curr > best):
                best = curr
                best_strategy = [(1, row, col)] + turn
            alpha = max(alpha, best)
            if (alpha >= beta):
                break
        return (best, best_strategy)
    else:
        best = inf
        available_moves = []
        best_strategy = []
        available_moves = game.possible_moves()
        for col in available_moves:
            row = game.next_find_move(col)
            game.board[row][col] = 2
            curr, turn = alpha_beta_prunning(game, 1, alpha, beta)
            game.board[row][col] = 0
            if (curr < best):
                best = curr
                best_strategy = [(2, row, col)] + turn
            beta = min(beta, best)
            if (alpha >= beta):
                break

        return (best, best_strategy)


# %%
game = ConnectFour(4, 4, winning_points=3)
start_time = time.time()
winner, turns = alpha_beta_prunning(game, 1, -inf, inf)
print('Minimax with alpha beta prunning')
game.simulate(turns)
if (winner):
    print('player', winner, 'wins')
else:
    print('Game is draw')

print(f'with {game.rows} by {game.columns} board and connecting {
      game.winning_points} points to win')
print('number of state simulate', game.number_of_states)
print('time taken', time.time() - start_time)


# %%
game = ConnectFour(4, 4, winning_points=3)
start_time = time.time()
winner, turns = MiniMax(game, 1)
print('Minimax without alpha beta prunning')
game.simulate(turns)
if (winner):
    print('player', winner, 'wins')
else:
    print('Game is draw')
print('number of state simulate', game.number_of_states)
print(f'with {game.rows} by {game.columns} board and connecting {
      game.winning_points} points to win')
print('number of state simulate', game.number_of_states)
print('time taken', time.time() - start_time)
