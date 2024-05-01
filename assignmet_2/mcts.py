import random
import math
import copy
import time


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


class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.wins = 0
        self.visits = 0
        self.children = []
        self.available_moves = state.possible_moves()
        self.parent = parent
        self.move = move

    def select_child(self):
        best_child = None
        best_score = -math.inf
        for child in self.children:
            score = child.wins / child.visits + \
                math.sqrt(2*math.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expansion(self, move, state):
        child_node = Node(state, parent=self, move=move)
        child_node.parent = self
        self.available_moves.remove(move)
        self.children.append(child_node)
        return child_node


class MCTS:
    def __init__(self, game) -> None:
        self.root = Node(state=game.clone())

    def search(self, root, num_iterations):

        for i in range(num_iterations):
            node = root
            game = root.state.clone()
            # selection
            while node.children:
                node = node.select_child()
                if node is None:
                    break
                game.make_move(node.move)
            if node is None:
                continue
            # expansion
            if node.available_moves != []:
                move = random.choice(node.available_moves)
                k = game.make_move(move)
                if (k == -1):
                    continue
                node = node.expansion(move, node.state.clone())
            # simulation
            while node.state.is_full() == False:
                possible_moves = node.state.possible_moves()
                node.state.make_move(random.choice(possible_moves))

            winner = -1
            if game.get_winner():
                winner = 3 - game.current_player
            elif game.is_full():
                winner = 0

            # update and backpropagation
            while node is not None:
                node.visits += 1
                if (game.current_player == winner):
                    node.wins += 1
                node = node.parent

    def find_best_move(self, state, num_iterations):
        root = Node(state=state.clone())
        self.search(root, num_iterations)
        value = 0
        best_move = None
        for child in root.children:
            cur = child.wins / child.visits if child.visits > 0 else 0
            if cur >= value:
                value = cur
                best_move = child.move
        return best_move

    def apply_mcts(self, game):
        max_moves = game.rows * game.columns
        iter = 0
        while game.is_full() == False and iter < max_moves:
            best_move = self.find_best_move(game, num_iterations=100)
            print(f'for player {game.current_player} at {
                  iter} iteration, best move is {best_move}')
            game.make_move(best_move)
            game.print_board()
            iter += 1

            result = game.get_winner()
            if (result):
                print(f'player {3 - result} wins')
                game.print_board()
                return result
            if game.is_full():
                print('game is draw')
                return 0
        return -1


game = ConnectFour(7, 10, 7)

start_time = time.time()
mcts = MCTS(game)
mcts.apply_mcts(game)
end_time = time.time()
print(f'For a {game.rows}x{game.columns} board, with {
      game.winning_points} connected points to win')
print(f'Time taken: {end_time - start_time} seconds')
