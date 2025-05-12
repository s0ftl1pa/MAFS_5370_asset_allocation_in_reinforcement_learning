import random
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs import EnvBase, TransformedEnv, Compose, RewardScaling, ToTensorImage
from torchrl.objectives import ClipPPOLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data import Composite, Unbounded, Categorical, Binary, Bounded


class SuperTicTacToe(EnvBase):
    def __init__(self, board_size=12, corner_size=4, device=None):
        super().__init__(device=device)
        self.board_size = board_size
        self.corner_size = corner_size
        self.max_steps = 200

        self.full_observation_spec = Composite(
            # board=Unbounded(
            #     shape=(5, board_size, board_size), #棋盘空格，棋手1，棋手2，有效空间，turn
            #     dtype=torch.float32,
            #     device=device
            # ),
            board=Unbounded(
                shape=(4, board_size, board_size),  # 棋盘空格，self，opponent，有效空间
                dtype=torch.float32,
                device=device
            ),
            mask=Categorical(
                n=2,
                shape=(board_size ** 2,),
                dtype=torch.bool,
                device=device
            ),
            turn=Categorical(
                n=2,
                shape=(1,),
                dtype=torch.int,
                device=device
            ),
            device=device
        )

        self.state_spec = self.observation_spec.clone()

        # self.reward_spec = Composite(
        #     {
        #         ("player0", "reward"): Unbounded(shape=(1,), device=device),
        #         ("player1", "reward"): Unbounded(shape=(1,), device=device),
        #     }
        # )
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32, device=device)

        self.action_spec = Categorical(
            n=board_size**2,
            shape=(1,),
            dtype=torch.int64,
            device=device
        )

        self.full_done_spec = Composite(
            done=Categorical(2, shape=(1,), dtype=torch.bool, device=device),
            device=device
        )
        self.full_done_spec["terminated"] = self.full_done_spec["done"].clone()
        self.full_done_spec["truncated"] = self.full_done_spec["done"].clone()


    # def to(self, device):
    #     super().to(device)
    #     if self.generator is not None:
    #         self.generator = self.generator.to(device)
    #     return self

    def _reset(self, reset_td: TensorDict) -> TensorDict:
        shape = reset_td.shape if reset_td is not None else ()
        state = self.state_spec.zeros(shape)

        board = torch.zeros((4, self.board_size, self.board_size), dtype=torch.float32)
        board[0].fill_(1.0)
        for x in range(self.board_size):
            for y in range(self.board_size):
                board[3, x, y] = 1.0 if self._is_valid_position(x, y) else 0.0

        mask = self._get_legal_actions_tensor(board)

        state["board"] = board
        state["mask"] = mask

        return state.update(self.full_done_spec.zero(shape))

    def _opponent_random_step(self, x, y, board, mask):
        valid_list = []
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if self._is_valid_position(nx, ny) and board[0, nx, ny] == 1:
                        valid_list.append((nx, ny))

        if not valid_list:
            action_list = torch.arange(self.board_size**2, device=self.device)
            action = random.choice(action_list[mask])
            x, y = divmod(action.item(), self.board_size)
            return x, y

        return random.choice(valid_list)



    def _step(self, state: TensorDict):
        board = state["board"].clone()
        turn = state["turn"].clone()
        action = state["action"]
        mask = state["mask"]

        x, y = divmod(action.item(), self.board_size)
        x, y = self._placement(x, y, board)
        mask[action.item()] = False

        board[0, x, y] = 0.0
        board[turn.item()+1, x, y] = 1.0

        win = self._check_win(x, y, turn.item(), board)
        done = win | ~mask.any(-1, keepdim=True)
        terminated = done.clone()

        reward = 0.0
        if win:
            reward = 1.0
        elif not done:
            turn = 1 - turn
            x, y = self._opponent_random_step(x, y, board, mask)
            action = x * self.board_size + y
            mask[action] = False

            board[0, x, y] = 0.0
            board[turn.item()+1, x, y] = 1.0

            win = self._check_win(x, y, turn.item(), board)
            done = win | ~mask.any(-1, keepdim=True)
            terminated = done.clone()

            if win:
                reward = -1.0

        state = TensorDict({
            "board": board,
            "mask": mask,
            "turn": 1-turn,
            "done": done,
            "terminated": terminated,
            "reward": reward,
        }, batch_size=state.batch_size)
        return state

    def _get_legal_actions_tensor(self, board):
        legal = torch.zeros(
            (self.board_size**2,),
            dtype=torch.bool
        )
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self._is_valid_position(x, y) and board[0, x, y] == 1:
                    legal[x * self.board_size + y] = True
        return legal

    def _is_valid_position(self, x, y):
        if (x < self.corner_size and y < self.corner_size) or \
                (x < self.corner_size and y >= self.board_size - self.corner_size) or \
                (x >= self.board_size - self.corner_size and y < self.corner_size) or \
                (x >= self.board_size - self.corner_size and y >= self.board_size - self.corner_size):
            return False
        return True

    def _placement(self, x, y, board):
        if torch.rand(1).item() < 0.5:
            return x, y

        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if self._is_valid_position(nx, ny) and board[0, nx, ny] == 1:
                        neighbors.append((nx, ny))

        return random.choice(neighbors) if neighbors else (x, y)

    def _check_win(self, x, y, turn, board):
        turn = turn + 1
        # rows
        for i in range(self.board_size-3):
            for j in range(self.board_size):
                if all(board[turn][i+k][j] == 1 for k in range(4)):
                    return True

        # col
        for i in range(self.board_size):
            for j in range(self.board_size-3):
                if all(board[turn][i][j+k] == 1 for k in range(4)):
                    return True

        # diagonals
        for i in range(self.board_size-4):
            for j in range(self.board_size-4):
                if all(board[turn][i+k][j+k] == 1 for k in range(5)):
                    return True

        for i in range(self.board_size-4):
            for j in range(4, self.board_size):
                if all(board[turn][i+k][j-k] == 1 for k in range(5)):
                    return True
        return False

    def render(self, td:TensorDict):
        """可视化渲染（保持原逻辑）"""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        valid_mask_np = self.chessboard.cpu().numpy()
        valid_mask = np.zeros_like(valid_mask_np, dtype=bool)
        for x in range(self.board_size):
            for y in range(self.board_size):
                valid_mask[x, y] = self._is_valid_position(x, y)

        print(f"Step: {self.step_count.item()}  Player: {self.current_player.item() + 1}")
        for x in range(self.board_size):
            row = []
            for y in range(self.board_size):
                if valid_mask[x, y]:
                    row.append(symbols[self.chessboard[x, y].item()])
                else:
                    row.append(' ')
            print(' '.join(row))
        print()

    def _set_seed(self, seed: int=None):
        if seed is None:
            seed = torch.seed() % 2**32
        self.generator.manual_seed(seed)
        return seed







