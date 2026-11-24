import chess
from typing import Optional, List, Dict, Tuple

class ChessLogic:
    """
    A class encapsulating chess evaluation and move search logic using material scoring,
    piece-square tables, and a minimax search with alpha-beta pruning.

    Attributes:
        PIECE_VALUES (Dict[int, int]): Base material values for each type of piece (pawn = 10).
        PIECE_SQUARE_TABLES (Dict[int, List[List[float]]]): Positional score tables for pieces based on square location.
        search_depth (int): Maximum depth to explore in the minimax tree.
    """
    # Base piece values (pawn = 10, i.e. tenths of a pawn)
    PIECE_VALUES: Dict[int, int] = {
        chess.PAWN:   10,
        chess.KNIGHT: 30,
        chess.BISHOP: 30,
        chess.ROOK:   50,
        chess.QUEEN:  90,
        chess.KING:   0,
    }

    # Piece-square tables: nested lists for 8x8 values.
    PIECE_SQUARE_TABLES: Dict[int, List[List[float]]] = {
        chess.KING: [
            [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
            [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
            [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
            [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
            [-2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0],
            [-1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0],
            [ 2.0,  2.0,  0.0,  0.0,  0.0,  0.0,  2.0,  2.0],
            [ 2.0,  3.0,  1.0,  0.0,  0.0,  1.0,  3.0,  2.0]
        ],
        chess.QUEEN: [
            [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
            [-1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
            [-1.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0],
            [-0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
            [ 0.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
            [-1.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0],
            [-1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
            [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]
        ],
        chess.ROOK: [
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [ 0.5,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.5],
            [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
            [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
            [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
            [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
            [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
            [ 0.0,  0.0,  0.0,  0.5,  0.5,  0.0,  0.0,  0.0]
        ],
        chess.BISHOP: [
            [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
            [-1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
            [-1.0,  0.0,  0.5,  1.0,  1.0,  0.5,  0.0, -1.0],
            [-1.0,  0.5,  0.5,  1.0,  1.0,  0.5,  0.5, -1.0],
            [-1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  0.0, -1.0],
            [-1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0],
            [-1.0,  0.5,  0.0,  0.0,  0.0,  0.0,  0.5, -1.0],
            [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0]
        ],
        chess.KNIGHT: [
            [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
            [-4.0, -2.0,  0.0,  0.0,  0.0,  0.0, -2.0, -4.0],
            [-3.0,  0.0,  1.0,  1.5,  1.5,  1.0,  0.0, -3.0],
            [-3.0,  0.5,  1.5,  1.7,  1.7,  1.5,  0.5, -3.0],
            [-3.0,  0.0,  1.5,  1.7,  1.7,  1.5,  0.0, -3.0],
            [-3.0,  0.5,  1.0,  1.5,  1.5,  1.0,  0.5, -3.0],
            [-2.0, -2.0,  0.0,  0.5,  0.5,  0.0, -2.0, -2.0],
            [-3.0, -0.5,  0.0, -1.0, -1.0,  0.0, -0.5, -3.0]
        ],
        chess.PAWN: [
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [ 5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0],
            [ 1.0,  1.0,  2.0,  3.0,  3.0,  2.0,  1.0,  1.0],
            [ 0.5,  0.5,  1.0,  2.5,  2.5,  1.0,  0.5,  0.5],
            [ 0.0,  0.0,  2.0,  2.0,  2.0,  2.0,  0.0,  0.0],
            [ 0.5, -0.5, -1.0,  2.0,  2.0, -1.0, -0.5,  0.5],
            [ 0.5,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.5],
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]
        ],
    }

    # Score for a checkmate. Finite (rather than infinity) so that a faster mate
    # can be scored higher than a slower one: see minimax().
    MATE_SCORE: int = 100000

    def __init__(self, depth: int = 3):
        """
        Initializes the ChessLogic object.

        Args:
            depth (int): Search depth for the minimax algorithm.
        """
        self.search_depth = depth
        # Transposition table: zobrist key -> (depth, score)
        # self.transposition: Dict[int, Tuple[int, float]] = {}

    def piece_value(self, piece: chess.Piece) -> int:
        """
        Returns the material value of a piece.

        Args:
            piece (chess.Piece): The chess piece to evaluate.

        Returns:
            int: Material value (pawn = 10).
        """
        return self.PIECE_VALUES.get(piece.piece_type, 0)

    def pos_value(self, piece_type: int, pos: int, color: bool) -> float:
        """
        Returns the positional value of a piece based on its location.

        The tables above are written the way a board is printed from White's
        side: row 0 is rank 8 (the far side) and row 7 is rank 1. python-chess
        numbers squares from a1 = 0 up to h8 = 63, so rank 1 is pos // 8 == 0.
        White therefore reads row 7 - rank, and Black (the mirror image) reads
        row rank.

        Args:
            piece_type (int): Type of the piece (e.g., chess.KNIGHT).
            pos (int): Square index (0-63).
            color (bool): True for white, False for black.

        Returns:
            float: Positional score from the piece-square table, from the point
            of view of the piece's own side.
        """
        table = self.PIECE_SQUARE_TABLES[piece_type]
        rank = chess.square_rank(pos)
        row = 7 - rank if color == chess.WHITE else rank
        col = chess.square_file(pos)
        return table[row][col]

    def evaluate_board(self, board: chess.Board) -> float:
        """
        Evaluates the board based on material and positional scores.

        Args:
            board (chess.Board): The current board state.

        Returns:
            float: Evaluation score. Positive if white is better, negative if black is better.
        """
        val = 0.0
        for sq, piece in board.piece_map().items():
            score = self.piece_value(piece) + self.pos_value(piece.piece_type, sq, piece.color)
            # Both material and position count for the piece's own side.
            val += score if piece.color == chess.WHITE else -score
        return val
    
    def capture_score(self, board, move):
        """
        Returns a score for a capture move using MVV-LVA heuristic.

        Args:
            board (chess.Board): The current board.
            move (chess.Move): The move being evaluated.

        Returns:
            int: Capture score. Higher values are better captures.
        """
        if board.is_en_passant(move):
            victim = chess.PAWN
        elif board.piece_at(move.to_square) is None:
            return 0
        else:
            victim = board.piece_at(move.to_square).piece_type
        attacker = board.piece_at(move.from_square).piece_type
        return self.PIECE_VALUES[victim]*100 - self.PIECE_VALUES[attacker]

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool) -> Tuple[float, Optional[chess.Move]]:
        """
        Recursive minimax search with alpha-beta pruning.

        Args:
            board (chess.Board): Current board state.
            depth (int): Remaining depth to search.
            alpha (float): Alpha value for pruning.
            beta (float): Beta value for pruning.
            maximizing_player (bool): True if the current player is maximizing.

        Returns:
            Tuple[float, Optional[chess.Move]]: Best evaluation score and corresponding move.
        """
        # Transposition lookup using private key
        # key = board._transposition_key()
        # entry = self.transposition.get(key)
        # if entry and entry[0] >= depth:
        #     return entry[1]
        
        best_move = None
        # Distance from the root, used to prefer quicker mates.
        ply = self.search_depth - depth

        if board.is_game_over():
            if board.is_checkmate():
                # The side to move has been mated. Mates closer to the root get
                # a larger magnitude, so the bot plays the fastest mate and, when
                # losing, delays mate as long as possible.
                mate = self.MATE_SCORE - ply
                score = -mate if maximizing_player else mate
            else:
                score = 0
            return score, None
        elif depth == 0:
            score = self.evaluate_board(board)
            return score, None
        else:
            moves = list(board.legal_moves)
            # MVV-LVA (Most Valuable Victim – Least Valuable Aggressor) based sorting
            moves.sort(key=lambda m: (board.is_capture(m), self.capture_score(board, m)), reverse=True)
            # Always have a legal move to return, even if every line loses.
            best_move = moves[0]
            if maximizing_player:
                score = -float('inf')
                for mv in moves:
                    board.push(mv)
                    val, _ = self.minimax(board, depth-1, alpha, beta, False)
                    board.pop()
                    if val > score:
                        score = val
                        best_move = mv
                    alpha = max(alpha, val)
                    if beta <= alpha:
                        break
            else:
                score = float('inf')
                for mv in moves:
                    board.push(mv)
                    val, _ = self.minimax(board, depth-1, alpha, beta, True)
                    board.pop()
                    if val < score:
                        score = val
                        best_move = mv
                    beta = min(beta, val)
                    if beta <= alpha:
                        break
        # Store in transposition table
        # self.transposition[key] = (depth, score)
        return score, best_move

    def get_move(self, board: chess.Board, turn : bool) -> Optional[chess.Move]:
        """
        Returns the best move for the current player using minimax search.

        Args:
            board (chess.Board): Current board state.
            turn (bool): True if it's white's turn, False for black.

        Returns:
            Optional[chess.Move]: The best move found, or None if no legal move exists.
        """
        # self.transposition.clear()
        _, best_move =  self.minimax(board, self.search_depth, -float('inf'), float('inf'), turn)
        return best_move