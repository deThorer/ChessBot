import pygame
import chess
import time
from evaluation import ChessLogic
from pathlib import Path
from constants import *
from renderer import Renderer, Phase

def mouse_to_square(pos):
    """
    Convert a pixel (x, y) position into a 0–63 board square index.
    
    Arguments:
        pos: Tuple[int, int] — (x, y) pixel coordinates from a mouse event.
    Returns:
        int: The corresponding square index (0 bottom-left through 63 top-right).
    """
    x, y = pos
    col = x // SQUARE_SIZE
    row = y // SQUARE_SIZE
    return (7 - row) * 8 + col

def ask_promotion():
    """
    Block until the user chooses a promotion piece or clicks elsewhere.
    
    Listens for:
      - Q → queen
      - R → rook
      - B → bishop
      - N or K → knight
      - Any mouse click → cancel promotion (return None)
    
    Returns:
        chess.PieceType or None: The chosen promotion piece, or None if canceled.
    """
    pygame.event.clear()  # discard any existing events so wait() truly blocks
    while True:
        ev = pygame.event.wait()
        if ev.type == pygame.MOUSEBUTTONDOWN:
            # Any click outside the keys cancels promotion
            return None
        if ev.type == pygame.KEYDOWN:
            return {
                pygame.K_q: chess.QUEEN,
                pygame.K_r: chess.ROOK,
                pygame.K_b: chess.BISHOP,
                pygame.K_n: chess.KNIGHT,
                pygame.K_k: chess.KNIGHT
            }.get(ev.key, None)

def choose_color(screen, renderer):
    """
    Display a prompt asking the user to pick White or Black.
    
    Arguments:
        screen: pygame.Surface — the main display surface.
        renderer: Renderer — used to draw the text prompt.
    
    Returns:
        (bool, bool): (running, bot_color)
            - running=False if the user closed the window.
            - bot_color: the colour the bot plays, in python-chess terms
              (chess.WHITE == True, chess.BLACK == False).
    """
    screen.fill(BLACK)
    renderer.show_text("Press W for White or B for Black")
    pygame.display.flip()
    while True:
        ev = pygame.event.wait()
        if ev.type == pygame.QUIT:
            return False, False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_w:
                return True, chess.BLACK  # user chose White, so the bot plays Black
            if ev.key == pygame.K_b:
                return True, chess.WHITE  # user chose Black, so the bot plays White

def check_for_gameover(board: chess.Board, screen: pygame.Surface, renderer: Renderer, bot_color: bool) -> bool:
    """
    Check board for a game‐over condition, display the appropriate message,
    pause briefly, and return False to signal that the main loop should exit.

    Arguments:
        board      : The current chess.Board instance.
        screen     : The Pygame display surface.
        renderer   : Renderer instance used to draw text.
        bot_color  : The colour the bot plays (chess.WHITE == True, chess.BLACK == False).

    Returns:
        bool: False if the game is over (after showing the message), True otherwise.
    """
    if not board.is_game_over():
        return True

    # Small delay before showing the final result
    time.sleep(0.5)

    # Black out the screen
    screen.fill((0, 0, 0))

    # Determine the appropriate endgame message
    if board.is_stalemate():
        msg = 'Draw by stalemate'
    elif board.is_insufficient_material():
        msg = 'Draw: insufficient material'
    elif board.is_seventyfive_moves():
        msg = 'Draw: 75-move rule'
    elif board.is_fivefold_repetition():
        msg = 'Draw: fivefold repetition'
    elif board.is_checkmate():
        # The side to move just got mated
        winner = "You Lost" if board.turn != bot_color else "You Won"
        msg = f"Checkmate! {winner}"
    else:
        msg = "Game Over"

    # Render the message to the screen
    renderer.show_text(msg)
    pygame.display.flip()
    time.sleep(5)
    return False

def make_move(board, move, renderer, redo_stack, clear_redo=True):
    """
    Animate a move and apply it to the board.

    This function:
    - Animates the moving piece sliding from source to destination square.
    - Pushes the move onto the board's move stack.
    - Clears the redo stack since a new move invalidates redo history.

    Args:
        board (chess.Board): The current game board.
        move (chess.Move): The move to be made.
        renderer (Renderer): Renderer object used to handle drawing and animations.
        redo_stack (list): Stack used to store undone moves for redo functionality.
        clear_redo (bool): True for a fresh move (which invalidates redo history),
            False when the move is being replayed by redo.
    """
    from_sq, to_sq = move.from_square, move.to_square
    piece = board.piece_at(from_sq)
    piece_img = renderer.images[piece.symbol()]
    renderer.animate_move(board, piece_img, from_sq, to_sq)
    board.push(move)
    if clear_redo:
        redo_stack.clear()  # New move invalidates redo history

def undo(board, renderer, redo_stack):
    """
    Undo the last two plies (bot and human), animating each move backward.

    This function:
    - Pops up to two moves from the board's move stack (most recent first).
    - Animates each piece returning to its original square.
    - Stores the undone moves in the redo stack for possible redoing later.

    Args:
        board (chess.Board): The current game board.
        renderer (Renderer): Renderer object used to handle drawing and animations.
        redo_stack (list): Stack used to store undone moves for redo functionality.
    """
    for _ in range(2):
        if not board.move_stack:
            break
        last_move = board.pop()
        # After pop() the piece is back on from_square; slide it there from
        # to_square, hiding the copy already drawn on from_square.
        piece = board.piece_at(last_move.from_square)
        if piece:
            piece_img = renderer.images[piece.symbol()]
            renderer.animate_move(board, piece_img, last_move.to_square, last_move.from_square,
                                  hide_sq=last_move.from_square)
        redo_stack.append(last_move)
            
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Chess Bot')
    clock = pygame.time.Clock()

    # Load images and renderer
    img_dir = Path(__file__).parent / 'img'
    renderer = Renderer(screen, img_dir)

    # Initialize the chess logic (bot) with a given search depth
    chesslogic = ChessLogic(depth=3)
    board = chess.Board()

    running, bot_color = choose_color(screen, renderer)
    if not running:
        pygame.quit()
        return

    # Different Phases of Game
    phase = Phase.WAITING_SOURCE
    # A separate stack to support redo after undo
    redo_stack = []
    # Currently selected square (None if no selection)
    selected = None
    # Legal moves from 'selected' square
    legal_moves = []

    while running:
        clock.tick(FPS)

        # 1) If it's the bot's turn, let it calculate and make its move
        if board.turn == bot_color:
            bot_move = chesslogic.get_move(board, bot_color)
            make_move(board, bot_move, renderer, redo_stack)
            phase = Phase.WAITING_SOURCE

        # 2) Poll events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                return

            # --- Undo with Left Arrow ---
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_LEFT:
                undo(board, renderer, redo_stack)
                selected, legal_moves = None, []
                continue

            # --- Redo with Right Arrow ---
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_RIGHT:
                # Push back up to two plies from the redo stack
                for _ in range(2):
                    if redo_stack:
                        next_move = redo_stack.pop()
                        make_move(board, next_move, renderer, redo_stack, clear_redo=False)
                selected, legal_moves = None, []
                continue

            # 3) Handle user clicking to make a move
            if board.turn != bot_color:
                # Phase 1: selecting a source square
                if phase == Phase.WAITING_SOURCE and ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    sq = mouse_to_square(ev.pos)
                    piece = board.piece_at(sq)
                    if piece and piece.color != bot_color:
                        selected = sq
                        # Gather all legal moves from this square
                        legal_moves = [m for m in board.legal_moves if m.from_square == sq]
                        phase = Phase.WAITING_DEST

                # Phase 2: selecting the destination square
                elif phase == Phase.WAITING_DEST and ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    dst = mouse_to_square(ev.pos)
                    clicked = board.piece_at(dst)

                    # Clicking another of your own pieces switches the selection
                    if clicked and clicked.color != bot_color and dst != selected:
                        selected = dst
                        legal_moves = [m for m in board.legal_moves if m.from_square == dst]
                        continue

                    candidate = chess.Move(from_square=selected, to_square=dst)

                    # Pawn-promotion check: covers straight pushes and diagonal
                    # captures onto the last rank.
                    if any(m.to_square == dst and m.promotion for m in legal_moves):
                        promotion_piece = ask_promotion()
                        if promotion_piece:
                            candidate = chess.Move(selected, dst, promotion=promotion_piece)

                    # If the constructed move is legal, animate + push
                    if candidate in board.legal_moves:
                        make_move(board, candidate, renderer, redo_stack)

                    # Reset selection state regardless of legality
                    phase = Phase.WAITING_SOURCE
                    selected, legal_moves = None, []

        # 4) Render current position
        renderer.draw_board()
        renderer.draw_pieces(board)
        # Highlight legal destinations for the selected square
        if selected is not None:
            renderer.highlight({'moves': legal_moves})

        pygame.display.flip()
        running = check_for_gameover(board, screen, renderer, bot_color)
       
    pygame.quit()

if __name__ == '__main__':
    main()