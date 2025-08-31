import random

def leaf_values(num_values):
    return [random.choice([-1, 1]) for _ in range(num_values)]

def alpha_beta(depth, is_sub_zero, alpha, beta, node_idx, outcomes):
    if depth == 0:
        return outcomes[node_idx]

    if is_sub_zero:
        max_score = float('-inf')
        for branch in range(2):
            score = alpha_beta(depth - 1, False, alpha, beta, node_idx * 2 + branch, outcomes)
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return max_score
    else:
        min_score = float('inf')
        for branch in range(2):
            score = alpha_beta(depth - 1, True, alpha, beta, node_idx * 2 + branch, outcomes)
            min_score = min(min_score, score)
            beta = min(beta, score)
            if beta <= alpha:
                break
        return min_score

def mortal_kombat_game(first_player):
    max_depth = 5
    total_leaves = 2 ** max_depth
    outcomes = leaf_values(total_leaves)
    current_player = first_player
    round_outcomes = []

    for round_number in range(1, 4):
        result = alpha_beta(max_depth, current_player, float('-inf'), float('inf'), 0, outcomes)
        if result == 1:
            round_outcomes.append("Sub-Zero")
        else:
            round_outcomes.append("Scorpion")
        current_player = 1 - current_player  # Alternating players

    scorpion_wins = round_outcomes.count("Scorpion")
    sub_zero_wins = round_outcomes.count("Sub-Zero")
    overall_winner = "Scorpion" if scorpion_wins > sub_zero_wins else "Sub-Zero"

    print(f"\nFinal Winner: {overall_winner}")
    print(f"Total Rounds Played: {len(round_outcomes)}")
    for idx, winner in enumerate(round_outcomes, 1):
        print(f"Winner of Round {idx}: {winner}")


def alpha_beta_minimax(depth, max_player, node_idx, scores, alpha, beta):
    if depth == 0:
        return scores[node_idx]

    if max_player:
        max_eval = float('-inf')
        for i in range(2):
            eval = alpha_beta_minimax(depth - 1, False, node_idx * 2 + i, scores, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(2):
            eval = alpha_beta_minimax(depth - 1, True, node_idx * 2 + i, scores, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def pacman_dark_magic(cost):
    scores = [3, 6, 2, 3, 7, 1, 2, 0]
    depth = 3
    root_value_no_magic = alpha_beta_minimax(depth, True, 0, scores, float('-inf'), float('inf'))
    left_subtree_value = max(scores[0:len(scores)//2]) - cost
    right_subtree_value = max(scores[len(scores)//2:]) - cost
    dark_magic_value = max(left_subtree_value, right_subtree_value)
    
    if dark_magic_value > root_value_no_magic:
        if left_subtree_value >= right_subtree_value:
            print(f"The new minimax value is {dark_magic_value}. Pacman goes left and uses dark magic.")
        else:
            print(f"The new minimax value is {dark_magic_value}. Pacman goes right and uses dark magic.")
    else:
        print(f"The new minimax value is {root_value_no_magic}. Pacman does not use dark magic.")


def explain_alpha_beta_concepts():
    print("\nAlpha-Beta Pruning Concepts:")
    print("1. Player Perspectives:")
    print("   The first player is always a maximizer as it wants to maximize its winning chances.")
    print("   The second player perspective depends on viewpoint:")
    print("   - Maximizer if we consider the second player's perspective")
    print("   - Minimizer if we consider the first player's perspective")
    
    print("\n2. Alpha-Beta in Stochastic Environments:")
    print("   Alpha-beta pruning is designed for deterministic environments.")
    print("   It works with fixed values and can safely prune branches.")
    print("   In stochastic environments (like dice games), randomness prevents")
    print("   guaranteed predictions, making traditional pruning unreliable.")


if __name__ == "__main__":
    print("Minimax with Alpha-Beta Pruning Demonstrations")
    
    print("\n=== Mortal Kombat Game ===")
    starting_player = int(input("Enter 0 for Scorpion, 1 for Sub-Zero: "))
    mortal_kombat_game(starting_player)
    
    print("\n=== Pacman Dark Magic Scenarios ===")
    print("Testing with cost = 2:")
    pacman_dark_magic(2)
    print("\nTesting with cost = 5:")
    pacman_dark_magic(5)
    
    explain_alpha_beta_concepts()
