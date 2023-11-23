EXPERIMENTS = {
    # start with hard one only one iteration! (deceive agent 3)
    "deceive-3 (pregame)":
        [10, 10,
         [[8, 9], [5, 9], [4, 9], [5, 8]],
         1,
         [[7, 2], [9, 9], [0, 9], [0, 0]],
         [5, 0],
         [5, 4],
         [600, 600]
         ],
    "split": [
        10, 10,
        [[7, 4], [7, 8]],
        0,
        [[2, 1], [8, 1]],
        [0, 6],
        [5, 1],
        [600, 600]],
    "clusters": [
        10, 10,
        [[7, 0], [8, 0], [9, 0], [2, 9], [1, 9], [0, 9]],
        5,
        [[3, 3], [5, 4], [5, 3], [5, 2], [3, 4], [3, 2]],
        [4, 3],
        [4, 6],
        [600, 600]],
    "split (flipped)": [
        10, 10,
        [[2, 1], [8, 1]],
        1,
        [[7, 4], [7, 8]],
        [0, 6],
        [5, 1],
        [600, 600]],
    "deceive-0 (flipped)": [
        10, 10,
        [[4, 1], [4, 0], [6, 0]],
        1,
        [[0, 9], [9, 9], [9, 8]],
        [4, 9],
        [3, 9],
        [600, 600]],
    "clusters (flipped)": [
        10, 10,
        [[7, 9], [8, 9], [9, 9], [2, 0], [1, 0], [0, 0]],
        2,
        [[3, 3], [5, 4], [5, 3], [5, 2], [3, 4], [3, 2]],
        [4, 3],
        [4, 6],
        [600, 600]],
    "deceive-2": [
        10, 10,
        [[4, 9], [5, 9], [8, 0], [5, 8]],
        1,
        [[9, 0], [0, 9], [0, 8], [9, 1]],
        [1, 1],
        [0, 1],
        [600, 600]],
    "deceive-1": [
        10, 10,
        [[1, 8], [1, 2], [2, 9], [6, 9]],
        2,
        [[8, 0], [6, 0], [8, 6], [6, 7]],
        [4, 2],
        [5, 2],
        [600, 600]],
    "deceive-0 (rotated)": [
        10, 10,
        [[8, 5], [9, 5], [9, 3]],
        1,
        [[9, 0], [0, 9], [1, 0]],
        [0, 5],
        [0, 6],
        [600, 600]],
    # "split (altered)": [
    #     10, 10,
    #     [[7, 4], [7, 8], [7, 6]],
    #     0,
    #     [[2, 1], [8, 1], [5, 6]],
    #     [0, 6],
    #     [5, 1],
    #     [600, 600]],
    # "deceive-0 (altered)": [
    #     10, 10,
    #     [[4, 8], [4, 9], [5, 9], [6, 9]],
    #     1,
    #     [[0, 0], [9, 0], [0, 1], [9, 1]],
    #     [4, 0],
    #     [3, 0],
    #     [600, 600]],
    "deceive-1 (rotated)": [
        10, 10,
        [[1, 1], [1, 7], [2, 0], [6, 0]],
        2,
        [[8, 9], [6, 9], [8, 3], [6, 2]],
        [4, 7],
        [5, 7],
        [600, 600]],
    "deceive-2 (rotated)": [
        10, 10,
        [[9, 5], [9, 4], [0, 1], [8, 4]],
        1,
        [[0, 0], [9, 9], [8, 9], [1, 0]],
        [1, 8],
        [1, 9],
        [600, 600]],
    # "clusters (altered)": [
    #     10, 10,
    #     [[0, 0], [8, 0], [9, 0], [2, 9], [1, 9], [0, 9]],
    #     5,
    #     [[6, 4], [6, 3], [4, 3], [6, 2], [4, 4], [4, 2]],
    #     [5, 3],
    #     [5, 5],
    #     [600, 600]],
    # "deceive-2 (altered)": [
    #     10, 10,
    #     [[4, 9], [5, 9], [8, 0], [5, 8], [9, 3]],
    #     1,
    #     [[9, 0], [1, 9], [0, 8], [9, 1], [0, 9]],
    #     [1, 1],
    #     [0, 1],
    #     [600, 600]],
    # "deceive-1 (altered)": [
    #     10, 10,
    #     [[1, 8], [1, 2], [2, 9], [6, 9], [0, 1]],
    #     0,
    #     [[9, 0], [6, 0], [8, 6], [6, 7], [9, 1]],
    #     [4, 2],
    #     [5, 2],
    #     [600, 600]],
    "deceive-3 (postgame)":
        [10, 10,
         [[8, 9], [5, 9], [4, 9], [5, 8]],
         1,
         [[7, 2], [9, 9], [0, 9], [0, 0]],
         [5, 0],
         [5, 4],
         [600, 600]
         ],
}
TUTORIALS = {
    # Station at every corner of square, worker in middle (for instructions)
    "tutorial": [
        10, 10,
        [[0, 0], [9, 0], [0, 9], [9, 9]],
        0,
        [[4, 6], [4, 5], [4, 3], [4, 4]],
        [5, 4],
        [0, 4],
        [600, 600],
        17, ],
    "deceive-0": [
        10, 10,
        [[4, 8], [4, 9], [6, 9]],
        1,
        [[0, 0], [9, 0], [9, 1]],
        [4, 0],
        [3, 0],
        [600, 600],
        27],
}