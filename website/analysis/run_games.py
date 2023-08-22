import copy
import time

from website.website import EXP_CONDS, GUI, FetcherYotamPolicy

from scenarios import SCENARIOS


def do_execute(gui, action, other_agent_move, inferred_goals, sleep):
    gui._move_agent(other_agent_move)
    if gui.arrived:
        action = 5
        time.sleep(sleep)  # yotam added
    else:
        action = do_event(gui, action)
        time.sleep(sleep)
    # Got input, return action, worker_pos, and fetcher_pos
    gui.steps += 1
    gui.on_render(inferred_goals)
    return action, gui.user, gui.robot


def do_event(gui, a):
    # Experiment screen
    if not gui.pause_screen:
        gui.prev_user[0] = gui.user[0]
        gui.prev_user[1] = gui.user[1]

        if a == "LEFT":
            if (gui.user[0] - 1) >= 0:
                gui.user[0] -= 1
            return 1
        elif a == "RIGHT":
            if (gui.user[0] + 1) < gui.num_cols:
                gui.user[0] += 1
            return 0
        elif a == "DOWN":
            if (gui.user[1] - 1) >= 0:
                gui.user[1] -= 1
            return 3
        elif a == "UP":
            if (gui.user[1] + 1) < gui.num_game_rows:
                gui.user[1] += 1
            return 2
        elif a == "WORK":  # Work
            gui.arrived = gui.user == gui.stn_pos[gui.goal_stn]
            return 5
        elif a == "NOOP":  # NOOP
            return 4


def visualize_game(exp, condition, actions, times, sleep=0.2):
    """Run the experiment"""
    condition = EXP_CONDS[condition]
    """ Environments: [Num Cols, Num Rows, Stations, Goal, Tool, Worker, Fetcher] """
    cols = exp[0]
    rows = exp[1]
    stn_pos = exp[2]
    goal_stn = exp[3]
    tool_pos = exp[4]
    worker_pos = exp[5]
    fetcher_pos = exp[6]
    size = exp[7]

    # Set up pygame gui
    gui = GUI(cols, rows, stn_pos, goal_stn, tool_pos, worker_pos, fetcher_pos, False,
              condition, size, 0, 1)
    # Set up fetcher robot
    # fetcher = FetcherQueryPolicy()
    fetcher = FetcherYotamPolicy(epsilon=0.05)
    gui.pause_screen = not gui.pause_screen
    gui.draw_experiment_screen()
    # Observation state
    f_obs = [worker_pos, fetcher_pos, stn_pos, tool_pos, None, None, None, None]
    done = False

    # Loop actions until expreiment is complete
    for action in actions:
        # Get fetcher move
        fetcher_move = fetcher(f_obs)

        inferred_goals = fetcher.inferred_goals
        # Get user action
        action, worker_pos, fetcher_pos = do_execute(gui, action, fetcher_move, inferred_goals,
                                                     sleep)

        # Move pickup tool
        if gui.pickup_tool != -1:
            modified_tool_pos = copy.deepcopy(tool_pos)
            modified_tool_pos[gui.pickup_tool] = fetcher_pos
            f_obs[3] = modified_tool_pos
            f_obs[4] = gui.pickup_tool

        # Modify observation state
        f_obs[0] = worker_pos
        f_obs[1] = fetcher_pos
        f_obs[5] = action
        f_obs[6] = fetcher_move[0]

    gui.on_end_level(0, False, 0, trajectory_analysis=True)
    print("finished in {} steps".format(gui.steps))
    gui.steps = 0
    print("done")


def get_trajectory(filename):
    elements = [i.split() for i in open(filename, 'r').read().split('\n')]
    scenario = 0
    for i, e in enumerate(elements):
        if not e: continue
        elif '#' in e[-1]:
            scenario = int(e[-1].split('#')[-1])
        elif '.' in e[-1]:
            break
    return [x[0] for x in elements[i:]], [y[-1] for y in elements[i:]], scenario


if __name__ == '__main__':
    filename = "participant_trajectory.txt"
    actions, times, scenario = get_trajectory(filename)
    exp = SCENARIOS[scenario]
    sleep = 0.3
    visualize_game(exp, "VG", actions, times, sleep=sleep)

    print()
