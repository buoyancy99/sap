import numpy as np

def loc2grid(loc):
    loc_x, loc_y, loc_z = loc[0], loc[1], loc[2]
    grid_size = 0.01
    # import pdb; pdb.set_trace()
    grid_x = (loc_x // grid_size).astype(np.int)
    grid_y = (loc_y // grid_size).astype(np.int)
    grid_z = (loc_z // grid_size).astype(np.int)
    return np.array([grid_x, grid_y, grid_z])


def get_block(world_array, current_grid, idx):
    # note idx is something like '0-10' to indicate location
    width = 5
    dx = 5
    dy = 5
    dz = 5

    grid_middle = (current_grid + idx * np.array([dx, dy, dz]))
    grid_start  = grid_middle - np.array([1,1,1])
    grid_end = grid_start + np.array([width, width, width])
    block = world_array[grid_start[0]:grid_end[0], grid_start[1]:grid_end[1], grid_start[2]:grid_end[2]]
    return block

def get_block_dict(world_array, current_pos):
    current_grid = loc2grid(current_pos)
    all_blocks = np.array(np.meshgrid(*([np.arange(3) - 1] * 3))).reshape(3, -1)
    block_dict = {}
    for idx in range(all_blocks.shape[1]):
        # if (1-block).all():
        #     continue
        block = all_blocks[:, idx]
        tmp_block = get_block(world_array, current_grid, block)
        if tmp_block.shape != (5,5,5):
            tmp_block = np.zeros((5, 5, 5))
        block_dict[''.join([str(e) for e in block])] = tmp_block
    return block_dict
