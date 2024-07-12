import torch


def _interpolate_bilinear(device, grid: torch.Tensor, query_points: torch.Tensor, indexing: str = 'ij') -> torch.Tensor:
    """
    Finds values for query points on a grid using bilinear interpolation.

    :param grid         : 4-D float Tensor
        :shape: (batch, height, width, channels)
    :param query points : 3-D float Tensor
        :shape: (batch, N, 2)
    :param indexing     : whether the query points are specified as row and column (ij),
      or Cartesian coordinates (xy)

    :return             : interp
        :shape: (batch, N, channels)

    """
    if indexing != 'ij' and indexing != 'xy':
        raise ValueError('Indexing mode must be \'ij\' or \'xy\'')

    grid = torch.as_tensor(grid)       
    query_points = torch.as_tensor(query_points)

    shape = list(grid.size())

    # if len(shape) != 4:
    if len(shape) != 5:
        msg = 'Grid must be 4 dimensional. Received size: '
        raise ValueError(msg + str(len(grid.shape)))

    # batch_size, height, width, channels = shape
    batch_size, height, width, depth, channels = shape      
    query_type = query_points.dtype
    grid_type = grid.dtype

    # if (len(query_points.shape) != 3 or query_points.shape[2] != 2):
    if (len(query_points.shape) != 3 or query_points.shape[2] != 3):
        msg = ('Query points must be 3 dimensional and size 2 in dim 2. Received '
                'size: ')
        raise ValueError(msg + str(query_points.shape))
    
    _, num_queries, _ = list(query_points.size())

    if height < 2 or width < 2:
      msg = 'Grid must be at least batch_size x 2 x 2 in size. Received size: '
      raise ValueError(msg + str(grid.shape))

    alphas = []
    floors = []
    ceils = []
    # index_order = [0, 1] if indexing == 'ij' else [1, 0]
    index_order = [0, 1, 2] if indexing == 'ij' else [1, 0, 2]

    unstacked_query_points = torch.unbind(query_points, dim=2)

    for dim in index_order:
        # queries = unstacked_query_points[dim]
        # size_in_indexing_dimension = shape[dim + 1]
        
        # # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # # is still a valid index into the grid.
        # max_floor = torch.tensor(size_in_indexing_dimension - 2, dtype=query_type).to(device)
        # min_floor = torch.tensor(0.0, dtype=query_type).to(device)
        # floor = torch.minimum(torch.maximum(min_floor, torch.floor(queries)), max_floor)
        # int_floor = floor.type(torch.int32)
        # floors.append(int_floor)

        # used for 3D interpolate
        queries = unstacked_query_points[dim]
        # if dim == 2:
        #     print("queries", queries, queries.shape)

        size_in_indexing_dimension = shape[dim + 1]
        max_floor = torch.tensor(size_in_indexing_dimension - 2, dtype=query_type).to(device)
        min_floor = torch.tensor(0.0, dtype=query_type).to(device)
        floor = torch.minimum(torch.maximum(min_floor, torch.floor(queries)), max_floor)
        int_floor = floor.type(torch.int32)
        floors.append(int_floor)


        ceil = int_floor + 1
        ceils.append(ceil)
        # print("int_floor", int_floor)
        # print("ceils", ceil.shape)
        
        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = (queries - floor).type(grid_type)
        # print("alpha", alpha)

        min_alpha = torch.tensor(0.0, dtype=grid_type).to(device)
        max_alpha = torch.tensor(1.0, dtype=grid_type).to(device)
        alpha = torch.minimum(torch.maximum(min_alpha, alpha), max_alpha)
        # print("alpha", alpha, alpha.shape)
        
        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = torch.unsqueeze(alpha, 2)
        alphas.append(alpha)
        # print("alphas", alphas)
        
    # flattened_grid = torch.reshape(grid, [batch_size * height * width, channels])
    flattened_grid = torch.reshape(grid, [batch_size * height * width * depth, channels])
    # batch_offsets  = torch.reshape(torch.arange(0, batch_size) * height * width, [batch_size, 1]).to(device)
    batch_offsets  = torch.reshape(torch.arange(0, batch_size) * height * width * depth, [batch_size, 1]).to(device)
    
    def gather(y_coords, x_coords, z_coords):
        # linear_coordinates = batch_offsets + y_coords * width + x_coords
        # print("coords", y_coords, x_coords, z_coords)
        # print( "z", z_coords, "off", batch_offsets)
        linear_coordinates = batch_offsets + y_coords * width * depth + x_coords * depth + z_coords
        gathered_values = flattened_grid[linear_coordinates]       
        return torch.reshape(gathered_values,[batch_size, num_queries, channels])
        
    # grab the pixel values in the 4 corners around each query point
    # 3D 8 corner points
    # print("floors", floors[0].shape)
    # print("ceils", ceils)
    # now, do the actual interpolation
    top_left_near        = gather(floors[0], floors[1], floors[2])
    top_left_far         = gather(floors[0], floors[1], ceils[2])
    interp_t_l      = alphas[2] * (top_left_far     - top_left_near)  + top_left_near

    del top_left_near
    del top_left_far

    top_right_near       = gather(floors[0], ceils[1], floors[2])
    top_right_far        = gather(floors[0], ceils[1], ceils[2])
    interp_t_r      = alphas[2] * (top_right_far    - top_right_near) + top_right_near

    del top_right_near
    del top_right_far

    interp_top      = alphas[1] * (interp_t_r     - interp_t_l)    + interp_t_l

    del interp_t_l
    del interp_t_r

    bottom_left_near     = gather(ceils[0], floors[1], floors[2])
    bottom_left_far      = gather(ceils[0], floors[1], ceils[2])
    interp_b_l      = alphas[2] * (bottom_left_far     - bottom_left_near)  + bottom_left_near

    del bottom_left_near
    del bottom_left_far

    bottom_right_near    = gather(ceils[0], ceils[1], floors[2])
    bottom_right_far     = gather(ceils[0], ceils[1], ceils[2])
    interp_b_r      = alphas[2] * (bottom_right_far   - bottom_right_near) + bottom_right_near

    del bottom_right_near
    del bottom_right_far

    interp_bottom   = alphas[1] * (interp_b_r    - interp_b_l)    + interp_b_l
    del interp_b_l
    del interp_b_r

    interp          = alphas[0] * (interp_bottom - interp_top)  + interp_top

    # print("interp", interp.shape, interp)
    
    return interp
    
def dense_image_warp(device, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Image warping using per-pixel flow vectors.

    :param image         : 4-D float Tensor
        :shape: (batch, height, width, channels)
    :param flow          : 4-D float Tensor
        :shape: (batch, height, width, 2)
   
    :return             : interpolated
        :shape: (batch, height, width, channels)

    """

    # batch_size, height, width, channels = list(image.size())
    # 3D
    batch_size, height, width, depth, channels = list(image.size())
    print("depth", height, width, depth)

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    # grid_x, grid_y = torch.meshgrid(torch.arange(width), torch.arange(height), indexing = 'xy')
    # real_3D
    # grid_x, grid_y, grid_z = torch.meshgrid(torch.linspace(-1, 1, width), torch.linspace(-1, 1, height), torch.linspace(0.88, 1.12, depth), indexing = 'xy')
    # grid 3D
    grid_x, grid_y, grid_z = torch.meshgrid(torch.arange(width), torch.arange(height), torch.arange(depth), indexing = 'xy')

    grid_x = grid_x.to(device)
    grid_y = grid_y.to(device)
    grid_z = grid_z.to(device)

    stacked_grid = torch.stack([grid_y, grid_x, grid_z], axis=3).type(flow.dtype)
    # print("stacked_grid", stacked_grid)
    batched_grid            = torch.unsqueeze(stacked_grid, axis=0)
    # print("batched_grid", batched_grid)
    # print("flow", flow)
    query_points_on_grid    = batched_grid - flow
    # print("query_points_on_grid", query_points_on_grid)
    

    # query_points_flattened  = torch.reshape(query_points_on_grid, [batch_size, height * width, 2])
    query_points_flattened  = torch.reshape(query_points_on_grid, [batch_size, height * width * depth, 3])

    # Compute values at the query points, then reshape the result back to the
    # image grid.
    interpolated = _interpolate_bilinear(device, image, query_points_flattened)
    interpolated = torch.reshape(interpolated, [batch_size, height, width, depth, channels])
    return interpolated
