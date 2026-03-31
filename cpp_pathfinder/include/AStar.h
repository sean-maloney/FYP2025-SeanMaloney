#pragma once

#include <vector>
#include "Types.h"

std::vector<Point> runAStar(
    const std::vector<std::vector<int>>& grid,
    Point start,
    Point goal
);