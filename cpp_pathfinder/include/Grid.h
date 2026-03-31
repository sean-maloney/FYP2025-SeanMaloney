#pragma once

#include <string>
#include <vector>
#include "Types.h"

bool readGridInputFile(
    const std::string& fileName,
    std::vector<std::vector<int>>& grid,
    Point& start,
    Point& goal
);

bool writePathOutputFile(
    const std::string& fileName,
    const std::vector<Point>& path
);