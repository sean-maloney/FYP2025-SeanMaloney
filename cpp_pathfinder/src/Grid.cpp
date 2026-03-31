#include "Grid.h"
#include <fstream>

bool readGridInputFile(
    const std::string& fileName,
    std::vector<std::vector<int>>& grid,
    Point& start,
    Point& goal
)
{
    std::ifstream file(fileName);

    if (!file.is_open())
    {
        return false;
    }

    int rows = 0;
    int cols = 0;

    file >> rows >> cols;
    file >> start.row >> start.col;
    file >> goal.row >> goal.col;

    if (rows <= 0 || cols <= 0)
    {
        return false;
    }

    grid.clear();
    grid.resize(rows, std::vector<int>(cols, 0));

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            file >> grid[r][c];
        }
    }

    file.close();
    return true;
}

bool writePathOutputFile(
    const std::string& fileName,
    const std::vector<Point>& path
)
{
    std::ofstream file(fileName);

    if (!file.is_open())
    {
        return false;
    }

    if (path.empty())
    {
        file << "NO_PATH\n";
        file.close();
        return true;
    }

    file << "PATH_FOUND\n";
    file << path.size() << "\n";

    for (const Point& point : path)
    {
        file << point.row << " " << point.col << "\n";
    }

    file.close();
    return true;
}