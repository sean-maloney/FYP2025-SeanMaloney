#include <iostream>
#include <vector>
#include <string>
#include "AStar.h"
#include "Grid.h"

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: AStarProject <input_file> <output_file>\n";
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];

    std::vector<std::vector<int>> grid;
    Point start{};
    Point goal{};

    bool inputOk = readGridInputFile(inputFile, grid, start, goal);

    if (!inputOk)
    {
        std::cerr << "Could not read input file\n";
        return 1;
    }

    std::vector<Point> path = runAStar(grid, start, goal);

    bool outputOk = writePathOutputFile(outputFile, path);

    if (!outputOk)
    {
        std::cerr << "Could not write output file\n";
        return 1;
    }

    std::cout << "Finished running A*\n";
    return 0;
}