#include "AStar.h"
#include <queue>
#include <cmath>
#include <algorithm>

struct Node
{
    int row;
    int col;
    int g;
    int h;
    Node* parent;
};

struct CompareNodes
{
    bool operator()(Node* a, Node* b)
    {
        return (a->g + a->h) > (b->g + b->h);
    }
};

static int getHeuristic(Point a, Point b)
{
    return std::abs(a.row - b.row) + std::abs(a.col - b.col);
}

static bool isInsideGrid(int row, int col, int rows, int cols)
{
    return row >= 0 && row < rows && col >= 0 && col < cols;
}

static std::vector<Point> buildFinalPath(Node* endNode)
{
    std::vector<Point> path;
    Node* current = endNode;

    while (current != nullptr)
    {
        path.push_back({ current->row, current->col });
        current = current->parent;
    }

    std::reverse(path.begin(), path.end());
    return path;
}

std::vector<Point> runAStar(
    const std::vector<std::vector<int>>& grid,
    Point start,
    Point goal
)
{
    if (grid.empty() || grid[0].empty())
    {
        return {};
    }

    int rows = static_cast<int>(grid.size());
    int cols = static_cast<int>(grid[0].size());

    std::priority_queue<Node*, std::vector<Node*>, CompareNodes> openList;
    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));
    std::vector<std::vector<int>> bestCost(rows, std::vector<int>(cols, 999999));

    Node* startNode = new Node{ start.row, start.col, 0, getHeuristic(start, goal), nullptr };
    openList.push(startNode);
    bestCost[start.row][start.col] = 0;

    int rowMove[4] = { -1, 1, 0, 0 };
    int colMove[4] = { 0, 0, -1, 1 };

    while (!openList.empty())
    {
        Node* current = openList.top();
        openList.pop();

        if (visited[current->row][current->col])
        {
            continue;
        }

        visited[current->row][current->col] = true;

        if (current->row == goal.row && current->col == goal.col)
        {
            return buildFinalPath(current);
        }

        for (int i = 0; i < 4; i++)
        {
            int nextRow = current->row + rowMove[i];
            int nextCol = current->col + colMove[i];

            if (!isInsideGrid(nextRow, nextCol, rows, cols))
            {
                continue;
            }

            if (grid[nextRow][nextCol] == 1)
            {
                continue;
            }

            if (visited[nextRow][nextCol])
            {
                continue;
            }

            int newG = current->g + 1;

            if (newG < bestCost[nextRow][nextCol])
            {
                bestCost[nextRow][nextCol] = newG;

                Node* nextNode = new Node{
                    nextRow,
                    nextCol,
                    newG,
                    getHeuristic({ nextRow, nextCol }, goal),
                    current
                };

                openList.push(nextNode);
            }
        }
    }

    return {};
}