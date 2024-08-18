#ifndef WORKER_POOL_HPP
#define WORKER_POOL_HPP

#include <condition_variable>
#include <functional>
#include <future>
#include <thread>
#include <atomic>
#include <memory>
#include <tbb/concurrent_priority_queue.h>
#include <boost/thread/shared_mutex.hpp>

enum class TaskType
{
    None,
    PointInsert,
    Rebalance,
    CollapseUnwanted,
    Cleanup
};

enum class PriorityRank
{
    None = 0,
    Min = 1,
    Medium = 2,
    Max = 3,
    Super = 4
};

struct Task
{
    TaskType type;
    std::function<void()> function;
    PriorityRank state = PriorityRank::None;
    size_t retry_count = 3;

    Task();

    Task(TaskType type, std::function<void()> function, PriorityRank state);

    bool operator<(const Task &other) const;
};

// ...... Building worker pool class. Helps to offload some work from the main thread ....
class WorkPool
{
public:
    using Ptr = std::shared_ptr<WorkPool>;

    WorkPool();

    ~WorkPool();

    void enqueue_task(TaskType type, std::function<void()> task, PriorityRank state);

    bool is_occupied();

private: // functions
    void stop();

    void run();

private: // attributes
    tbb::concurrent_priority_queue<Task> task_q;
    std::thread work_thread;
    boost::shared_mutex mutex;
    std::mutex cv_mutex;
    std::condition_variable cv;
    bool stop_flag;
    bool is_working;
};

using PoolPtr = typename WorkPool::Ptr;
#endif