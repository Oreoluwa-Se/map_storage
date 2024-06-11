#include "map_storage/sections/worker_pool.hpp"
#include <iostream>

Task::Task() : type(TaskType::None) {}

Task::Task(TaskType type, std::function<void()> function, PriorityRank state)
    : type(type), function(std::move(function)), state(state) {}

bool Task::operator<(const Task &other) const
{
    return static_cast<int>(state) < static_cast<int>(other.state);
}

// ...... Building worker pool class. Helps to offload some work from the main thread ....
WorkPool::WorkPool() : stop_flag(false), is_working(false), should_run(false) {}

WorkPool::~WorkPool() { stop(); }

void WorkPool::stop()
{
    {
        {
            boost::unique_lock<boost::shared_mutex> lock(mutex);
            stop_flag = true;
        }

        std::unique_lock<std::mutex> lock(cv_mutex);
        cv.notify_one();
    }

    if (work_thread.joinable())
        work_thread.join();
}

void WorkPool::start()
{
    {
        boost::shared_lock<boost::shared_mutex> lock(mutex);
        if (should_run)
            return;
    }

    boost::unique_lock<boost::shared_mutex> lock(mutex);
    if (should_run)
        return;

    should_run = true;
    stop_flag = false;

    // run the thread
    work_thread = std::thread(&WorkPool::run, this);
}

void WorkPool::enqueue_task(TaskType type, std::function<void()> task, PriorityRank state)
{
    task_q.emplace(type, std::move(task), state);
    start();
    cv.notify_one();
}

bool WorkPool::is_occupied()
{
    boost::shared_lock<boost::shared_mutex> lock(mutex);
    return is_working || !task_q.empty();
}

void WorkPool::run()
{
    while (true)
    {
        Task task_item;
        {
            std::unique_lock<std::mutex> lock(cv_mutex);
            cv.wait(
                lock, [this]()
                { return stop_flag || !task_q.empty(); });

            if (stop_flag && task_q.empty())
            {
                should_run = false;
                return;
            }

            if (task_q.try_pop(task_item))
                is_working = true;
        }

        if (is_working)
        {
            try
            {
                // run the function
                task_item.function();
            }
            catch (const std::exception &e)
            {
                std::cerr << "Exception in worker thread: " << e.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "Unknown exception in worker thread." << std::endl;
            }

            // finalize run
            {
                boost::unique_lock<boost::shared_mutex> lock(mutex);
                is_working = false;
            }

            cv.notify_one();
        }
    }
}
