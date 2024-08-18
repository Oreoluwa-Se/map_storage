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
WorkPool::WorkPool() : stop_flag(false), is_working(false) { work_thread = std::thread(&WorkPool::run, this); }

WorkPool::~WorkPool() { stop(); }

void WorkPool::stop()
{
    {
        std::unique_lock<std::mutex> lock(cv_mutex);
        stop_flag = true;
        cv.notify_one();
    }

    if (work_thread.joinable())
        work_thread.join();
}

void WorkPool::enqueue_task(TaskType type, std::function<void()> task, PriorityRank state)
{
    task_q.emplace(type, std::move(task), state);
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
        {
            std::unique_lock<std::mutex> lock(cv_mutex);
            cv.wait(
                lock, [this]()
                { return stop_flag || !task_q.empty(); });

            if (stop_flag && task_q.empty())
                return;
        }

        Task task_item;
        if (task_q.try_pop(task_item))
        {
            boost::unique_lock<boost::shared_mutex> lock(mutex);
            is_working = true;
            lock.unlock();

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

            lock.lock();
            is_working = false;
        }

        cv.notify_one();
    }
}
