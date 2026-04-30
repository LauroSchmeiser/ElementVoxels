// MainThreadDispatcher.h
#pragma once
#include <mutex>
#include <vector>
#include <functional>
#include <cstdint>

class MainThreadDispatcher {
private:
    std::mutex mutex;

    struct TaskItem {
        uint64_t epoch;
        std::function<void()> fn;
    };

    std::vector<TaskItem> tasks;
    uint64_t currentEpoch = 1;

public:
    uint64_t epoch() const { return currentEpoch; }

    void bumpEpochAndClear() {
        std::lock_guard<std::mutex> lock(mutex);
        ++currentEpoch;
        tasks.clear();
    }

    void dispatch(std::function<void()> task) {
        std::lock_guard<std::mutex> lock(mutex);
        tasks.push_back(TaskItem{ currentEpoch, std::move(task) });
    }

    void execute() {
        std::vector<TaskItem> local;
        {
            std::lock_guard<std::mutex> lock(mutex);
            local.swap(tasks);
        }

        // Only run tasks belonging to the current epoch
        for (auto& t : local) {
            if (t.epoch == currentEpoch && t.fn) {
                t.fn();
            }
        }
    }
};