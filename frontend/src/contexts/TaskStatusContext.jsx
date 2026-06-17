/**
 * TaskStatusContext
 * 全域追蹤背景任務（pdf_analyze / vectorize / vl_vectorize）的狀態，每 4 秒 poll 一次。
 * 任務 ID 存在 localStorage key: "activeTasks" (JSON array)
 */
import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import apiClient from "../services/api";
import useAuthStore from "../stores/authStore";

const TaskStatusContext = createContext(null);

const STORAGE_KEY = "activeTasks"; // [{task_id, document_id, document_title, task_type}]
const POLL_INTERVAL = 4000;

export const TaskStatusProvider = ({ children }) => {
  const [tasks, setTasks] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
    } catch {
      return [];
    }
  });
  // taskStatuses: { [task_id]: TaskRead }
  const [taskStatuses, setTaskStatuses] = useState({});
  // onComplete callbacks: { [task_id]: (taskData) => void }
  const callbacksRef = useRef({});

  const _persist = (list) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  };

  const addTask = useCallback((taskEntry, onCompleteCallback) => {
    setTasks((prev) => {
      const next = [...prev.filter((t) => t.task_id !== taskEntry.task_id), taskEntry];
      _persist(next);
      return next;
    });
    if (onCompleteCallback) {
      callbacksRef.current[taskEntry.task_id] = onCompleteCallback;
    }
  }, []);

  const removeTask = useCallback((taskId) => {
    delete callbacksRef.current[taskId];
    setTasks((prev) => {
      const next = prev.filter((t) => t.task_id !== taskId);
      _persist(next);
      return next;
    });
    setTaskStatuses((prev) => {
      const { [taskId]: _, ...rest } = prev;
      return rest;
    });
  }, []);

  // Poll active tasks
  useEffect(() => {
    const poll = async () => {
      // 未登入就不打 API —— 否則所有 request 會 401，孤兒清理也走不到。
      if (!useAuthStore.getState().token) return;

      const activeTasks = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
      if (!activeTasks.length) return;

      const results = await Promise.allSettled(
        activeTasks.map((t) => apiClient.get(`tasks/${t.task_id}`))
      );

      const newStatuses = {};
      const orphanTaskIds = [];
      results.forEach((result, idx) => {
        const entry = activeTasks[idx];
        if (result.status === "fulfilled") {
          const data = result.value.data;
          newStatuses[entry.task_id] = data;

          // 觸發 onComplete callback（completed 或 failed 時各呼叫一次）
          const isDone = data.status === "completed" || data.status === "failed";
          const cb = callbacksRef.current[entry.task_id];
          if (isDone && cb) {
            cb(data);
            delete callbacksRef.current[entry.task_id];
          }
        } else {
          // server 回 404 / 401 / 403 → 該任務已不存在或當前使用者沒權限看，
          // 一律視為孤兒並從 localStorage 清掉。
          // 5xx / 網路錯誤不在這條清，避免後端短暫掛掉就丟失追蹤。
          const httpStatus = result.reason?.response?.status;
          if (httpStatus === 404 || httpStatus === 401 || httpStatus === 403) {
            orphanTaskIds.push(entry.task_id);
          }
        }
      });

      setTaskStatuses((prev) => ({ ...prev, ...newStatuses }));

      if (orphanTaskIds.length) {
        const orphanSet = new Set(orphanTaskIds);
        orphanTaskIds.forEach((tid) => {
          delete callbacksRef.current[tid];
        });
        setTasks((prev) => {
          const next = prev.filter((t) => !orphanSet.has(t.task_id));
          _persist(next);
          return next;
        });
        setTaskStatuses((prev) => {
          const next = { ...prev };
          orphanTaskIds.forEach((tid) => delete next[tid]);
          return next;
        });
      }
    };

    const timer = setInterval(poll, POLL_INTERVAL);
    poll(); // immediate first poll
    return () => clearInterval(timer);
  }, []);

  return (
    <TaskStatusContext.Provider value={{ tasks, taskStatuses, addTask, removeTask }}>
      {children}
    </TaskStatusContext.Provider>
  );
};

// eslint-disable-next-line react-refresh/only-export-components
export const useTaskStatus = () => {
  const ctx = useContext(TaskStatusContext);
  if (!ctx) throw new Error("useTaskStatus must be used within TaskStatusProvider");
  return ctx;
};
