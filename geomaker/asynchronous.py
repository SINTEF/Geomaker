from functools import partial, wraps
from inspect import signature
from threading import get_ident
from time import sleep
import traceback

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QProgressDialog, QProgressBar


class ThreadAbortedException(BaseException):
    pass


class ProgressManager(QObject):
    """A wrapper for QProgressBar or QProgressDialog with a unified
    interface, that can be used to listen to signals emitted by
    WorkManager.
    """

    def __init__(self, obj, parent=None):
        super().__init__(parent=parent)
        assert isinstance(obj, (QProgressBar, QProgressDialog, type(None)))
        self.obj = obj
        if obj:
            obj.setMinimum(0)
            obj.setValue(0)

    @pyqtSlot(int)
    def set_maximum(self, value):
        if self.obj:
            self.obj.setMaximum(value)

    @pyqtSlot(int)
    def set_progress(self, value):
        if self.obj:
            self.obj.setValue(value)

    @pyqtSlot(str)
    def set_message(self, message):
        if isinstance(self.obj, QProgressBar):
            self.obj.setFormat(f'{message} · %p%')
        elif isinstance(self.obj, QProgressDialog):
            self.obj.setLabelText(message)

    @pyqtSlot(object)
    def finished(self, _):
        if isinstance(self.obj, QProgressBar):
            self.obj.setValue(0)
            self.obj.setMaximum(1)
            self.obj.setFormat('')


class ThreadManager(QObject):
    """Manager for handling asynchronous job execution."""

    def __init__(self, parent):
        super().__init__(parent=parent)
        self._thread = None
        self._worker = None
        self._parent = parent
        self._queue = []

    def enqueue(self, job, progress=None, priority='inherit', callback=None):
        if self._thread is None:
            self._run(job, progress, priority, callback)
        else:
            self._queue.append((job, progress, priority, callback))

    def _run(self, job, progress, priority, callback):
        """Run a job asynchronously, using a QProgressBar or a
        QProgressDialog to report progress.
        """

        # Only one thread at a time
        assert self._thread is None

        # Create the worker, progress manager and thread
        worker = WorkManager(job)
        reporter = ProgressManager(progress, parent=self._parent)
        reporter.set_maximum(worker.total_progress)

        thread = QThread(parent=self._parent)
        worker.moveToThread(thread)

        # Connect signals
        thread.started.connect(worker.process)
        worker.sync_query.connect(self.sync_query)
        worker.progress_changed.connect(reporter.set_progress)
        worker.message_changed.connect(reporter.set_message)
        worker.finished.connect(reporter.finished)
        worker.finished.connect(self.finished)

        # Prevent undue garbage collection!
        self._thread = thread
        self._worker = worker
        self._reporter = reporter
        self._callback = callback

        # Select priority and run thread
        priority = {
            'idle': QThread.IdlePriority,
            'lowest': QThread.LowestPriority,
            'low': QThread.LowPriority,
            'normal': QThread.NormalPriority,
            'high': QThread.HighPriority,
            'highest': QThread.HighestPriority,
            'critical': QThread.TimeCriticalPriority,
            'inherit': QThread.InheritPriority,
        }[priority]
        thread.start(priority)

    @pyqtSlot(object)
    def sync_query(self, func):
        """Execute FUNC synchronously and report the return value back."""
        try:
            retval = func()
        except Exception:
            traceback.print_exc()
            retval = None
        self._worker.respond_synchronously(retval)

    @pyqtSlot(object)
    def finished(self, result):
        """Cleanly exit the thread and clean up.
        After this, the thread manager can be re-used for new jobs.
        """
        if self._callback:
            self._callback(result)

        self._thread.quit()
        self._thread.wait()
        self._thread.deleteLater()
        self._worker.deleteLater()
        self._reporter.deleteLater()
        self._thread = None
        self._worker = None
        self._reporter = None
        self._callback = None

        if self._queue:
            args = self._queue.pop(0)
            self._run(*args)

    def close(self):
        if self._thread is not None:
            self._worker.close()
            self._thread.quit()
            self._thread.wait()


def check_close(func):
    @wraps(func)
    def inner(self, *args, **kwargs):
        if self._closing:
            raise ThreadAbortedException
        return func(self, *args, **kwargs)
    return inner


class WorkManager(QObject):
    """A QObject that executes a body of work in a thread.
    This object should be moved to the relevant thread, and the
    'process' slot should be signalled upon start.
    """

    # Signalled when work has been completed
    finished = pyqtSignal(object)

    # Signalled when a function must be executed synchronously,
    # in the calling thread. If nothing listens to this signal,
    # the thread will stall upon such jobs.
    sync_query = pyqtSignal(object)

    # Signalled when the progress meter is updated.
    progress_changed = pyqtSignal(int)

    # Signalled when the name of the current job changes.
    message_changed = pyqtSignal(str)

    def __init__(self, job):
        super().__init__()
        self.job = job
        self._closing = False

        # The job may consist of sub-jobs. To simplify, each atomic
        # job is assumed to constitute 100 progress points.
        self.total_progress = 100 * job.njobs()

        # The total progress points for all completed jobs.
        self.root_progress = 0

        # Progress of the currently running job. The job itself may
        # change both of these as it runs.
        self.current_max = 100
        self.current_progress = 0

        # Flags for executing synchronous jobs.
        self.received = False
        self.response = None

    def close(self):
        self._closing = True

    @pyqtSlot()
    def process(self):
        """Main entry point for job processing."""
        try:
            result = self.job.process(self)
            self.finished.emit(result)
        except ThreadAbortedException:
            pass
        from .db import Database
        Database().remove_session()

    def report_total_progress(self):
        """Compute the total progress and emit the progress_changed signal."""
        progress = int(self.root_progress + 100 * self.current_progress / self.current_max)
        self.progress_changed.emit(progress)

    @check_close
    def report_message(self, message):
        """Emit the message_changed signal. Called from the job."""
        self.message_changed.emit(message)

    @check_close
    def report_finished(self, num=1):
        """Signal that NUM jobs have been finished."""
        self.root_progress += 100 * num
        self.current_progress = 0
        self.current_max = 100
        self.report_total_progress()

    @check_close
    def report_max(self, value):
        """Change the maximal progress value of the currently running job."""
        self.current_max = value
        self.report_total_progress()

    @check_close
    def report_progress(self, value):
        """Change the progress value of the currently running job."""
        self.current_progress = value
        self.report_total_progress()

    @check_close
    def increment_progress(self, value=1):
        """Change the progress value of the currently running job."""
        self.current_progress += value
        self.report_total_progress()

    def run_synchronously(self, func):
        """Run the function FUNC synchronously in the calling thread,
        and return its return value.
        """
        self.received = False
        self.sync_query.emit(func)
        while not self.received:
            sleep(1e-3)
        return self.response

    def respond_synchronously(self, response):
        """Trigger a response from the calling thread. This function
        should be called after receiving the sync_query signal.
        """
        self.response = response
        self.received = True


class Job:
    """Abstract base class for a job.
    Subclasses should implement process(manager, *args, **kwargs).
    """

    def njobs(self):
        return 1


class SyncJob(Job):
    """A job that runs synchronously in the calling thread."""

    def __init__(self, func, message):
        self.message = message
        self.func = func

    def njobs(self):
        # Synchronous jobs should be small, quick and invisible
        return 0

    def process(self, manager, *args, **kwargs):
        retval = manager.run_synchronously(partial(self.func, *args, **kwargs))
        return retval


class AsyncJob(Job):
    """A job that runs asynchronously in the child thread.
    The wrapped function will receive the work manager object
    as the first argument and must report progress to it.

    If message is given, that message is reported before the wrappee
    runs, in which case it does not have to report it.

    If maximum is given, it is reported before the wrappee runs, and
    the progress is set to maximum after it returns. In this case, the
    wrappee only has to report intermediate progress.
    """

    def __init__(self, func, message=None, maximum=None):
        self.message = message
        self.maximum = maximum
        self.func = func

    def njobs(self):
        if self.maximum == 'empty':
            return 0
        return 1

    def pre_process(self, manager):
        if self.maximum == 'empty':
            return
        if self.maximum == 'simple':
            manager.report_max(1)
        elif self.maximum is not None:
            manager.report_max(self.maximum)
        if self.message is not None:
            manager.report_message(self.message)

    def post_process(self, manager):
        if self.maximum == 'empty':
            return
        if self.maximum == 'simple':
            manager.report_progress(1)
        elif self.maximum is not None:
            manager.report_progress(self.maximum)
        manager.report_finished()

    def process(self, manager, *args, **kwargs):
        self.pre_process(manager)
        try:
            if self.maximum in ('empty', 'simple'):
                retval = self.func(*args, **kwargs)
            else:
                retval = self.func(*args, **kwargs, manager=manager)
        except Exception:
            traceback.print_exc()
            retval = None
        self.post_process(manager)
        return retval


class SequenceJob(Job):
    """A sequence of jobs executed in order."""

    def __init__(self, jobs):
        self.jobs = jobs

    def njobs(self):
        return sum(job.njobs() for job in self.jobs)

    def process(self, manager, *args, **kwargs):
        for job in self.jobs:
            job.process(manager, *args, **kwargs)
        return None


class ConditionalJob(Job):

    def __init__(self, job):
        self.job = job

    def njobs(self):
        return self.job.njobs()

    def process(self, manager, arg):
        if arg:
            return self.job.process(manager, arg)
        manager.report_finished(self.njobs())
        return None


class PipeJob(SequenceJob):
    """A sequence of jobs executed in order.
    The return value from one is passed to the next.
    If any job returns None, the sequence terminates.
    """

    def process(self, manager, *args, **kwargs):
        job, *rest = self.jobs
        retval = job.process(manager, *args, **kwargs)
        while rest and retval is not None:
            job, *rest = rest
            retval = job.process(manager, retval)
        manager.report_finished(len(rest))
        return retval


def async_job(**async_kwargs):
    """Wrap a function so that it instead returns an asynchronous job,
    which when executed runs the wrappee in a different thread.
    """

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            return AsyncJob(partial(func, *args, **kwargs), **async_kwargs)
        return inner
    return decorator


def sync_job(**sync_kwargs):
    """Wrap a function so that it instead returns a synchronous job,
    which when executed runs the wrappee in the calling thread.
    """

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            return SyncJob(partial(func, *args, **kwargs), **sync_kwargs)
        return inner
    return decorator
