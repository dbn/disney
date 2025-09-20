"""Periodic scheduling for data pipeline."""

import schedule
import time
import asyncio
from typing import Callable, Optional
from datetime import datetime

from ..shared.logging import setup_logging

logger = setup_logging("data-pipeline-scheduler")


class PipelineScheduler:
    """Scheduler for periodic data pipeline execution."""
    
    def __init__(self):
        self.running = False
        self.scheduled_jobs = []
    
    def add_job(
        self, 
        func: Callable, 
        schedule_time: str, 
        job_name: str = "pipeline_job"
    ):
        """Add a scheduled job.
        
        Args:
            func: Function to execute
            schedule_time: Schedule time (e.g., "daily at 02:00", "hourly")
            job_name: Name for the job
        """
        try:
            # Parse schedule time and add job
            if schedule_time.startswith("daily at "):
                time_str = schedule_time.replace("daily at ", "")
                job = schedule.every().day.at(time_str).do(self._run_job, func, job_name)
            elif schedule_time == "hourly":
                job = schedule.every().hour.do(self._run_job, func, job_name)
            elif schedule_time == "daily":
                job = schedule.every().day.do(self._run_job, func, job_name)
            elif schedule_time == "weekly":
                job = schedule.every().week.do(self._run_job, func, job_name)
            else:
                raise ValueError(f"Unsupported schedule time: {schedule_time}")
            
            self.scheduled_jobs.append(job)
            logger.info(f"Added scheduled job '{job_name}' with schedule: {schedule_time}")
            
        except Exception as e:
            logger.error(f"Error adding scheduled job: {str(e)}")
            raise
    
    def _run_job(self, func: Callable, job_name: str):
        """Run a scheduled job.
        
        Args:
            func: Function to execute
            job_name: Name of the job
        """
        try:
            logger.info(f"Starting scheduled job: {job_name}")
            start_time = datetime.now()
            
            # Run the job (handle both sync and async functions)
            if asyncio.iscoroutinefunction(func):
                asyncio.run(func())
            else:
                func()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Completed scheduled job '{job_name}' in {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error running scheduled job '{job_name}': {str(e)}")
    
    def start(self):
        """Start the scheduler."""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        logger.info("Starting pipeline scheduler")
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {str(e)}")
        finally:
            self.running = False
    
    def stop(self):
        """Stop the scheduler."""
        logger.info("Stopping pipeline scheduler")
        self.running = False
    
    def get_next_run_times(self) -> dict:
        """Get next run times for all scheduled jobs.
        
        Returns:
            Dictionary with job names and next run times
        """
        next_runs = {}
        
        for job in self.scheduled_jobs:
            if hasattr(job, 'next_run'):
                next_runs[job.job_func.__name__] = job.next_run.isoformat()
        
        return next_runs


# Global scheduler instance
_scheduler = None


def get_scheduler() -> PipelineScheduler:
    """Get or create scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = PipelineScheduler()
    return _scheduler
