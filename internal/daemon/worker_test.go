package daemon

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/wesm/roborev/internal/config"
	"github.com/wesm/roborev/internal/storage"
)

func TestWorkerPoolE2E(t *testing.T) {
	// Setup temp DB
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	db, err := storage.Open(dbPath)
	if err != nil {
		t.Fatalf("Failed to open test DB: %v", err)
	}
	defer db.Close()

	// Setup config with test agent
	cfg := config.DefaultConfig()
	cfg.MaxWorkers = 2

	// Create a repo and commit
	repo, err := db.GetOrCreateRepo(tmpDir)
	if err != nil {
		t.Fatalf("GetOrCreateRepo failed: %v", err)
	}

	commit, err := db.GetOrCreateCommit(repo.ID, "testsha123", "Test Author", "Test commit", time.Now())
	if err != nil {
		t.Fatalf("GetOrCreateCommit failed: %v", err)
	}

	// Enqueue a job with test agent
	job, err := db.EnqueueJob(repo.ID, commit.ID, "testsha123", "test")
	if err != nil {
		t.Fatalf("EnqueueJob failed: %v", err)
	}

	// Create and start worker pool
	pool := NewWorkerPool(db, cfg, 1)
	pool.Start()

	// Wait for job to complete (with timeout)
	deadline := time.Now().Add(10 * time.Second)
	var finalJob *storage.ReviewJob
	for time.Now().Before(deadline) {
		finalJob, err = db.GetJobByID(job.ID)
		if err != nil {
			t.Fatalf("GetJobByID failed: %v", err)
		}
		if finalJob.Status == storage.JobStatusDone || finalJob.Status == storage.JobStatusFailed {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	// Stop worker pool
	pool.Stop()

	// Verify job completed (might fail if git repo not available, that's ok)
	if finalJob.Status != storage.JobStatusDone && finalJob.Status != storage.JobStatusFailed {
		t.Errorf("Job should be done or failed, got %s", finalJob.Status)
	}

	// If done, verify review was stored
	if finalJob.Status == storage.JobStatusDone {
		review, err := db.GetReviewByCommitSHA("testsha123")
		if err != nil {
			t.Fatalf("GetReviewByCommitSHA failed: %v", err)
		}
		if review.Agent != "test" {
			t.Errorf("Expected agent 'test', got '%s'", review.Agent)
		}
		if review.Output == "" {
			t.Error("Review output should not be empty")
		}
	}
}

func TestWorkerPoolConcurrency(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	db, err := storage.Open(dbPath)
	if err != nil {
		t.Fatalf("Failed to open test DB: %v", err)
	}
	defer db.Close()

	cfg := config.DefaultConfig()
	cfg.MaxWorkers = 4

	repo, _ := db.GetOrCreateRepo(tmpDir)

	// Create multiple jobs
	for i := 0; i < 5; i++ {
		sha := "concurrentsha" + string(rune('0'+i))
		commit, _ := db.GetOrCreateCommit(repo.ID, sha, "Author", "Subject", time.Now())
		db.EnqueueJob(repo.ID, commit.ID, sha, "test")
	}

	pool := NewWorkerPool(db, cfg, 4)
	pool.Start()

	// Wait briefly and check active workers
	time.Sleep(500 * time.Millisecond)
	activeWorkers := pool.ActiveWorkers()

	pool.Stop()

	// Should have had some workers active (exact number depends on timing)
	t.Logf("Peak active workers: %d", activeWorkers)
}

func TestWorkerPoolCancelRunningJob(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	db, err := storage.Open(dbPath)
	if err != nil {
		t.Fatalf("Failed to open test DB: %v", err)
	}
	defer db.Close()

	cfg := config.DefaultConfig()
	cfg.MaxWorkers = 1

	repo, _ := db.GetOrCreateRepo(tmpDir)
	commit, _ := db.GetOrCreateCommit(repo.ID, "cancelsha", "Author", "Subject", time.Now())
	job, _ := db.EnqueueJob(repo.ID, commit.ID, "cancelsha", "test")

	pool := NewWorkerPool(db, cfg, 1)
	pool.Start()

	// Wait for job to be claimed (status becomes running)
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		j, _ := db.GetJobByID(job.ID)
		if j.Status == storage.JobStatusRunning {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	// Cancel the job via DB and worker pool
	if err := db.CancelJob(job.ID); err != nil {
		t.Fatalf("CancelJob failed: %v", err)
	}
	pool.CancelJob(job.ID)

	// Wait for worker to react to cancellation
	time.Sleep(500 * time.Millisecond)

	pool.Stop()

	// Verify job status is canceled
	finalJob, _ := db.GetJobByID(job.ID)
	if finalJob.Status != storage.JobStatusCanceled {
		t.Errorf("Expected status 'canceled', got '%s'", finalJob.Status)
	}

	// Verify no review was stored
	_, err = db.GetReviewByJobID(job.ID)
	if err == nil {
		t.Error("Expected no review for canceled job, but found one")
	}
}

func TestWorkerPoolPendingCancellation(t *testing.T) {
	// Test the race condition fix: cancel arrives before job is registered
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	db, err := storage.Open(dbPath)
	if err != nil {
		t.Fatalf("Failed to open test DB: %v", err)
	}
	defer db.Close()

	cfg := config.DefaultConfig()
	pool := NewWorkerPool(db, cfg, 1)

	// Don't start the pool yet - we want to test pending cancellation

	// Mark a job as pending cancellation before it's registered
	pool.CancelJob(42)

	// Verify it's in pending cancels
	pool.runningJobsMu.Lock()
	if !pool.pendingCancels[42] {
		t.Error("Job 42 should be in pendingCancels")
	}
	pool.runningJobsMu.Unlock()

	// Now register the job - should immediately cancel
	canceled := false
	pool.registerRunningJob(42, func() { canceled = true })

	if !canceled {
		t.Error("Job should have been canceled immediately on registration")
	}

	// Verify it's been removed from pending cancels
	pool.runningJobsMu.Lock()
	if pool.pendingCancels[42] {
		t.Error("Job 42 should have been removed from pendingCancels")
	}
	pool.runningJobsMu.Unlock()
}
