package agent

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"

	"github.com/wesm/roborev/internal/git"
)

// ExperimentRunnerAgent runs ML experiments based on commit changes
type ExperimentRunnerAgent struct {
	PythonCmd string
}

// NewExperimentRunnerAgent creates a new experiment runner agent
func NewExperimentRunnerAgent() *ExperimentRunnerAgent {
	return &ExperimentRunnerAgent{
		PythonCmd: "poetry",
	}
}

func (a *ExperimentRunnerAgent) Name() string {
	return "experiment-runner"
}

func (a *ExperimentRunnerAgent) Review(ctx context.Context, repoPath, commitSHA, prompt string) (string, error) {
	// 1. Get changed files
	files, err := git.GetFilesChanged(repoPath, commitSHA)
	if err != nil {
		return "", fmt.Errorf("failed to get changed files: %w", err)
	}

	// 2. Detect which experiments to run
	experiments := a.detectExperiments(files)

	// 3. Run each experiment and collect results
	var results []string
	for _, exp := range experiments {
		result := a.runExperiment(ctx, repoPath, exp)
		results = append(results, result)
	}

	// 4. Format and return output
	return a.formatOutput(files, experiments, results), nil
}

func (a *ExperimentRunnerAgent) detectExperiments(files []string) []string {
	// Only run one experiment at a time to avoid timeout/resource issues
	// Priority: new experiment file > modified experiment file > baseline (default)

	for _, f := range files {
		// If a new/modified experiment file exists in experiments/, run that one
		// This catches new experiments like "experiments/my_new_experiment.py"
		if strings.HasPrefix(f, "experiments/") && strings.HasSuffix(f, ".py") {
			return []string{f}
		}
	}

	// Default: run baseline for any other changes (including library_code/)
	return []string{"experiments/baseline_experiment.py"}
}

func (a *ExperimentRunnerAgent) runExperiment(ctx context.Context, repoPath, experimentPath string) string {
	// Use the context timeout from the daemon (configured via job_timeout_minutes)
	cmd := exec.CommandContext(ctx, a.PythonCmd, "run", "python", experimentPath)
	cmd.Dir = repoPath

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	output := stdout.String()
	errOutput := stderr.String()

	// Parse MLFlow run ID and experiment ID from output
	runID, experimentID := a.parseMLFlowInfo(output + errOutput)
	mlflowURL := a.constructMLFlowURL(runID, experimentID)

	// Parse metrics (look for F1, precision, recall patterns)
	metrics := a.parseMetrics(output)

	// Format result
	var result strings.Builder
	result.WriteString(fmt.Sprintf("## Experiment: %s\n", experimentPath))

	if err != nil {
		result.WriteString("**Status**: Failed\n")
		// Check if this was a timeout/cancellation
		if ctx.Err() == context.DeadlineExceeded {
			result.WriteString("**Error**: Timeout exceeded\n")
		} else if ctx.Err() == context.Canceled {
			result.WriteString("**Error**: Job was canceled\n")
		} else {
			result.WriteString(fmt.Sprintf("**Error**: %s\n", err.Error()))
		}

		// Show stderr if present
		if errOutput != "" {
			result.WriteString("\n### Stderr\n```\n")
			result.WriteString(truncateOutput(errOutput, 50))
			result.WriteString("\n```\n")
		}

		// Show last part of stdout to see what was happening before failure
		if output != "" {
			result.WriteString("\n### Last Output (stdout)\n```\n")
			result.WriteString(truncateOutput(output, 100))
			result.WriteString("\n```\n")
		}
	} else {
		result.WriteString("**Status**: Success\n")
		if mlflowURL != "" {
			result.WriteString(fmt.Sprintf("**MLFlow Run**: %s\n", mlflowURL))
		}
		if metrics != "" {
			result.WriteString(fmt.Sprintf("\n### Metrics\n%s\n", metrics))
		}
	}

	return result.String()
}

func (a *ExperimentRunnerAgent) parseMLFlowInfo(output string) (runID, experimentID string) {
	// Look for run_id patterns in output
	runPatterns := []string{
		`run_id[:\s=]+([a-f0-9]{32})`,
		`mlflow\.run_id[:\s=]+([a-f0-9]{32})`,
		`Run ID: ([a-f0-9]{32})`,
	}
	for _, pattern := range runPatterns {
		re := regexp.MustCompile(pattern)
		if matches := re.FindStringSubmatch(output); len(matches) > 1 {
			runID = matches[1]
			break
		}
	}

	// Look for experiment_id patterns
	expPatterns := []string{
		`experiment_id[:\s=]+(\d+)`,
		`Experiment ID: (\d+)`,
	}
	for _, pattern := range expPatterns {
		re := regexp.MustCompile(pattern)
		if matches := re.FindStringSubmatch(output); len(matches) > 1 {
			experimentID = matches[1]
			break
		}
	}

	return runID, experimentID
}

func (a *ExperimentRunnerAgent) constructMLFlowURL(runID, experimentID string) string {
	if runID == "" {
		return ""
	}
	// Databricks MLFlow URL format: {host}/ml/experiments/{experiment_id}/runs/{run_id}
	databricksHost := os.Getenv("DATABRICKS_HOST")
	if databricksHost == "" {
		return fmt.Sprintf("MLFlow Run ID: %s (set DATABRICKS_HOST for full URL)", runID)
	}
	if experimentID == "" {
		return fmt.Sprintf("%s/ml/runs/%s", strings.TrimSuffix(databricksHost, "/"), runID)
	}
	return fmt.Sprintf("%s/ml/experiments/%s/runs/%s",
		strings.TrimSuffix(databricksHost, "/"), experimentID, runID)
}

func (a *ExperimentRunnerAgent) parseMetrics(output string) string {
	// Look for metrics in the output
	var metrics []string

	patterns := map[string]*regexp.Regexp{
		"F1":        regexp.MustCompile(`[Ff]1[:\s]+([0-9.]+)`),
		"Precision": regexp.MustCompile(`[Pp]recision[:\s]+([0-9.]+)`),
		"Recall":    regexp.MustCompile(`[Rr]ecall[:\s]+([0-9.]+)`),
	}

	for name, pattern := range patterns {
		if matches := pattern.FindStringSubmatch(output); len(matches) > 1 {
			metrics = append(metrics, fmt.Sprintf("- %s: %s", name, matches[1]))
		}
	}

	if len(metrics) == 0 {
		return ""
	}
	return strings.Join(metrics, "\n")
}

// truncateOutput returns the last N lines of output
func truncateOutput(output string, maxLines int) string {
	lines := strings.Split(output, "\n")
	if len(lines) <= maxLines {
		return strings.TrimSpace(output)
	}
	// Return last maxLines lines with indicator
	truncated := lines[len(lines)-maxLines:]
	return fmt.Sprintf("... (%d lines truncated)\n%s", len(lines)-maxLines, strings.Join(truncated, "\n"))
}

func (a *ExperimentRunnerAgent) formatOutput(files []string, experiments []string, results []string) string {
	var output strings.Builder

	output.WriteString("# Experiment Results\n\n")
	output.WriteString("## Changes Detected\n")
	output.WriteString(fmt.Sprintf("Modified files: %s\n", strings.Join(files, ", ")))
	output.WriteString(fmt.Sprintf("Running experiments: %s\n\n", strings.Join(experiments, ", ")))
	output.WriteString("---\n\n")

	for _, result := range results {
		output.WriteString(result)
		output.WriteString("\n---\n\n")
	}

	return output.String()
}

func init() {
	Register(NewExperimentRunnerAgent())
}
