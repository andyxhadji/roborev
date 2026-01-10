package agent

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/wesm/roborev/internal/git"
)

// ExperimentRunnerAgent runs ML experiments based on commit changes
type ExperimentRunnerAgent struct{}

// NewExperimentRunnerAgent creates a new experiment runner agent
func NewExperimentRunnerAgent() *ExperimentRunnerAgent {
	return &ExperimentRunnerAgent{}
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

// getLogFile returns a file handle for writing live logs
func (a *ExperimentRunnerAgent) getLogFile(repoPath string) *os.File {
	logDir := filepath.Join(os.TempDir(), "roborev-logs")
	os.MkdirAll(logDir, 0755)

	// Use repo name as part of log filename for uniqueness
	repoName := filepath.Base(repoPath)
	logPath := filepath.Join(logDir, repoName+".log")

	f, err := os.Create(logPath)
	if err != nil {
		return nil
	}
	return f
}

// GetLogPath returns the path to the log file for a repo (used by API)
func GetLogPath(repoPath string) string {
	logDir := filepath.Join(os.TempDir(), "roborev-logs")
	repoName := filepath.Base(repoPath)
	return filepath.Join(logDir, repoName+".log")
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
	// Hard-coded to use "poetry run python" for ML experiment workflows
	cmd := exec.CommandContext(ctx, "poetry", "run", "python", experimentPath)
	cmd.Dir = repoPath

	var stdout, stderr bytes.Buffer

	// Also write to log file for live viewing
	logFile := a.getLogFile(repoPath)
	if logFile != nil {
		defer logFile.Close()
		cmd.Stdout = io.MultiWriter(&stdout, logFile)
		cmd.Stderr = io.MultiWriter(&stderr, logFile)
	} else {
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
	}

	err := cmd.Run()

	output := stdout.String()
	errOutput := stderr.String()

	// Parse evaluation details (tables, reports, etc.)
	evalDetails := a.parseEvaluationDetails(output)

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
		if evalDetails != "" {
			result.WriteString(fmt.Sprintf("\n### Evaluation Results\n%s\n", evalDetails))
		}
	}

	return result.String()
}

func (a *ExperimentRunnerAgent) parseMLFlowInfo(output string) (runID, experimentID string) {
	// Look for run_id patterns in output (supports both 32-char hex and UUID formats)
	runPatterns := []string{
		`Run ID: ([a-f0-9-]{32,36})`,
		`MLflow run: ([a-f0-9]{32})`,
		`run_id[:\s=]+([a-f0-9-]{32,36})`,
		`mlflow\.run_id[:\s=]+([a-f0-9-]{32,36})`,
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
		`Experiment ID: (\d+)`,
		`experiment_id[:\s=]+(\d+)`,
		`/experiments/(\d+)/runs/`,
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

// parseMLFlowURL extracts a direct MLFlow URL from output if available
func (a *ExperimentRunnerAgent) parseMLFlowURL(output string) string {
	// Look for direct URL patterns in output
	// Match URLs ending with /experiments/{id}/runs/{run_id} or /#/experiments/{id}/runs/{run_id}
	urlPatterns := []string{
		`Databricks experiment URL: (https://[^\s]+)`,
		`View run [^\s]+ at: (https://[^\s]+)`,
		`MLFlow URL: (https://[^\s]+)`,
	}
	for _, pattern := range urlPatterns {
		re := regexp.MustCompile(pattern)
		if matches := re.FindStringSubmatch(output); len(matches) > 1 {
			// Clean any trailing newlines or whitespace from the URL
			url := strings.TrimSpace(matches[1])
			// Verify it looks like an MLFlow URL
			if strings.Contains(url, "/experiments/") && strings.Contains(url, "/runs/") {
				return url
			}
		}
	}
	return ""
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
		"F1":        regexp.MustCompile(`[Ff]1[_\-\s]*[Ss]core[:\s=]+([0-9.]+)|[Ff]1[:\s=]+([0-9.]+)`),
		"Precision": regexp.MustCompile(`[Pp]recision[:\s=]+([0-9.]+)`),
		"Recall":    regexp.MustCompile(`[Rr]ecall[:\s=]+([0-9.]+)`),
		"Accuracy":  regexp.MustCompile(`[Aa]ccuracy[:\s=]+([0-9.]+)`),
		"AUC":       regexp.MustCompile(`[Aa][Uu][Cc][:\s=]+([0-9.]+)`),
		"Loss":      regexp.MustCompile(`[Ll]oss[:\s=]+([0-9.]+)`),
	}

	for name, pattern := range patterns {
		if matches := pattern.FindStringSubmatch(output); len(matches) > 1 {
			// Find first non-empty capture group
			for _, m := range matches[1:] {
				if m != "" {
					metrics = append(metrics, fmt.Sprintf("- %s: %s", name, m))
					break
				}
			}
		}
	}

	if len(metrics) == 0 {
		return ""
	}
	return strings.Join(metrics, "\n")
}

// parseEvaluationDetails extracts detailed evaluation information from output
func (a *ExperimentRunnerAgent) parseEvaluationDetails(output string) string {
	var details []string

	// Look for confusion matrix or classification report sections
	lines := strings.Split(output, "\n")
	inEvalSection := false
	evalLines := []string{}

	for _, line := range lines {
		lower := strings.ToLower(line)
		trimmed := strings.TrimSpace(line)

		// Check if this is a "=== Section ===" header
		isHeaderSection := strings.HasPrefix(trimmed, "=== ") && strings.HasSuffix(trimmed, " ===")

		// Start capturing evaluation sections
		if strings.Contains(lower, "classification report") ||
			strings.Contains(lower, "confusion matrix") ||
			strings.Contains(lower, "evaluation results") ||
			strings.Contains(lower, "test results") ||
			strings.Contains(lower, "validation results") ||
			strings.Contains(lower, "evaluating cohort") ||
			isHeaderSection {

			// If we were already in a section, save it first
			if inEvalSection && len(evalLines) > 1 {
				details = append(details, strings.Join(evalLines, "\n"))
				evalLines = []string{}
			}

			inEvalSection = true
			evalLines = append(evalLines, line)
			continue
		}

		// Continue capturing if in section (until empty line or new section)
		if inEvalSection {
			// Check if line looks like a markdown table row or separator
			isTableLine := strings.Contains(line, "|") ||
			               (strings.Contains(line, "---") && strings.Contains(line, "-")) ||
			               regexp.MustCompile(`^\s*[a-zA-Z_]+\s*\|`).MatchString(line)

			if strings.TrimSpace(line) == "" && len(evalLines) > 3 && !isTableLine {
				// End of section due to empty line (but not if we're in a table)
				details = append(details, strings.Join(evalLines, "\n"))
				evalLines = []string{}
				inEvalSection = false
			} else {
				evalLines = append(evalLines, line)
			}
		}
	}

	// Capture any remaining eval lines
	if len(evalLines) > 1 {
		details = append(details, strings.Join(evalLines, "\n"))
	}

	// Also look for specific metric summaries
	summaryPatterns := []*regexp.Regexp{
		regexp.MustCompile(`(?i)(total|overall|macro|micro|weighted)[^\n]*(?:f1|precision|recall|accuracy)[^\n]*`),
		regexp.MustCompile(`(?i)samples[:\s]+\d+`),
		regexp.MustCompile(`(?i)support[:\s]+\d+`),
	}

	for _, pattern := range summaryPatterns {
		if matches := pattern.FindAllString(output, -1); len(matches) > 0 {
			for _, m := range matches {
				details = append(details, m)
			}
		}
	}

	if len(details) == 0 {
		return ""
	}

	return "```\n" + strings.Join(details, "\n") + "\n```"
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

	for _, result := range results {
		output.WriteString(result)
		output.WriteString("\n---\n\n")
	}

	return output.String()
}

// summarizeChanges returns a concise summary of changed files by directory
func summarizeChanges(files []string) string {
	if len(files) == 0 {
		return "No files changed\n"
	}

	// Count files by top-level directory
	dirCounts := make(map[string]int)
	for _, f := range files {
		parts := strings.SplitN(f, "/", 2)
		dir := parts[0]
		if len(parts) > 1 {
			dir += "/"
		}
		dirCounts[dir]++
	}

	// Format summary
	var parts []string
	for dir, count := range dirCounts {
		if count == 1 {
			parts = append(parts, fmt.Sprintf("%s (1 file)", dir))
		} else {
			parts = append(parts, fmt.Sprintf("%s (%d files)", dir, count))
		}
	}

	return fmt.Sprintf("%d files changed: %s\n", len(files), strings.Join(parts, ", "))
}

func init() {
	Register(NewExperimentRunnerAgent())
}
