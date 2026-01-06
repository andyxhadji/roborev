package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/spf13/cobra"
	"github.com/wesm/roborev/internal/daemon"
	"github.com/wesm/roborev/internal/storage"
	"github.com/wesm/roborev/internal/version"
)

// TUI styles
var (
	tuiTitleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("205")).
			MarginBottom(1)

	tuiStatusStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("241"))

	tuiSelectedStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("212"))

	tuiQueuedStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("226")) // Yellow
	tuiRunningStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("33"))  // Blue
	tuiDoneStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("46"))  // Green
	tuiFailedStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("196")) // Red

	tuiHelpStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("241")).
			MarginTop(1)
)

type tuiView int

const (
	tuiViewQueue tuiView = iota
	tuiViewReview
	tuiViewPrompt
)

type tuiModel struct {
	serverAddr      string
	daemonVersion   string
	jobs            []storage.ReviewJob
	status          storage.DaemonStatus
	selectedIdx     int
	currentView     tuiView
	currentReview   *storage.Review
	reviewScroll    int
	promptScroll    int
	promptFromQueue bool // true if prompt view was entered from queue (not review)
	width           int
	height          int
	err             error
}

type tuiTickMsg time.Time
type tuiJobsMsg []storage.ReviewJob
type tuiStatusMsg storage.DaemonStatus
type tuiReviewMsg *storage.Review
type tuiPromptMsg *storage.Review
type tuiErrMsg error

func newTuiModel(serverAddr string) tuiModel {
	// Get daemon version from runtime info
	daemonVersion := "unknown"
	if info, err := daemon.ReadRuntime(); err == nil {
		daemonVersion = info.Version
	}

	return tuiModel{
		serverAddr:    serverAddr,
		daemonVersion: daemonVersion,
		jobs:          []storage.ReviewJob{},
		currentView:   tuiViewQueue,
		width:         80, // sensible defaults until we get WindowSizeMsg
		height:        24,
	}
}

func (m tuiModel) Init() tea.Cmd {
	return tea.Batch(
		tea.WindowSize(), // request initial window size
		m.tick(),
		m.fetchJobs(),
		m.fetchStatus(),
	)
}

func (m tuiModel) tick() tea.Cmd {
	return tea.Tick(2*time.Second, func(t time.Time) tea.Msg {
		return tuiTickMsg(t)
	})
}

func (m tuiModel) fetchJobs() tea.Cmd {
	return func() tea.Msg {
		resp, err := http.Get(m.serverAddr + "/api/jobs?limit=50")
		if err != nil {
			return tuiErrMsg(err)
		}
		defer resp.Body.Close()

		var result struct {
			Jobs []storage.ReviewJob `json:"jobs"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return tuiErrMsg(err)
		}
		return tuiJobsMsg(result.Jobs)
	}
}

func (m tuiModel) fetchStatus() tea.Cmd {
	return func() tea.Msg {
		resp, err := http.Get(m.serverAddr + "/api/status")
		if err != nil {
			return tuiErrMsg(err)
		}
		defer resp.Body.Close()

		var status storage.DaemonStatus
		if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
			return tuiErrMsg(err)
		}
		return tuiStatusMsg(status)
	}
}

func (m tuiModel) fetchReview(jobID int64) tea.Cmd {
	return func() tea.Msg {
		resp, err := http.Get(fmt.Sprintf("%s/api/review?job_id=%d", m.serverAddr, jobID))
		if err != nil {
			return tuiErrMsg(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode == http.StatusNotFound {
			return tuiErrMsg(fmt.Errorf("no review found"))
		}

		var review storage.Review
		if err := json.NewDecoder(resp.Body).Decode(&review); err != nil {
			return tuiErrMsg(err)
		}
		return tuiReviewMsg(&review)
	}
}

func (m tuiModel) fetchReviewForPrompt(jobID int64) tea.Cmd {
	return func() tea.Msg {
		resp, err := http.Get(fmt.Sprintf("%s/api/review?job_id=%d", m.serverAddr, jobID))
		if err != nil {
			return tuiErrMsg(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode == http.StatusNotFound {
			return tuiErrMsg(fmt.Errorf("no review found"))
		}

		var review storage.Review
		if err := json.NewDecoder(resp.Body).Decode(&review); err != nil {
			return tuiErrMsg(err)
		}
		return tuiPromptMsg(&review)
	}
}

func (m tuiModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			if m.currentView == tuiViewReview {
				m.currentView = tuiViewQueue
				m.currentReview = nil
				m.reviewScroll = 0
				return m, nil
			}
			if m.currentView == tuiViewPrompt {
				// Go back to where we came from
				if m.promptFromQueue {
					m.currentView = tuiViewQueue
					m.currentReview = nil
					m.promptScroll = 0
				} else {
					m.currentView = tuiViewReview
					m.promptScroll = 0
				}
				return m, nil
			}
			return m, tea.Quit

		case "up", "k":
			if m.currentView == tuiViewQueue {
				if m.selectedIdx > 0 {
					m.selectedIdx--
				}
			} else if m.currentView == tuiViewReview {
				if m.reviewScroll > 0 {
					m.reviewScroll--
				}
			} else if m.currentView == tuiViewPrompt {
				if m.promptScroll > 0 {
					m.promptScroll--
				}
			}

		case "down", "j":
			if m.currentView == tuiViewQueue {
				if m.selectedIdx < len(m.jobs)-1 {
					m.selectedIdx++
				}
			} else if m.currentView == tuiViewReview {
				m.reviewScroll++
			} else if m.currentView == tuiViewPrompt {
				m.promptScroll++
			}

		case "pgup":
			pageSize := max(1, m.height-10)
			if m.currentView == tuiViewQueue {
				if len(m.jobs) > 0 {
					m.selectedIdx = max(0, m.selectedIdx-pageSize)
				}
			} else if m.currentView == tuiViewReview {
				m.reviewScroll = max(0, m.reviewScroll-pageSize)
			} else if m.currentView == tuiViewPrompt {
				m.promptScroll = max(0, m.promptScroll-pageSize)
			}

		case "pgdown":
			pageSize := max(1, m.height-10)
			if m.currentView == tuiViewQueue {
				if len(m.jobs) > 0 {
					m.selectedIdx = min(len(m.jobs)-1, m.selectedIdx+pageSize)
				}
			} else if m.currentView == tuiViewReview {
				m.reviewScroll += pageSize
			} else if m.currentView == tuiViewPrompt {
				m.promptScroll += pageSize
			}

		case "enter":
			if m.currentView == tuiViewQueue && len(m.jobs) > 0 {
				job := m.jobs[m.selectedIdx]
				if job.Status == storage.JobStatusDone {
					return m, m.fetchReview(job.ID)
				} else if job.Status == storage.JobStatusFailed {
					// Show error inline for failed jobs
					m.currentReview = &storage.Review{
						Agent:  job.Agent,
						Output: "Job failed:\n\n" + job.Error,
						Job:    &job,
					}
					m.currentView = tuiViewReview
					m.reviewScroll = 0
					return m, nil
				}
			}

		case "p":
			if m.currentView == tuiViewQueue && len(m.jobs) > 0 {
				job := m.jobs[m.selectedIdx]
				if job.Status == storage.JobStatusDone {
					// Fetch review and go directly to prompt view
					m.promptFromQueue = true
					return m, m.fetchReviewForPrompt(job.ID)
				} else if job.Status == storage.JobStatusRunning && job.Prompt != "" {
					// Show prompt from job directly for running jobs
					m.currentReview = &storage.Review{
						Agent:  job.Agent,
						Prompt: job.Prompt,
						Job:    &job,
					}
					m.currentView = tuiViewPrompt
					m.promptScroll = 0
					m.promptFromQueue = true
					return m, nil
				}
			} else if m.currentView == tuiViewReview && m.currentReview != nil && m.currentReview.Prompt != "" {
				m.currentView = tuiViewPrompt
				m.promptScroll = 0
				m.promptFromQueue = false
			} else if m.currentView == tuiViewPrompt {
				// Toggle back: go to review if came from review, queue if came from queue
				if m.promptFromQueue {
					m.currentView = tuiViewQueue
					m.currentReview = nil
					m.promptScroll = 0
				} else {
					m.currentView = tuiViewReview
					m.promptScroll = 0
				}
			}

		case "esc":
			if m.currentView == tuiViewReview {
				m.currentView = tuiViewQueue
				m.currentReview = nil
				m.reviewScroll = 0
			} else if m.currentView == tuiViewPrompt {
				// Go back to where we came from
				if m.promptFromQueue {
					m.currentView = tuiViewQueue
					m.currentReview = nil
					m.promptScroll = 0
				} else {
					m.currentView = tuiViewReview
					m.promptScroll = 0
				}
			}
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

	case tuiTickMsg:
		return m, tea.Batch(m.tick(), m.fetchJobs(), m.fetchStatus())

	case tuiJobsMsg:
		m.jobs = msg
		if m.selectedIdx >= len(m.jobs) {
			m.selectedIdx = max(0, len(m.jobs)-1)
		}

	case tuiStatusMsg:
		m.status = storage.DaemonStatus(msg)

	case tuiReviewMsg:
		m.currentReview = msg
		m.currentView = tuiViewReview
		m.reviewScroll = 0

	case tuiPromptMsg:
		m.currentReview = msg
		m.currentView = tuiViewPrompt
		m.promptScroll = 0

	case tuiErrMsg:
		m.err = msg
	}

	return m, nil
}

func (m tuiModel) View() string {
	if m.currentView == tuiViewPrompt && m.currentReview != nil {
		return m.renderPromptView()
	}
	if m.currentView == tuiViewReview && m.currentReview != nil {
		return m.renderReviewView()
	}
	return m.renderQueueView()
}

func (m tuiModel) renderQueueView() string {
	var b strings.Builder

	// Title with version info
	b.WriteString(tuiTitleStyle.Render(fmt.Sprintf("RoboRev Queue (cli: %s, daemon: %s)", version.Version, m.daemonVersion)))
	b.WriteString("\n")

	// Status line
	statusLine := fmt.Sprintf("Workers: %d/%d | Queued: %d | Running: %d | Done: %d | Failed: %d | Size: %dx%d",
		m.status.ActiveWorkers, m.status.MaxWorkers,
		m.status.QueuedJobs, m.status.RunningJobs,
		m.status.CompletedJobs, m.status.FailedJobs,
		m.width, m.height)
	b.WriteString(tuiStatusStyle.Render(statusLine))
	b.WriteString("\n\n")

	if len(m.jobs) == 0 {
		b.WriteString("No jobs in queue\n")
	} else {
		// Header (with 2-char prefix to align with row selector)
		header := fmt.Sprintf("  %-4s %-17s %-15s %-8s %-8s %-19s %s",
			"ID", "Ref", "Repo", "Agent", "Status", "Queued", "Elapsed")
		b.WriteString(tuiStatusStyle.Render(header))
		b.WriteString("\n")
		b.WriteString("  " + strings.Repeat("-", min(m.width-4, 78)))
		b.WriteString("\n")

		// Calculate visible job range based on terminal height
		// Reserve lines for: title(2) + status(2) + header(2) + help(2) + scroll indicator(1)
		reservedLines := 9
		visibleJobs := m.height - reservedLines
		if visibleJobs < 3 {
			visibleJobs = 3 // Show at least 3 jobs
		}

		// Determine which jobs to show, keeping selected item visible
		start := 0
		end := len(m.jobs)

		if len(m.jobs) > visibleJobs {
			// Center the selected item when possible
			start = m.selectedIdx - visibleJobs/2
			if start < 0 {
				start = 0
			}
			end = start + visibleJobs
			if end > len(m.jobs) {
				end = len(m.jobs)
				start = end - visibleJobs
			}
		}

		// Jobs
		for i := start; i < end; i++ {
			job := m.jobs[i]
			line := m.renderJobLine(job)
			if i == m.selectedIdx {
				line = tuiSelectedStyle.Render("> " + line)
			} else {
				line = "  " + line
			}
			b.WriteString(line)
			b.WriteString("\n")
		}

		// Show scroll indicator if not all jobs visible
		if len(m.jobs) > visibleJobs {
			scrollInfo := fmt.Sprintf("[showing %d-%d of %d]", start+1, end, len(m.jobs))
			b.WriteString(tuiStatusStyle.Render(scrollInfo))
			b.WriteString("\n")
		}
	}

	// Help
	b.WriteString(tuiHelpStyle.Render("up/down/pgup/pgdn: navigate | enter: review | p: prompt | q: quit"))

	return b.String()
}

func (m tuiModel) renderJobLine(job storage.ReviewJob) string {
	ref := shortRef(job.GitRef)

	repo := job.RepoName
	if len(repo) > 15 {
		repo = repo[:12] + "..."
	}

	agent := job.Agent
	if len(agent) > 8 {
		agent = agent[:8]
	}

	// Format enqueue time as timestamp in local time
	enqueued := job.EnqueuedAt.Local().Format("2006-01-02 15:04:05")

	// Format elapsed time
	elapsed := ""
	if job.StartedAt != nil {
		if job.FinishedAt != nil {
			elapsed = job.FinishedAt.Sub(*job.StartedAt).Round(time.Second).String()
		} else {
			elapsed = time.Since(*job.StartedAt).Round(time.Second).String()
		}
	}

	// Color the status, then pad to fixed width (lipgloss strips trailing spaces)
	status := string(job.Status)
	var styledStatus string
	switch job.Status {
	case storage.JobStatusQueued:
		styledStatus = tuiQueuedStyle.Render(status)
	case storage.JobStatusRunning:
		styledStatus = tuiRunningStyle.Render(status)
	case storage.JobStatusDone:
		styledStatus = tuiDoneStyle.Render(status)
	case storage.JobStatusFailed:
		styledStatus = tuiFailedStyle.Render(status)
	default:
		styledStatus = status
	}
	// Pad after coloring since lipgloss strips trailing spaces
	padding := 8 - len(status)
	if padding > 0 {
		styledStatus += strings.Repeat(" ", padding)
	}

	return fmt.Sprintf("%-4d %-17s %-15s %-8s %s %-19s %s",
		job.ID, ref, repo, agent, styledStatus, enqueued, elapsed)
}

// wrapText wraps text to the specified width, preserving existing line breaks
// and breaking at word boundaries when possible
func wrapText(text string, width int) []string {
	if width <= 0 {
		width = 100
	}

	var result []string
	for _, line := range strings.Split(text, "\n") {
		if len(line) <= width {
			result = append(result, line)
			continue
		}

		// Wrap long lines
		for len(line) > width {
			// Find a good break point (space) near the width
			breakPoint := width
			for i := width; i > width/2; i-- {
				if i < len(line) && line[i] == ' ' {
					breakPoint = i
					break
				}
			}

			result = append(result, line[:breakPoint])
			line = strings.TrimLeft(line[breakPoint:], " ")
		}
		if len(line) > 0 {
			result = append(result, line)
		}
	}

	return result
}

func (m tuiModel) renderReviewView() string {
	var b strings.Builder

	review := m.currentReview
	if review.Job != nil {
		ref := shortRef(review.Job.GitRef)
		title := fmt.Sprintf("Review: %s (%s)", ref, review.Agent)
		b.WriteString(tuiTitleStyle.Render(title))
	} else {
		b.WriteString(tuiTitleStyle.Render("Review"))
	}
	b.WriteString("\n")

	// Wrap text to terminal width (max 100 chars)
	wrapWidth := min(m.width-2, 100)
	lines := wrapText(review.Output, wrapWidth)

	visibleLines := m.height - 5 // Leave room for title and help

	start := m.reviewScroll
	if start >= len(lines) {
		start = max(0, len(lines)-1)
	}
	end := min(start+visibleLines, len(lines))

	for i := start; i < end; i++ {
		b.WriteString(lines[i])
		b.WriteString("\n")
	}

	// Scroll indicator
	if len(lines) > visibleLines {
		scrollInfo := fmt.Sprintf("[%d-%d of %d lines]", start+1, end, len(lines))
		b.WriteString(tuiStatusStyle.Render(scrollInfo))
		b.WriteString("\n")
	}

	b.WriteString(tuiHelpStyle.Render("up/down: scroll | p: view prompt | esc/q: back"))

	return b.String()
}

func (m tuiModel) renderPromptView() string {
	var b strings.Builder

	review := m.currentReview
	if review.Job != nil {
		ref := shortRef(review.Job.GitRef)
		title := fmt.Sprintf("Prompt: %s (%s)", ref, review.Agent)
		b.WriteString(tuiTitleStyle.Render(title))
	} else {
		b.WriteString(tuiTitleStyle.Render("Prompt"))
	}
	b.WriteString("\n")

	// Wrap text to terminal width (max 100 chars)
	wrapWidth := min(m.width-2, 100)
	lines := wrapText(review.Prompt, wrapWidth)

	visibleLines := m.height - 5 // Leave room for title and help

	start := m.promptScroll
	if start >= len(lines) {
		start = max(0, len(lines)-1)
	}
	end := min(start+visibleLines, len(lines))

	for i := start; i < end; i++ {
		b.WriteString(lines[i])
		b.WriteString("\n")
	}

	// Scroll indicator
	if len(lines) > visibleLines {
		scrollInfo := fmt.Sprintf("[%d-%d of %d lines]", start+1, end, len(lines))
		b.WriteString(tuiStatusStyle.Render(scrollInfo))
		b.WriteString("\n")
	}

	b.WriteString(tuiHelpStyle.Render("up/down: scroll | p: back to review | esc/q: back"))

	return b.String()
}

func tuiCmd() *cobra.Command {
	var addr string

	cmd := &cobra.Command{
		Use:   "tui",
		Short: "Interactive terminal UI for monitoring reviews",
		RunE: func(cmd *cobra.Command, args []string) error {
			// Ensure daemon is running (and restart if version mismatch)
			if err := ensureDaemon(); err != nil {
				return fmt.Errorf("daemon error: %w", err)
			}

			if addr == "" {
				addr = getDaemonAddr()
			} else if !strings.HasPrefix(addr, "http://") && !strings.HasPrefix(addr, "https://") {
				addr = "http://" + addr
			}
			p := tea.NewProgram(newTuiModel(addr), tea.WithAltScreen())
			if _, err := p.Run(); err != nil {
				return fmt.Errorf("TUI error: %w", err)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&addr, "addr", "", "daemon address (default: auto-detect)")

	return cmd
}
