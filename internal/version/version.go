package version

import (
	"runtime/debug"
	"strings"
)

// Version returns the build version, automatically detected from VCS info
// embedded by Go 1.18+ when building from a git repository.
var Version = getVersion()

func getVersion() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "dev"
	}

	var revision string
	var modified bool

	for _, setting := range info.Settings {
		switch setting.Key {
		case "vcs.revision":
			revision = setting.Value
		case "vcs.modified":
			modified = setting.Value == "true"
		}
	}

	if revision == "" {
		return "dev"
	}

	// Use short hash
	if len(revision) > 7 {
		revision = revision[:7]
	}

	// Mark dirty builds
	if modified {
		revision += "-dirty"
	}

	return revision
}

// Full returns the full version string with additional build info
func Full() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "dev"
	}

	var parts []string
	parts = append(parts, Version)

	for _, setting := range info.Settings {
		if setting.Key == "vcs.time" {
			parts = append(parts, setting.Value)
			break
		}
	}

	return strings.Join(parts, " ")
}
