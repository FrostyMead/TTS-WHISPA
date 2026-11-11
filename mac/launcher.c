/*
 * Murmr.app's executable (Contents/MacOS/Murmr).
 *
 * macOS grants Microphone, Input Monitoring and Accessibility to the app that is
 * "responsible" for a process. A shell script that execs Python makes the responsible
 * process Python itself, so Murmr never appears in Privacy & Security. This launcher
 * stays alive as Python's parent instead: Python inherits Murmr.app as its responsible
 * app, and the permissions you grant to Murmr apply to it.
 *
 * It runs the command in Contents/Resources/command (one argument per line), adds the
 * arguments it was given (e.g. from `open --args`), forwards quit signals, and exits
 * with Python's exit code.
 *
 * Build (universal):  clang -O2 -arch arm64 -arch x86_64 -mmacosx-version-min=11.0 \
 *                         -o whispa/resources/MurmrLauncher mac/launcher.c
 */
#include <errno.h>
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
static pid_t child = 0;

static void forward(int sig) {
    if (child > 0) kill(child, sig);
}

int main(int argc, char **argv) {
    char exe[PATH_MAX], real[PATH_MAX], path[PATH_MAX];
    uint32_t size = sizeof exe;
    if (_NSGetExecutablePath(exe, &size) != 0 || !realpath(exe, real)) return 70;
    snprintf(path, sizeof path, "%s/../Resources/command", dirname(real));

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Murmr: can't read %s: %s\n", path, strerror(errno));
        return 66;
    }
    static char buf[65536];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;

    char *args[256];
    int k = 0;
    for (char *line = strtok(buf, "\n"); line && k < 200; line = strtok(NULL, "\n"))
        if (*line) args[k++] = line;
    if (k == 0) return 66;
    for (int i = 1; i < argc && k < 255; i++)
        if (strncmp(argv[i], "-psn_", 5) != 0) args[k++] = argv[i];  /* old Finder argument */
    args[k] = NULL;

    setenv("WHISPA_BUNDLE", "1", 1);  /* tells Murmr it was started through the app */
    signal(SIGTERM, forward);
    signal(SIGINT, forward);
    signal(SIGHUP, forward);
    int err = posix_spawn(&child, args[0], NULL, NULL, args, environ);
    if (err) {
        fprintf(stderr, "Murmr: can't start %s: %s\n", args[0], strerror(err));
        return 69;
    }
    int status;
    while (waitpid(child, &status, 0) < 0)
        if (errno != EINTR) return 71;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
