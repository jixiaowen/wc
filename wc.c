/* wc.c — print delimiter, word, and byte counts for files
 *
 * Based on GNU coreutils wc.c (GPL v3+)
 * Original authors: Paul Rubin, David MacKenzie
 *
 * Extension: -d STR / --delimiter=STR
 *   Use STR (one or more bytes) as the record separator instead of newline.
 *   Escape sequences in STR: \0 \a \b \t \n \r \v \\ \xHH
 *   Multi-char examples:  -d '||'   -d $'\r\n'   -d '---'   -d '\x01\x02'
 *
 * Speed strategy:
 *   1-byte delimiter → memchr() with 15-byte adaptive heuristic (SIMD via glibc)
 *   N-byte delimiter → memmem() with (N-1)-byte carry buffer for boundary matches
 *   bytes-only path  → fstat+lseek shortcut for regular files
 *
 * Build:  gcc -O2 -Wall -Wextra -o wc wc.c
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <locale.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* =========================================================================
 * Constants
 * ========================================================================= */

#define IO_BUFSIZE    (32 * 1024)
#define MAX_DELIM_LEN 256

/* =========================================================================
 * Global state
 * ========================================================================= */

static const char *program_name;

/* Delimiter bytes and length.  Default: single '\n'. */
static unsigned char opt_delim[MAX_DELIM_LEN];
static size_t        opt_delim_len = 1;

static bool print_lines;
static bool print_words;
static bool print_chars;
static bool print_bytes;
static bool print_linelength;

static uintmax_t total_lines;
static uintmax_t total_words;
static uintmax_t total_chars;
static uintmax_t total_bytes;
static intmax_t  max_line_length;

static int  number_width = 7;
static bool have_read_stdin;

/* =========================================================================
 * Delimiter parsing
 *
 * Parses a string of literal chars and/or escape sequences into
 * opt_delim[] / opt_delim_len.
 * Supported: \0 \a \b \t \n \r \v \\  \xHH (1–2 hex digits)
 * Returns 0 on success, -1 on error.
 * ========================================================================= */

static int
parse_delimiter(const char *s)
{
    if (!s || !*s)
        return -1;

    size_t      len = 0;
    const char *p   = s;

    while (*p) {
        if (len >= MAX_DELIM_LEN)
            return -1;

        if (*p != '\\') {
            opt_delim[len++] = (unsigned char)*p++;
            continue;
        }

        /* Escape sequence — skip the backslash */
        ++p;
        switch (*p) {
        case '\0': return -1;
        case '0':  opt_delim[len++] = '\0'; ++p; break;
        case 'a':  opt_delim[len++] = '\a'; ++p; break;
        case 'b':  opt_delim[len++] = '\b'; ++p; break;
        case 't':  opt_delim[len++] = '\t'; ++p; break;
        case 'n':  opt_delim[len++] = '\n'; ++p; break;
        case 'r':  opt_delim[len++] = '\r'; ++p; break;
        case 'v':  opt_delim[len++] = '\v'; ++p; break;
        case '\\': opt_delim[len++] = '\\'; ++p; break;
        case 'x': {
            ++p;
            if (!isxdigit((unsigned char)*p)) return -1;
            unsigned v = (unsigned)(*p > '9' ? tolower(*p)-'a'+10 : *p-'0');
            ++p;
            if (isxdigit((unsigned char)*p)) {
                v = v * 16 + (unsigned)(*p > '9' ? tolower(*p)-'a'+10 : *p-'0');
                ++p;
            }
            opt_delim[len++] = (unsigned char)v;
            break;
        }
        default: return -1;
        }
    }

    if (len == 0) return -1;
    opt_delim_len = len;
    return 0;
}

/* =========================================================================
 * Usage
 * ========================================================================= */

static void
usage(int status)
{
    FILE *out = (status == EXIT_SUCCESS) ? stdout : stderr;
    fprintf(out,
"Usage: %s [OPTION]... [FILE]...\n"
"  or:  %s [OPTION]... -\n"
"\n"
"Print delimiter, word, and byte counts for each FILE, and a total line\n"
"when more than one FILE is specified.  With no FILE, or when FILE is -,\n"
"read standard input.\n"
"\n"
"  -c, --bytes              print the byte counts\n"
"  -m, --chars              print the character counts\n"
"  -l, --lines              print the record-separator counts\n"
"  -w, --words              print the word counts\n"
"  -L, --max-line-length    print the maximum display width per record\n"
"  -d, --delimiter=STR      use STR as record separator (multi-char OK)\n"
"                           Escapes: \\0 \\a \\b \\t \\n \\r \\v \\\\ \\xHH\n"
"                           e.g.  -d ','       single comma\n"
"                                 -d '||'      two pipes\n"
"                                 -d $'\\r\\n'   CRLF (in bash)\n"
"                                 -d '---'     three dashes\n"
"      --help               display this help and exit\n"
"      --version            output version information and exit\n",
            program_name, program_name);
    exit(status);
}

/* =========================================================================
 * Output
 * ========================================================================= */

static void
write_counts(uintmax_t lines, uintmax_t words, uintmax_t chars,
             uintmax_t bytes, intmax_t linelen, const char *file)
{
    const char *fmt   = "%*"  PRIuMAX;
    const char *fmtsp = " %*" PRIuMAX;
    const char *cur   = fmt;

    if (print_lines)      { printf(cur, number_width, lines); cur = fmtsp; }
    if (print_words)      { printf(cur, number_width, words); cur = fmtsp; }
    if (print_chars)      { printf(cur, number_width, chars); cur = fmtsp; }
    if (print_bytes)      { printf(cur, number_width, bytes); cur = fmtsp; }
    if (print_linelength) {
        printf((cur == fmt) ? "%*" PRIdMAX : " %*" PRIdMAX,
               number_width, linelen);
        cur = fmtsp;
    }
    (void)cur;
    if (file) printf(" %s", file);
    putchar('\n');
}

/* =========================================================================
 * process_byte — word / line-length accounting for one non-delimiter byte
 * ========================================================================= */

static inline void
process_byte(unsigned char c,
             intmax_t  *linepos,
             intmax_t  *linelength,
             uintmax_t *words,
             bool      *in_word)
{
    switch (c) {
    case '\r': case '\f': case '\n':
        if (*linepos > *linelength) *linelength = *linepos;
        *linepos = 0; *in_word = false;
        break;
    case '\t':
        *linepos += 8 - (*linepos % 8); *in_word = false;
        break;
    case ' ':
        (*linepos)++; *in_word = false;
        break;
    case '\v':
        *in_word = false;
        break;
    default:
        if (isspace(c)) {
            *in_word = false;
        } else {
            *linepos += isprint(c) ? 1 : 0;
            if (!*in_word) { (*words)++; *in_word = true; }
        }
        break;
    }
}

/* =========================================================================
 * count_lines_fast — delimiter-only counting (no word/linelen tracking)
 *
 * 1-byte:  adaptive memchr() — per-byte loop when records are dense,
 *          SIMD memchr when records are sparse (heuristic: avg >= 15 B).
 *
 * N-byte:  memmem() with a carry buffer of (N-1) bytes.
 *
 *   Buffer layout each iteration:
 *     buf[ 0 .. carry-1 ]            — tail of previous read (carry bytes)
 *     buf[ carry .. carry+n-1 ]      — freshly read bytes
 *
 *   bytes counts only freshly-read bytes; carry bytes were counted earlier.
 *   After scanning, the last min(remaining, N-1) bytes become the new carry.
 * ========================================================================= */

static bool
count_lines_fast(int fd, const char *label,
                 uintmax_t *lines_out, uintmax_t *bytes_out)
{
    uintmax_t lines = 0, bytes = 0;
    ssize_t   n;

    if (opt_delim_len == 1) {
        /* ---- Single-byte: adaptive memchr ---- */
        static char   buf[IO_BUFSIZE];
        unsigned char delim     = opt_delim[0];
        bool          use_memchr = false;

        while ((n = read(fd, buf, IO_BUFSIZE)) > 0) {
            bytes += (uintmax_t)n;
            const char *end   = buf + n;
            uintmax_t   found = 0;

            if (!use_memchr) {
                for (const char *p = buf; p < end; p++)
                    found += (unsigned char)*p == delim;
            } else {
                const char *p = buf, *hit;
                while ((hit = memchr(p, delim, (size_t)(end - p)))) {
                    ++found; p = hit + 1;
                }
            }
            use_memchr = (found == 0 || 15 * found <= (uintmax_t)n);
            lines += found;
        }

    } else {
        /* ---- Multi-byte: memmem + carry buffer ---- */
        static char buf[MAX_DELIM_LEN + IO_BUFSIZE];
        size_t carry = 0;

        while ((n = read(fd, buf + carry, IO_BUFSIZE)) > 0) {
            bytes += (uintmax_t)n;
            size_t      avail = carry + (size_t)n;
            const char *p     = buf;
            const char *end   = buf + avail;
            const char *hit;

            while ((hit = memmem(p, (size_t)(end - p),
                                 opt_delim, opt_delim_len))) {
                ++lines;
                p = hit + opt_delim_len;
            }

            /* Preserve the last (delim_len-1) bytes as carry */
            size_t rem = (size_t)(end - p);
            carry = (rem < opt_delim_len) ? rem : opt_delim_len - 1;
            memmove(buf, end - carry, carry);
        }
    }

    if (n < 0) {
        fprintf(stderr, "%s: %s: %s\n",
                program_name, label ? label : "-", strerror(errno));
        return false;
    }
    *lines_out = lines;
    *bytes_out = bytes;
    return true;
}

/* =========================================================================
 * count_bytes_fast — lseek shortcut for regular files
 * ========================================================================= */

static bool
count_bytes_fast(int fd, const char *label, uintmax_t *bytes_out)
{
    struct stat st;
    if (fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size >= 0) {
        off_t cur = lseek(fd, 0, SEEK_CUR);
        if (cur >= 0) {
            uintmax_t rem = (st.st_size > cur)
                            ? (uintmax_t)(st.st_size - cur) : 0;
            if (lseek(fd, 0, SEEK_END) >= 0) {
                *bytes_out = rem; return true;
            }
        }
    }
    static char buf[IO_BUFSIZE];
    uintmax_t bytes = 0;
    ssize_t n;
    while ((n = read(fd, buf, IO_BUFSIZE)) > 0) bytes += (uintmax_t)n;
    if (n < 0) {
        fprintf(stderr, "%s: %s: %s\n",
                program_name, label ? label : "-", strerror(errno));
        return false;
    }
    *bytes_out = bytes;
    return true;
}

/* =========================================================================
 * wc_full — single-pass: lines + words + bytes + max-line-length
 *
 * 1-byte delimiter:
 *   Classic per-byte switch-case loop (same as upstream non-SIMD branch).
 *
 * N-byte delimiter:
 *   memmem() locates each delimiter.  Bytes before each hit go to
 *   process_byte(); bytes in the carry window are held for next iteration.
 *   Carry bytes at EOF are flushed as regular content (they cannot contain
 *   a complete delimiter since carry_len < delim_len).
 *
 *   Why is it safe to hold the carry unprocessed?
 *   The word/linelen state (in_word, linepos) is preserved across
 *   iterations, so a word or visual line spanning a buffer boundary is
 *   handled correctly — the carry bytes are simply processed a little later.
 * ========================================================================= */

static bool
wc_full(int fd, const char *label,
        uintmax_t *lines_out, uintmax_t *words_out,
        uintmax_t *chars_out, uintmax_t *bytes_out,
        intmax_t  *linelen_out)
{
    uintmax_t lines = 0, words = 0, bytes = 0;
    intmax_t  linelength = 0, linepos = 0;
    bool      in_word = false, ok = true;
    ssize_t   n;

    if (opt_delim_len == 1) {
        /* ----------------------------------------------------------------
         * Single-byte delimiter: per-byte switch loop
         * --------------------------------------------------------------- */
        static char   buf[IO_BUFSIZE];
        unsigned char delim = opt_delim[0];

        while ((n = read(fd, buf, IO_BUFSIZE)) > 0) {
            bytes += (uintmax_t)n;
            const unsigned char *p   = (const unsigned char *)buf;
            const unsigned char *end = p + (size_t)n;

            while (p < end) {
                unsigned char c = *p++;
                if (c == delim) {
                    ++lines;
                    if (linepos > linelength) linelength = linepos;
                    linepos = 0; in_word = false;
                } else {
                    process_byte(c, &linepos, &linelength, &words, &in_word);
                }
            }
        }

    } else {
        /* ----------------------------------------------------------------
         * Multi-byte delimiter: memmem + carry buffer + process_byte
         * --------------------------------------------------------------- */
        static char buf[MAX_DELIM_LEN + IO_BUFSIZE];
        size_t carry = 0;

        while ((n = read(fd, buf + carry, IO_BUFSIZE)) > 0) {
            bytes += (uintmax_t)n;
            size_t avail = carry + (size_t)n;

            const unsigned char *p   = (const unsigned char *)buf;
            const unsigned char *end = p + avail;

            for (;;) {
                const char *hit = memmem(p, (size_t)(end - p),
                                         opt_delim, opt_delim_len);

                /* Safe processing boundary:
                   - delimiter found   → process up to the hit
                   - delimiter absent  → process up to (end - carry_window)
                     leaving the last (delim_len-1) bytes for the carry   */
                const unsigned char *proc_end;
                if (hit) {
                    proc_end = (const unsigned char *)hit;
                } else {
                    size_t rem  = (size_t)(end - p);
                    size_t safe = (rem >= opt_delim_len)
                                  ? rem - (opt_delim_len - 1) : 0;
                    proc_end = p + safe;
                }

                while (p < proc_end)
                    process_byte(*p++, &linepos, &linelength,
                                 &words, &in_word);

                if (!hit) break;

                /* Delimiter hit → record boundary */
                ++lines;
                if (linepos > linelength) linelength = linepos;
                linepos = 0; in_word = false;
                p = (const unsigned char *)hit + opt_delim_len;
            }

            size_t rem = (size_t)(end - p);
            carry = (rem < opt_delim_len) ? rem : opt_delim_len - 1;
            memmove(buf, end - carry, carry);
        }

        /* EOF — flush carry: no complete delimiter possible here */
        {
            const unsigned char *p   = (const unsigned char *)buf;
            const unsigned char *end = p + carry;
            while (p < end)
                process_byte(*p++, &linepos, &linelength, &words, &in_word);
        }
    }

    if (linepos > linelength) linelength = linepos;

    if (n < 0) {
        fprintf(stderr, "%s: %s: %s\n",
                program_name, label ? label : "-", strerror(errno));
        ok = false;
    }

    *lines_out   = lines;
    *words_out   = words;
    *chars_out   = bytes;   /* chars == bytes in single-byte locale */
    *bytes_out   = bytes;
    *linelen_out = linelength;
    return ok;
}

/* =========================================================================
 * wc_fd — choose the cheapest counting path, then accumulate totals
 * ========================================================================= */

static bool
wc_fd(int fd, const char *filename)
{
    uintmax_t lines = 0, words = 0, chars = 0, bytes = 0;
    intmax_t  linelen = 0;
    bool ok;

#ifdef POSIX_FADV_SEQUENTIAL
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif

    if (print_bytes && !print_lines && !print_words
            && !print_chars && !print_linelength) {
        ok = count_bytes_fast(fd, filename, &bytes);

    } else if (print_lines && !print_words
               && !print_chars && !print_linelength) {
        ok = count_lines_fast(fd, filename, &lines, &bytes);

    } else {
        ok = wc_full(fd, filename,
                     &lines, &words, &chars, &bytes, &linelen);
        if (!print_chars) chars = bytes;
    }

    write_counts(lines, words, chars, bytes, linelen, filename);

    total_lines += lines;
    total_words += words;
    total_chars += chars;
    total_bytes += bytes;
    if (linelen > max_line_length) max_line_length = linelen;

    return ok;
}

static bool
wc_file(const char *filename)
{
    if (!filename || strcmp(filename, "-") == 0) {
        have_read_stdin = true;
        return wc_fd(STDIN_FILENO, NULL);
    }
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "%s: %s: %s\n",
                program_name, filename, strerror(errno));
        return false;
    }
    bool ok = wc_fd(fd, filename);
    if (close(fd) != 0) {
        fprintf(stderr, "%s: %s: %s\n",
                program_name, filename, strerror(errno));
        return false;
    }
    return ok;
}

/* =========================================================================
 * Column-width calculation
 * ========================================================================= */

static int
compute_number_width(int nfiles, char * const *files)
{
    if (nfiles <= 1) return 1;

    uintmax_t max_size = 0;
    for (int i = 0; i < nfiles; i++) {
        struct stat st;
        if (stat(files[i], &st) == 0 && S_ISREG(st.st_mode)
                && st.st_size > 0
                && (uintmax_t)st.st_size > max_size)
            max_size = (uintmax_t)st.st_size;
    }
    int w = 1;
    for (uintmax_t v = max_size; v >= 10; v /= 10) ++w;
    return (w < 7) ? 7 : w;
}

/* =========================================================================
 * main
 * ========================================================================= */

enum { OPT_HELP = CHAR_MAX + 1, OPT_VERSION };

static const struct option longopts[] = {
    { "bytes",           no_argument,       NULL, 'c'         },
    { "chars",           no_argument,       NULL, 'm'         },
    { "lines",           no_argument,       NULL, 'l'         },
    { "words",           no_argument,       NULL, 'w'         },
    { "max-line-length", no_argument,       NULL, 'L'         },
    { "delimiter",       required_argument, NULL, 'd'         },
    { "help",            no_argument,       NULL, OPT_HELP    },
    { "version",         no_argument,       NULL, OPT_VERSION },
    { NULL, 0, NULL, 0 }
};

int
main(int argc, char **argv)
{
    program_name = argv[0];
    setlocale(LC_ALL, "");

    /* Default delimiter = '\n' */
    opt_delim[0]  = '\n';
    opt_delim_len = 1;

    int c;
    while ((c = getopt_long(argc, argv, "cmlwLd:", longopts, NULL)) != -1) {
        switch (c) {
        case 'c': print_bytes      = true; break;
        case 'm': print_chars      = true; break;
        case 'l': print_lines      = true; break;
        case 'w': print_words      = true; break;
        case 'L': print_linelength = true; break;
        case 'd':
            if (parse_delimiter(optarg) != 0) {
                fprintf(stderr,
                        "%s: invalid delimiter '%s'\n"
                        "Try '%s --help' for more information.\n",
                        program_name, optarg, program_name);
                return EXIT_FAILURE;
            }
            break;
        case OPT_HELP:    usage(EXIT_SUCCESS);
        case OPT_VERSION:
            printf("wc (custom, multi-char delimiter support) 2.0\n"
                   "Based on GNU coreutils wc (GPL v3+)\n");
            return EXIT_SUCCESS;
        default:
            usage(EXIT_FAILURE);
        }
    }

    /* Default output: lines + words + bytes */
    if (!(print_lines || print_words || print_chars
              || print_bytes || print_linelength))
        print_lines = print_words = print_bytes = true;

    int    nfiles = argc - optind;
    char **files  = argv + optind;

    number_width = compute_number_width(nfiles, files);
    setvbuf(stdout, NULL, _IOLBF, 0);

    bool ok = true;

    if (nfiles == 0) {
        ok = wc_file(NULL);
    } else {
        for (int i = 0; i < nfiles; i++)
            ok &= wc_file(files[i]);

        if (nfiles > 1)
            write_counts(total_lines, total_words, total_chars,
                         total_bytes, max_line_length, "total");
    }

    if (have_read_stdin && close(STDIN_FILENO) != 0) {
        fprintf(stderr, "%s: closing stdin: %s\n",
                program_name, strerror(errno));
        ok = false;
    }

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
